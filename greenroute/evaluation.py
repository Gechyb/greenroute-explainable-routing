"""Offline evaluation of GreenRoute on the labeled benchmark.

Protocol
  * Leave-one-out evidence: each question is routed using the other 999 as the
    evidence pool, so it never sees its own labels.
  * Dev/test split (30/70, fixed seed): thresholds are tuned on dev questions
    only; every headline number is reported on test questions.
  * Compute: FLOPs proxy = 2 x parameters x (prompt + output tokens), using the
    tokens each model actually processed on each question during labeling.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import MODELS, TIERS
from .data import LABELS, load_questions, question_text
from .embed import embed_cached

PARAMS = np.array([m["params_b"] for m in MODELS]) * 1e9
SEED = 42
DEV_FRACTION = 0.30
MAX_ACC_DROP = 0.02  # tuning target: within 2 points of always-largest accuracy


@dataclass
class Bench:
    ids: list
    sources: np.ndarray
    texts: list
    emb: np.ndarray
    correct: np.ndarray   # [n, T] bool
    flops: np.ndarray     # [n, T]
    latency: np.ndarray   # [n, T] seconds
    sims: np.ndarray      # [n, n] cosine, diagonal = -inf
    order: np.ndarray     # [n, n-1] neighbour indices by similarity
    dev: np.ndarray       # bool mask
    test: np.ndarray      # bool mask


def load_bench() -> Bench:
    qs = load_questions()
    ids = [q["id"] for q in qs]
    row = {qid: i for i, qid in enumerate(ids)}
    col = {m["name"]: j for j, m in enumerate(MODELS)}
    n, T = len(ids), len(MODELS)
    correct = np.full((n, T), np.nan)
    tokens = np.zeros((n, T))
    latency = np.zeros((n, T))
    with LABELS.open() as f:
        for line in f:
            r = json.loads(line)
            i, j = row[r["id"]], col[r["model"]]
            correct[i, j] = r["correct"]
            tokens[i, j] = r["prompt_tokens"] + r["output_tokens"]
            latency[i, j] = r["latency_s"]
    if np.isnan(correct).any():
        raise ValueError(f"{int(np.isnan(correct).sum())} (question, model) labels missing")

    texts = [question_text(q) for q in qs]
    emb = embed_cached(texts)
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    order = np.argsort(-sims, axis=1)[:, : n - 1]

    rng = np.random.default_rng(SEED)
    dev = np.zeros(n, bool)
    dev[rng.choice(n, int(DEV_FRACTION * n), replace=False)] = True
    return Bench(ids, np.array([q["source"] for q in qs]), texts, emb, correct.astype(bool),
                 2 * PARAMS * tokens, latency, sims, order, dev, ~dev)


# ----------------------------------------------------------------- routing

def route_all(b: Bench, k: int, min_correct: int, min_similarity: float):
    """Vectorized version of router.decide() for every question (leave-one-out).

    Returns (tier index [n], reason [n] in {"evidence","mixed_evidence","unfamiliar"}).
    `check_matches_router` verifies this equals the explained policy exactly.
    """
    nbr = b.order[:, :k]
    tally = b.correct[nbr].sum(axis=1)               # [n, T]
    top_sim = np.take_along_axis(b.sims, nbr[:, :1], axis=1)[:, 0]
    largest = len(TIERS) - 1
    tier = np.full(len(b.ids), largest)
    reason = np.full(len(b.ids), "mixed_evidence", dtype=object)
    for t in reversed(range(largest)):              # smallest qualifying tier wins
        ok = tally[:, t] >= min_correct
        tier[ok] = t
        reason[ok] = "evidence"
    unfamiliar = top_sim < min_similarity
    tier[unfamiliar] = largest
    reason[unfamiliar] = "unfamiliar"
    return tier, reason


def check_matches_router(b: Bench, k, min_correct, min_similarity, n_check=200):
    """Guarantee the evaluated numbers come from the same policy that writes explanations."""
    from .router import KNNRouter
    r = KNNRouter(b.ids, b.texts, b.emb, b.correct.astype(float), k, min_correct, min_similarity)
    tier, reason = route_all(b, k, min_correct, min_similarity)
    for i in np.linspace(0, len(b.ids) - 1, n_check).astype(int):
        d = r.route_embedding(b.emb[i], exclude_ids={b.ids[i]})
        assert (TIERS.index(d.tier), d.reason) == (tier[i], reason[i]), b.ids[i]


def score(b: Bench, tier: np.ndarray, mask: np.ndarray) -> dict:
    idx = np.where(mask)[0]
    t = tier[idx]
    big = len(TIERS) - 1
    return {
        "accuracy": float(b.correct[idx, t].mean()),
        "compute_rel": float(b.flops[idx, t].sum() / b.flops[idx, big].sum()),
        "latency_rel": float(b.latency[idx, t].sum() / b.latency[idx, big].sum()),
        "share": {TIERS[j]: float((t == j).mean()) for j in range(len(TIERS))},
        "n": int(len(idx)),
    }


def baselines(b: Bench, mask: np.ndarray) -> dict:
    n = len(b.ids)
    out = {f"always_{t}": score(b, np.full(n, j), mask) for j, t in enumerate(TIERS)}
    # Oracle: cheapest model that answers correctly; if none does, the cheapest overall.
    any_ok = b.correct.any(axis=1)
    oracle = np.where(any_ok, b.correct.argmax(axis=1), 0)
    out["oracle"] = score(b, oracle, mask)
    return out


def sweep(b: Bench, ks=(3, 5, 7, 10), sims=(0.0, 0.3, 0.4, 0.5)) -> list[dict]:
    rows = []
    for k in ks:
        for m in range(1, k + 1):
            for s in sims:
                tier, _ = route_all(b, k, m, s)
                row = {"k": k, "min_correct": m, "min_similarity": s}
                for split, mask in (("dev", b.dev), ("test", b.test)):
                    sc = score(b, tier, mask)
                    row[f"{split}_accuracy"] = sc["accuracy"]
                    row[f"{split}_compute_rel"] = sc["compute_rel"]
                rows.append(row)
    return rows


def pick_config(rows: list[dict], dev_largest_acc: float) -> dict:
    """On DEV only: the cheapest setting within MAX_ACC_DROP of always-largest accuracy."""
    ok = [r for r in rows if r["dev_accuracy"] >= dev_largest_acc - MAX_ACC_DROP]
    return min(ok, key=lambda r: (r["dev_compute_rel"], -r["dev_accuracy"]))


# ------------------------------------------------------ failures & safeguard

def failure_analysis(b: Bench, tier: np.ndarray, reason: np.ndarray, mask: np.ndarray) -> dict:
    idx = np.where(mask)[0]
    big = len(TIERS) - 1
    got = b.correct[idx, tier[idx]]
    big_ok = b.correct[idx, big]
    downgraded = tier[idx] < big
    cats = {
        "correct": int(got.sum()),
        # the costly error: we used a smaller model, it failed, the largest would have succeeded
        "harmful_downgrade": int((~got & downgraded & big_ok).sum()),
        # wrong, but the largest model fails too, so routing is not to blame
        "unavoidable": int((~got & ~big_ok).sum()),
    }

    by_reason = {}
    for rsn in ("evidence", "mixed_evidence", "unfamiliar"):
        sel = idx[reason[idx] == rsn]
        by_reason[rsn] = {
            "n": int(len(sel)),
            "model_accuracy": {t: (float(b.correct[sel, j].mean()) if len(sel) else None)
                               for j, t in enumerate(TIERS)},
        }

    # Counterfactual "no safeguard": questions the safeguard sent to the largest model
    # would instead go to the next-largest model (the cheapest plausible alternative).
    fb = idx[reason[idx] != "evidence"]
    alt = big - 1
    safeguard = {
        "n_fallback": int(len(fb)),
        "errors_prevented": int((~b.correct[fb, alt] & b.correct[fb, big]).sum()),
        "errors_introduced": int((b.correct[fb, alt] & ~b.correct[fb, big]).sum()),
        "unneeded_fallbacks": int((b.correct[fb, alt] & b.correct[fb, big]).sum()),
        "accuracy_with": float(b.correct[fb, big].mean()) if len(fb) else None,
        "accuracy_without": float(b.correct[fb, alt].mean()) if len(fb) else None,
    }

    by_source = {}
    for src in sorted(set(b.sources[idx])):
        s_mask = mask & (b.sources == src)
        sc = score(b, tier, s_mask)
        sc["always_largest_accuracy"] = float(b.correct[s_mask, big].mean())
        by_source[src] = sc

    return {"categories": cats, "by_reason": by_reason, "safeguard": safeguard,
            "by_source": by_source}


# ------------------------------------------------------------------ cascade

CONFIDENCE = LABELS.parent / "confidence.jsonl"


def load_confidence(ids: list[str], path: Path = CONFIDENCE):
    """Answer confidence and correctness for the cheap tiers: two arrays [n, n_tiers - 1]."""
    row = {qid: i for i, qid in enumerate(ids)}
    col = {m["name"]: j for j, m in enumerate(MODELS[:-1])}
    conf = np.full((len(ids), len(col)), np.nan)
    corr = np.zeros((len(ids), len(col)), bool)
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            i, j = row.get(r["id"]), col.get(r["model"])
            if i is not None and j is not None:
                conf[i, j], corr[i, j] = r["answer_prob"], r["correct"]
    if np.isnan(conf).any():
        raise ValueError(f"{int(np.isnan(conf).sum())} confidence values missing; is the run finished?")
    return conf, corr


def calibration_tally(conf: np.ndarray, correct: np.ndarray, k: int) -> np.ndarray:
    """For every question and cheap tier: how many of the k other benchmark questions
    with the closest confidence that tier got right (leave-one-out). Shape [n, T-1]."""
    n, T = conf.shape
    out = np.zeros((n, T), int)
    for t in range(T):
        dist = np.abs(conf[:, t][:, None] - conf[:, t][None, :])
        np.fill_diagonal(dist, np.inf)
        nbr = np.argsort(dist, axis=1, kind="stable")[:, :k]
        out[:, t] = correct[nbr, t].sum(axis=1)
    return out


def cascade_all(b: Bench, tally: np.ndarray, stages: tuple, min_correct: int):
    """Vectorized cascade. Returns (final tier [n], models_run mask [n, T])."""
    n = len(b.ids)
    big = len(TIERS) - 1
    final = np.full(n, big)
    ran = np.zeros((n, len(TIERS)), bool)
    open_ = np.ones(n, bool)
    for t in stages:
        ran[open_, t] = True
        acc = open_ & (tally[:, t] >= min_correct)
        final[acc] = t
        open_ &= ~acc
    ran[open_, big] = True
    return final, ran


def cascade_score(b: Bench, final, ran, mask) -> dict:
    idx = np.where(mask)[0]
    big = len(TIERS) - 1
    return {
        "accuracy": float(b.correct[idx, final[idx]].mean()),
        "compute_rel": float((b.flops[idx] * ran[idx]).sum() / b.flops[idx, big].sum()),
        "latency_rel": float((b.latency[idx] * ran[idx]).sum() / b.latency[idx, big].sum()),
        "share": {TIERS[j]: float((final[idx] == j).mean()) for j in range(len(TIERS))},
        "n": int(len(idx)),
    }


STAGE_SETS = {"tiny>small>medium": (0, 1), "small>medium": (1,), "tiny>medium": (0,)}


def cascade_sweep(b: Bench, conf, ks=(25, 50, 100), accs=np.round(np.arange(0.50, 1.001, 0.02), 2)):
    rows = []
    for k in ks:
        tally = calibration_tally(conf, b.correct[:, :-1], k)
        for name, stages in STAGE_SETS.items():
            for a in accs:
                m = int(np.ceil(a * k - 1e-9))
                final, ran = cascade_all(b, tally, stages, m)
                row = {"stages": name, "k": k, "min_accuracy": float(a), "min_correct": m}
                for split, mask in (("dev", b.dev), ("test", b.test)):
                    sc = cascade_score(b, final, ran, mask)
                    row[f"{split}_accuracy"], row[f"{split}_compute_rel"] = sc["accuracy"], sc["compute_rel"]
                rows.append(row)
    return rows


def threshold_cascade_sweep(b: Bench, conf, taus=np.round(np.arange(0.30, 1.0, 0.02), 2)):
    """Standard, unexplained baseline: keep the answer if raw confidence >= tau."""
    rows = []
    for name, stages in STAGE_SETS.items():
        for tau in taus:
            tally = (conf >= tau).astype(int)
            final, ran = cascade_all(b, tally, stages, 1)
            row = {"stages": name, "tau": float(tau)}
            for split, mask in (("dev", b.dev), ("test", b.test)):
                sc = cascade_score(b, final, ran, mask)
                row[f"{split}_accuracy"], row[f"{split}_compute_rel"] = sc["accuracy"], sc["compute_rel"]
            rows.append(row)
    return rows


def check_cascade_matches_router(b: Bench, conf, k, stages, min_accuracy, n_check=200):
    """The vectorized numbers must equal what the explained CascadeRouter decides."""
    from .cascade import CascadeRouter
    r = CascadeRouter(b.ids, conf, b.correct[:, :-1], stages, k, min_accuracy)
    tally = calibration_tally(conf, b.correct[:, :-1], k)
    final, _ = cascade_all(b, tally, stages, r.min_correct)
    for i in np.linspace(0, len(b.ids) - 1, n_check).astype(int):
        t_final = len(TIERS) - 1
        for t in stages:
            if r.judge(t, conf[i, t], exclude_id=b.ids[i]).accepted:
                t_final = t
                break
        assert t_final == final[i], b.ids[i]
