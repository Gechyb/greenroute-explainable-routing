"""Counterfactual faithfulness test: does changing the cited evidence change the decision?

Two routers, both shown with the SAME style of explanation ("these k similar
questions, and which models got them right"):

  * GreenRoute: the decision is computed from exactly that evidence.
  * Black box: a logistic-regression classifier per cheap tier predicts
    "will this model be correct?" from the embedding; it routes to the smallest
    tier with predicted success >= theta. Its similar-question explanation is
    post hoc: shown alongside, but not what produced the decision.

Perturbations, applied identically to both routers for each test question:
  * cited:   flip the labels of the k cited neighbours so they argue AGAINST the
             decision (chosen cheap tier: correct -> wrong; fallback to largest:
             next tier wrong -> correct). A faithful explanation should mean the
             decision now changes.
  * uncited: flip the same number of labels on random questions that were NOT
             cited. A faithful explanation should mean the decision stays put.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from .config import TIERS
from .evaluation import MAX_ACC_DROP, Bench, route_all, score

BIG = len(TIERS) - 1


# ---------------------------------------------------------------- GreenRoute

def _gr_decide(evidence: np.ndarray, top_sim: float, min_correct: int, min_sim: float) -> int:
    """Same rule as router.decide(), on a [k, T] boolean evidence matrix."""
    if top_sim < min_sim:
        return BIG
    tally = evidence.sum(axis=0)
    for t in range(BIG):
        if tally[t] >= min_correct:
            return t
    return BIG


def _flip_target(decided: int) -> tuple[int, bool]:
    """Which tier's labels to flip, and to what value, to argue against `decided`."""
    return (decided, False) if decided < BIG else (BIG - 1, True)


def greenroute_test(b: Bench, cfg: dict, rng) -> dict:
    k, m, s = cfg["k"], cfg["min_correct"], cfg["min_similarity"]
    tier, reason = route_all(b, k, m, s)
    idx = [i for i in np.where(b.test)[0] if reason[i] != "unfamiliar"]  # similarity-driven: N/A
    cited_changed = uncited_changed = 0
    for i in idx:
        nbr = b.order[i, :k]
        top_sim = b.sims[i, nbr[0]]
        assert _gr_decide(b.correct[nbr], top_sim, m, s) == tier[i]  # sufficiency
        col, val = _flip_target(tier[i])

        ev = b.correct[nbr].copy()
        ev[:, col] = val
        cited_changed += _gr_decide(ev, top_sim, m, s) != tier[i]

        # Flipping labels of uncited questions cannot reach the decision; verify by
        # recomputing the neighbour set and evidence from a perturbed label matrix.
        others = np.setdiff1d(np.arange(len(b.ids)), np.append(nbr, i))
        flip = rng.choice(others, k, replace=False)
        lab = b.correct.copy()
        lab[flip, col] = val
        uncited_changed += _gr_decide(lab[b.order[i, :k]], top_sim, m, s) != tier[i]
    n = len(idx)
    return {"n": n, "n_unfamiliar_skipped": int(b.test.sum() - n),
            "cited_change_rate": cited_changed / n, "uncited_change_rate": uncited_changed / n}


# ----------------------------------------------------------------- black box

def _fit(X, y):
    if y.all() or not y.any():  # degenerate after flips: constant predictor
        return float(y.mean())
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, y)


def _prob(model, x):
    return model if isinstance(model, float) else float(model.predict_proba(x[None])[0, 1])


def _bb_decide(probs: list[float], theta: float) -> int:
    for t, p in enumerate(probs):
        if p >= theta:
            return t
    return BIG


def _loo_probs(b: Bench, i: int, labels: np.ndarray) -> list[float]:
    train = np.arange(len(b.ids)) != i
    return [_prob(_fit(b.emb[train], labels[train, t]), b.emb[i]) for t in range(BIG)]


def blackbox_test(b: Bench, k: int, rng, thetas=np.round(np.arange(0.40, 0.96, 0.05), 2)) -> dict:
    # Leave-one-out probabilities for every question (one fit per question per cheap tier).
    probs = np.array([_loo_probs(b, i, b.correct) for i in range(len(b.ids))])

    def route(theta):
        return np.array([_bb_decide(p, theta) for p in probs])

    dev_big = b.correct[b.dev, BIG].mean()
    cands = [(score(b, route(th), b.dev), th) for th in thetas]
    ok = [(sc, th) for sc, th in cands if sc["accuracy"] >= dev_big - MAX_ACC_DROP]
    theta = min(ok, key=lambda x: x[0]["compute_rel"])[1] if ok else max(thetas)
    tier = route(theta)

    cited_changed = uncited_changed = 0
    idx = np.where(b.test)[0]
    for i in idx:
        col, val = _flip_target(tier[i])
        nbr = b.order[i, :k]
        for kind, rows in (("cited", nbr),
                           ("uncited", rng.choice(np.setdiff1d(np.arange(len(b.ids)),
                                                               np.append(nbr, i)), k, replace=False))):
            lab = b.correct.copy()
            lab[rows, col] = val
            p = list(probs[i])
            train = np.arange(len(b.ids)) != i
            p[col] = _prob(_fit(b.emb[train], lab[train, col]), b.emb[i])
            changed = _bb_decide(p, theta) != tier[i]
            if kind == "cited":
                cited_changed += changed
            else:
                uncited_changed += changed
    return {"theta": float(theta), "tier": tier, "test_score": score(b, tier, b.test),
            "n": int(len(idx)), "cited_change_rate": cited_changed / len(idx),
            "uncited_change_rate": uncited_changed / len(idx)}
