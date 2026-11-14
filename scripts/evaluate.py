"""Run the full evaluation and write results/ (numbers) for the figures and poster.

  .venv/bin/python -m scripts.evaluate

Two GreenRoute designs are compared:
  * similarity router  route BEFORE answering, from similar benchmark questions
  * cascade            answer with a small model first; keep it if past questions
                       answered with similar confidence were usually right
plus fixed-model baselines, an oracle, a black-box router, and an unexplained
raw-threshold cascade.

Outputs
  results/metrics.json            headline numbers, failures, faithfulness
  results/sweep.csv               similarity-router settings (accuracy vs compute)
  results/cascade_sweep.csv       cascade settings
  results/threshold_sweep.csv     raw-threshold cascade settings
  results/per_question.csv        decisions and outcomes for every question
"""

import csv
import json
from pathlib import Path

import numpy as np

from greenroute.config import TIERS
from greenroute.evaluation import (DEV_FRACTION, MAX_ACC_DROP, SEED, STAGE_SETS, baselines,
                                   calibration_tally, cascade_all, cascade_score, cascade_sweep,
                                   check_cascade_matches_router, check_matches_router, failure_analysis,
                                   load_bench, load_confidence, pick_config, route_all, score, sweep,
                                   threshold_cascade_sweep)
from greenroute.faithfulness import blackbox_test, cascade_test, greenroute_test

OUT = Path(__file__).resolve().parent.parent / "results"
BIG = len(TIERS) - 1


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cascade_failures(b, final, mask):
    idx = np.where(mask)[0]
    got = b.correct[idx, final[idx]]
    big_ok = b.correct[idx, BIG]
    by_source = {}
    for src in sorted(set(b.sources[idx])):
        sel = idx[b.sources[idx] == src]
        by_source[src] = {"n": int(len(sel)), "accuracy": float(b.correct[sel, final[sel]].mean()),
                          "always_largest_accuracy": float(b.correct[sel, BIG].mean()),
                          "share": {TIERS[j]: float((final[sel] == j).mean()) for j in range(len(TIERS))}}
    return {
        "correct": int(got.sum()),
        "harmful_downgrade": int((~got & (final[idx] < BIG) & big_ok).sum()),
        "unavoidable": int((~got & ~big_ok).sum()),
        # the cascade kept a cheap answer the largest model would have gotten wrong, and it was right
        "cheap_beats_largest": int((got & (final[idx] < BIG) & ~big_ok).sum()),
        "by_source": by_source,
    }


def main():
    b = load_bench()
    print(f"Loaded {len(b.ids)} questions: {int(b.dev.sum())} dev / {int(b.test.sum())} test")
    base_dev = baselines(b, b.dev)
    base = baselines(b, b.test)
    LARGEST = f"always_{TIERS[-1]}"
    dev_big = base_dev[LARGEST]["accuracy"]
    rng = np.random.default_rng(SEED)

    # -------------------------------------------------- similarity router
    rows = sweep(b)
    cfg = pick_config(rows, dev_big)
    k, m, s = cfg["k"], cfg["min_correct"], cfg["min_similarity"]
    check_matches_router(b, k, m, s)
    tier, reason = route_all(b, k, m, s)
    sim_score = score(b, tier, b.test)
    sim_fails = failure_analysis(b, tier, reason, b.test)
    faith_sim = greenroute_test(b, cfg, rng)
    print("Black-box baseline (fits a few thousand small classifiers) ...")
    bb = blackbox_test(b, k, rng)

    # ------------------------------------------------------------ cascade
    conf, _ = load_confidence(b.ids)
    c_rows = cascade_sweep(b, conf)
    c_cfg = pick_config(c_rows, dev_big)
    ck, cm, cstages = c_cfg["k"], c_cfg["min_correct"], STAGE_SETS[c_cfg["stages"]]
    print(f"Cascade chosen on dev: {c_cfg['stages']}, k={ck}, min_accuracy={c_cfg['min_accuracy']}")
    check_cascade_matches_router(b, conf, ck, cstages, c_cfg["min_accuracy"])
    final, ran = cascade_all(b, calibration_tally(conf, b.correct[:, :-1], ck), cstages, cm)
    c_score = cascade_score(b, final, ran, b.test)
    c_fails = cascade_failures(b, final, b.test)
    faith_c = cascade_test(b, conf, ck, cstages, cm, rng)

    t_rows = threshold_cascade_sweep(b, conf)
    t_cfg = pick_config(t_rows, dev_big)
    t_final, t_ran = cascade_all(b, (conf >= t_cfg["tau"]).astype(int), STAGE_SETS[t_cfg["stages"]], 1)
    t_score = cascade_score(b, t_final, t_ran, b.test)

    big_c, orc_c = base[LARGEST]["compute_rel"], base["oracle"]["compute_rel"]
    metrics = {
        "protocol": {"leave_one_out": True, "dev_fraction": DEV_FRACTION, "seed": SEED,
                     "tuning_rule": f"cheapest dev setting within {MAX_ACC_DROP:.0%} of always-largest accuracy",
                     "compute": "2 x params x (prompt+output tokens) of every model run, relative to always-largest"},
        "config": {"similarity": {"k": k, "min_correct": m, "min_similarity": s},
                   "cascade": {"stages": c_cfg["stages"], "k": ck, "min_accuracy": c_cfg["min_accuracy"],
                               "min_correct": cm},
                   "threshold_cascade": {"stages": t_cfg["stages"], "tau": t_cfg["tau"]},
                   "blackbox": {"theta": bb["theta"]}},
        "test": {**base, "similarity": sim_score, "blackbox": bb["test_score"],
                 "cascade": c_score, "threshold_cascade": t_score},
        "oracle_gap_closed": {"similarity": (big_c - sim_score["compute_rel"]) / (big_c - orc_c),
                              "cascade": (big_c - c_score["compute_rel"]) / (big_c - orc_c)},
        "failures": {"similarity": sim_fails, "cascade": c_fails},
        "faithfulness": {
            "similarity": faith_sim,
            "blackbox": {kk: v for kk, v in bb.items() if kk not in ("tier", "test_score")},
            "cascade": faith_c,
        },
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    write_csv(OUT / "sweep.csv", rows)
    write_csv(OUT / "cascade_sweep.csv", c_rows)
    write_csv(OUT / "threshold_sweep.csv", t_rows)
    with (OUT / "per_question.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "split", "similarity_tier", "similarity_reason", "blackbox_tier",
                    "cascade_tier", *[f"{t}_confidence" for t in TIERS[:-1]],
                    *[f"{t}_correct" for t in TIERS]])
        for i, qid in enumerate(b.ids):
            w.writerow([qid, b.sources[i], "dev" if b.dev[i] else "test", TIERS[tier[i]], reason[i],
                        TIERS[bb["tier"][i]], TIERS[final[i]], *[round(c, 4) for c in conf[i]],
                        *[int(x) for x in b.correct[i]]])

    print("\n=== TEST SET (700 questions) ===")
    for name in (*[f"always_{t}" for t in TIERS], "oracle",
                 "similarity", "blackbox", "threshold_cascade", "cascade"):
        sc = metrics["test"][name]
        print(f"  {name:<18} acc {sc['accuracy']:.1%}   compute {sc['compute_rel']:.1%}   "
              f"time {sc['latency_rel']:.1%}   final " +
              " ".join(f"{t} {v:.0%}" for t, v in sc["share"].items()))
    print(f"  oracle gap closed: {metrics['oracle_gap_closed']}")
    print(f"  cascade failures: { {kk: v for kk, v in c_fails.items() if kk != 'by_source'} }")
    print(f"  faithfulness: {json.dumps(metrics['faithfulness'], default=float)}")


if __name__ == "__main__":
    main()
