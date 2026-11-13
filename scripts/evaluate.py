"""Run the full evaluation and write results/ (numbers) for the figures and poster.

  .venv/bin/python -m scripts.evaluate

Outputs
  results/metrics.json       headline numbers, baselines, failures, faithfulness
  results/sweep.csv          every threshold setting (accuracy vs compute, dev and test)
  results/per_question.csv   routing decision and outcomes for every question
"""

import csv
import json
from pathlib import Path

import numpy as np

from greenroute.config import TIERS
from greenroute.evaluation import (DEV_FRACTION, MAX_ACC_DROP, SEED, baselines, check_matches_router,
                                   failure_analysis, load_bench, pick_config, route_all, score, sweep)
from greenroute.faithfulness import blackbox_test, greenroute_test

OUT = Path(__file__).resolve().parent.parent / "results"


def main():
    b = load_bench()
    print(f"Loaded {len(b.ids)} questions: {int(b.dev.sum())} dev / {int(b.test.sum())} test")

    rows = sweep(b)
    base_dev = baselines(b, b.dev)
    cfg = pick_config(rows, base_dev["always_medium"]["accuracy"])
    k, m, s = cfg["k"], cfg["min_correct"], cfg["min_similarity"]
    print(f"Chosen on dev: k={k}, min_correct={m}, min_similarity={s}")
    check_matches_router(b, k, m, s)  # evaluated policy == explained policy

    tier, reason = route_all(b, k, m, s)
    base = baselines(b, b.test)
    gr = score(b, tier, b.test)
    fails = failure_analysis(b, tier, reason, b.test)

    rng = np.random.default_rng(SEED)
    print("Faithfulness: GreenRoute ...")
    faith_gr = greenroute_test(b, cfg, rng)
    print("Faithfulness: black-box baseline (fits ~3,400 small classifiers, a few minutes) ...")
    bb = blackbox_test(b, k, rng)

    metrics = {
        "protocol": {"leave_one_out": True, "dev_fraction": DEV_FRACTION, "seed": SEED,
                     "tuning_rule": f"cheapest dev setting within {MAX_ACC_DROP:.0%} of always-medium accuracy",
                     "compute": "2 x params x (prompt+output tokens), relative to always-medium"},
        "config": {"k": k, "min_correct": m, "min_similarity": s},
        "test": {"greenroute": gr, **base, "blackbox": bb["test_score"]},
        "oracle_gap_closed": (base["always_medium"]["compute_rel"] - gr["compute_rel"])
                             / (base["always_medium"]["compute_rel"] - base["oracle"]["compute_rel"]),
        "failures": fails,
        "faithfulness": {
            "greenroute": faith_gr,
            "blackbox": {kk: v for kk, v in bb.items() if kk not in ("tier", "test_score")},
        },
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with (OUT / "sweep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with (OUT / "per_question.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "source", "split", "routed_tier", "reason", "routed_correct", "blackbox_tier",
                    *[f"{t}_correct" for t in TIERS]])
        for i, qid in enumerate(b.ids):
            w.writerow([qid, b.sources[i], "dev" if b.dev[i] else "test", TIERS[tier[i]], reason[i],
                        int(b.correct[i, tier[i]]), TIERS[bb["tier"][i]],
                        *[int(x) for x in b.correct[i]]])

    # ------------------------------------------------------------ summary
    print("\n=== TEST SET ===")
    for name in ("always_tiny", "always_small", "always_medium", "oracle", "blackbox", "greenroute"):
        sc = metrics["test"][name]
        print(f"  {name:<14} acc {sc['accuracy']:.1%}   compute {sc['compute_rel']:.1%}   "
              f"time {sc['latency_rel']:.1%}   mix " + " ".join(f"{t} {v:.0%}" for t, v in sc["share"].items()))
    print(f"  oracle compute-savings gap closed: {metrics['oracle_gap_closed']:.0%}")
    print(f"  failures: {fails['categories']}")
    print(f"  safeguard: {fails['safeguard']}")
    for r, v in fails["by_reason"].items():
        print(f"  reason {r:<15} n={v['n']:<4} " +
              " ".join(f"{t} {a:.0%}" for t, a in v["model_accuracy"].items() if a is not None))
    print(f"  faithfulness GreenRoute: {faith_gr}")
    print(f"  faithfulness black box : {metrics['faithfulness']['blackbox']}")


if __name__ == "__main__":
    main()
