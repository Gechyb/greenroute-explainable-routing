"""Poster figures from results/ (run scripts.evaluate first).

  .venv/bin/python -m scripts.make_figures

Writes figures/fig1_tradeoff.png, fig2_signal.png, fig3_where.png, fig4_faithfulness.png
(and .svg versions for print).
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from greenroute.config import MODELS, TIERS  # noqa: E402
from greenroute.evaluation import calibration_tally, load_bench, load_confidence  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "figures"

# Reference palette (dataviz skill, light mode): categorical slots 1-3 validate all-pairs.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
# Ordinal blue ramp (steps 250 -> 700), one step per model size, smallest = lightest.
_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
TIER_RAMP = dict(zip(TIERS, _RAMP[-len(TIERS):] if len(TIERS) < 4 else _RAMP))
LARGEST = f"always_{TIERS[-1]}"
SIZE = {m["tier"]: f"{m['params_b']:g}B" for m in MODELS}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def read_csv(name):
    with (RES / name).open() as f:
        return [{k: (float(v) if k not in ("stages",) else v) for k, v in r.items()} for r in csv.DictReader(f)]


def frontier(rows, x="test_compute_rel", y="test_accuracy"):
    """Upper-left envelope: best accuracy achievable at or below each compute level."""
    pts = sorted((r[x], r[y]) for r in rows)
    out, best = [], -1
    for cx, cy in pts:
        if cy > best:
            out.append((cx, cy))
            best = cy
    return np.array(out)


# ------------------------------------------------------------------ figure 1

def fig_tradeoff(m):
    t = m["test"]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fixed = [t[f"always_{x}"] for x in TIERS]
    ax.plot([f["compute_rel"] * 100 for f in fixed], [f["accuracy"] * 100 for f in fixed],
            ls=(0, (4, 3)), lw=1.5, color=MUTED, zorder=1)
    ax.text(31, 55, "dashed: random mix\nof fixed models", color=MUTED, fontsize=9)

    for name, color, label in (("cascade_sweep.csv", BLUE, "GreenRoute cascade (explained)"),
                               ("threshold_sweep.csv", AQUA, "Raw-confidence cascade (no explanation)"),
                               ("sweep.csv", ORANGE, "Similar-questions router")):
        f = frontier(read_csv(name))
        ax.plot(f[:, 0] * 100, f[:, 1] * 100, color=color, lw=2, label=label, zorder=3,
                drawstyle="steps-post")

    for tier, f in zip(TIERS, fixed):
        ax.scatter(f["compute_rel"] * 100, f["accuracy"] * 100, s=60, color=INK2, zorder=4,
                   edgecolor=SURFACE, linewidth=2)
        off = (6, -14) if tier == TIERS[0] else (-8, 8)
        ax.annotate(f"always {tier}", (f["compute_rel"] * 100, f["accuracy"] * 100), xytext=off,
                    textcoords="offset points", fontsize=9, color=INK2,
                    ha="left" if tier == TIERS[0] else "right")
    o = t["oracle"]
    ax.scatter(o["compute_rel"] * 100, o["accuracy"] * 100, marker="*", s=180, color=INK, zorder=4,
               edgecolor=SURFACE, linewidth=1.5)
    ax.annotate("perfect oracle", (o["compute_rel"] * 100, o["accuracy"] * 100),
                xytext=(8, 2), textcoords="offset points", fontsize=9, color=INK)

    c = t["cascade"]
    ax.scatter(c["compute_rel"] * 100, c["accuracy"] * 100, s=110, color=BLUE, zorder=5,
               edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"chosen setting\n{c['accuracy']:.1%} acc. at {c['compute_rel']:.0%} compute",
                (c["compute_rel"] * 100, c["accuracy"] * 100), xytext=(14, -34),
                textcoords="offset points", fontsize=9, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})

    ax.set_xlabel(f"Compute used (% of always using the {SIZE[TIERS[-1]]} model)")
    ax.set_ylabel("Accuracy on test questions (%)")
    ax.set_xlim(0, 105)
    ax.set_title(f"The cascade saves {1 - c['compute_rel']:.0%} of compute for "
                 f"{(t[LARGEST]['accuracy'] - c['accuracy']) * 100:.1f} points of accuracy")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig1_tradeoff")


# ------------------------------------------------------------------ figure 2

def fig_signal():
    b = load_bench()
    conf, _ = load_confidence(b.ids)
    rows = []
    for j, tier in enumerate(TIERS[:-1]):
        nbr_rate = b.correct[b.order[:, :10]].mean(axis=1)[:, j]
        cal = calibration_tally(conf, b.correct[:, :-1], 50)[:, j] / 50
        rows.append((tier, roc_auc_score(b.correct[:, j], nbr_rate),
                     roc_auc_score(b.correct[:, j], conf[:, j]),
                     roc_auc_score(b.correct[:, j], cal)))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = np.arange(len(rows))
    h = 0.34
    for k, (col, lab, off) in enumerate(((ORANGE, "Similar questions (topic)", h / 2 + 0.02),
                                         (BLUE, "Model's own confidence", -h / 2 - 0.02))):
        vals = [r[1] if k == 0 else r[3] for r in rows]
        ax.barh(y + off, [v - 0.5 for v in vals], left=0.5, height=h, color=col, label=lab)
        for yi, v in zip(y, vals):
            ax.text(v + 0.006, yi + off, f"{v:.2f}", va="center", fontsize=10, color=INK)
    ax.axvline(0.5, color=INK2, lw=1)
    ax.text(0.505, -0.62, "coin flip = 0.50", color=INK2, fontsize=9)
    ax.set_ylim(-0.75, len(rows) - 0.4)
    ax.set_axisbelow(True)
    ax.set_yticks(y, [f"{r[0]} model" for r in rows])
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("How well the evidence predicts a correct answer (AUC)")
    ax.set_title("Confidence predicts success; topic similarity barely does")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_signal")
    return [{"tier": r[0], "auc_similar_questions": r[1], "auc_raw_confidence": r[2],
             "auc_confidence_evidence": r[3]} for r in rows]


# ------------------------------------------------------------------ figure 3

def fig_where(m):
    src = m["failures"]["cascade"]["by_source"]
    order = ["arc_easy", "arc_challenge", "mmlu", "gsm8k"]
    names = {"arc_easy": "ARC-Easy (science)", "arc_challenge": "ARC-Challenge (science)",
             "mmlu": "MMLU (academic)", "gsm8k": "GSM8K (math)"}
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(order))[::-1]
    for yi, s in zip(y, order):
        left = 0
        for tier in TIERS:
            w = src[s]["share"][tier] * 100
            if w > 0:
                ax.barh(yi, w - 0.4, left=left + 0.2, color=TIER_RAMP[tier], height=0.62)
                if w >= 7:
                    ax.text(left + w / 2, yi, f"{w:.0f}%", ha="center", va="center", fontsize=9,
                            color=INK if tier == "tiny" else "#ffffff")
            left += w
        d = (src[s]["accuracy"] - src[s]["always_largest_accuracy"]) * 100
        ax.text(102, yi, f"{src[s]['accuracy']:.0%} acc. ({d:+.1f} vs {SIZE[TIERS[-1]]})", va="center",
                fontsize=9, color=INK2)
    ax.set_yticks(y, [names[s] for s in order])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of questions answered by each model (%)")
    ax.set_axisbelow(True)
    ax.set_title("Where questions go under the cascade, by benchmark")
    handles = [plt.Rectangle((0, 0), 1, 1, color=TIER_RAMP[t]) for t in TIERS]
    ax.legend(handles, [f"{t} ({SIZE[t]})" for t in TIERS],
              loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=len(TIERS), fontsize=9)
    ax.grid(axis="y", visible=False)
    save(fig, "fig3_where")


# ------------------------------------------------------------------ figure 4

def fig_faithfulness(m):
    f = m["faithfulness"]
    routers = [("cascade", "GreenRoute cascade"), ("similarity", "Similar-questions router"),
               ("blackbox", "Black-box router +\npost-hoc explanation")]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(routers))
    w = 0.36
    for k, (key, col, lab) in enumerate((("cited_change_rate", BLUE, "Flip the evidence it cites"),
                                         ("uncited_change_rate", "#b9b8b2", "Flip evidence it doesn't cite"))):
        vals = [f[r][key] * 100 for r, _ in routers]
        xs = x + (k - 0.5) * (w + 0.03)
        ax.bar(xs, [max(v, 0.6) for v in vals], width=w, color=col, label=lab)
        for xi, v in zip(xs, vals):
            ax.text(xi, max(v, 0.6) + 2, f"{v:.0f}%", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x, [lab for _, lab in routers])
    ax.set_ylim(0, 112)
    ax.set_ylabel("Decisions that changed (%)")
    ax.set_axisbelow(True)
    ax.set_title("Faithful explanations: change the cited evidence, the decision follows")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_faithfulness")


def main():
    m = json.loads((RES / "metrics.json").read_text())
    fig_tradeoff(m)
    aucs = fig_signal()
    fig_where(m)
    fig_faithfulness(m)
    (RES / "signal_auc.json").write_text(json.dumps(aucs, indent=2))
    print("Wrote figures to", FIG)
    for a in aucs:
        print(" ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in a.items()})


if __name__ == "__main__":
    main()
