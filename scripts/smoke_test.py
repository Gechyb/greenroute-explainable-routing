"""Smoke test: every model answers a handful of questions, gets graded, and
we extrapolate how long the full ~1,000-question labeling run will take.

Run:  .venv/bin/python -m scripts.smoke_test
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from greenroute.config import MODELS
from greenroute.grading import SYSTEM, build_prompt, is_correct
from greenroute.llm import available_models, generate

QUESTIONS = [
    {"id": "mc-easy", "fmt": "mc", "gold": "B",
     "q": "What is the chemical symbol for water?",
     "choices": ["O2", "H2O", "CO2", "NaCl"]},
    {"id": "mc-mid", "fmt": "mc", "gold": "C",
     "q": "Which planet has the shortest orbital period around the Sun?",
     "choices": ["Venus", "Earth", "Mercury", "Mars"]},
    {"id": "mc-hard", "fmt": "mc", "gold": "D",
     "q": "In a perfectly competitive market in long-run equilibrium, which of the following is true?",
     "choices": ["Price exceeds average total cost", "Firms earn positive economic profit",
                 "Marginal revenue exceeds price", "Price equals minimum average total cost"]},
    {"id": "num-easy", "fmt": "numeric", "gold": 12,
     "q": "Sam has 7 apples and buys 5 more. How many apples does Sam have?"},
    {"id": "num-mid", "fmt": "numeric", "gold": 18,
     "q": "Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes muffins with four. "
          "She sells the remainder at $2 per egg. How many dollars does she make every day?"},
    {"id": "num-hard", "fmt": "numeric", "gold": 70000,
     "q": "Josh buys a house for $80,000 and puts in $50,000 in repairs. This increased the value of the "
          "house by 150%. How much profit did he make, in dollars?"},
]


def main():
    have = set(available_models())
    missing = [m["name"] for m in MODELS if m["name"] not in have]
    if missing:
        sys.exit(f"Missing models: {missing}. Run: ollama pull <name>")

    print(f"{'model':<14}{'question':<10}{'ok':<4}{'pred':<10}{'secs':>6}{'out_tok':>9}")
    totals = {}
    for m in MODELS:
        name = m["name"]
        generate(name, "Say OK.")  # warm-up: load weights so timing is fair
        n_ok, secs = 0, 0.0
        for item in QUESTIONS:
            g = generate(name, build_prompt(item["q"], item["fmt"], item.get("choices")), system=SYSTEM)
            ok, pred = is_correct(g.text, item["fmt"], item["gold"])
            n_ok += ok
            secs += g.latency_s
            print(f"{name:<14}{item['id']:<10}{'✓' if ok else '✗':<4}{str(pred):<10}"
                  f"{g.latency_s:>6.1f}{g.output_tokens:>9}")
            if not ok:
                tail = g.text.strip().replace("\n", " ")[-160:]
                print(f"{'':<14}  ↳ …{tail}")
        totals[name] = (n_ok, secs / len(QUESTIONS))

    print("\nSummary and projected labeling time for 1,000 questions:")
    grand = 0.0
    for name, (n_ok, avg) in totals.items():
        est_h = avg * 1000 / 3600
        grand += est_h
        print(f"  {name:<14} {n_ok}/{len(QUESTIONS)} correct   avg {avg:5.1f}s/q   ≈ {est_h:4.1f} h")
    print(f"  {'all models':<14} ≈ {grand:.1f} h total (math-heavy mixes run longer)")


if __name__ == "__main__":
    main()
