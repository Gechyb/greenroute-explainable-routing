"""Route a question from the command line and show the evidence.

  .venv/bin/python -m scripts.route "What is the capital of France?"
  .venv/bin/python -m scripts.route --answer "A train travels 60 km in 1.5 hours..."
  .venv/bin/python -m scripts.route            # interactive: type questions, blank line quits
"""

import argparse

from greenroute.config import TIERS
from greenroute.router import build_router


def show(question: str, d, answer: bool):
    print(f"\n→ {d.tier.upper()}  ({d.model})   reason: {d.reason}")
    print(f"  {d.explanation}\n")
    print(f"  {'sim':>5}  " + "  ".join(f"{t:<6}" for t in TIERS) + "  similar benchmark question")
    for n in d.neighbors:
        marks = "  ".join(f"{'✓' if v else ('✗' if v is False else '·'):<6}" for v in n.outcomes.values())
        q = n.text.split("\n")[0]
        print(f"  {n.similarity:5.2f}  {marks}  {q[:70] + ('…' if len(q) > 70 else '')}")
    if answer:
        from greenroute.llm import generate
        g = generate(d.model, question)
        print(f"\n  {d.model} answered in {g.latency_s:.1f}s:\n  {g.text.strip()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--answer", action="store_true", help="also run the chosen model")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--min-correct", type=int, default=4)
    ap.add_argument("--min-similarity", type=float, default=0.30)
    args = ap.parse_args()

    router = build_router(k=args.k, min_correct=args.min_correct, min_similarity=args.min_similarity)
    questions = [args.question] if args.question else iter(lambda: input("\nQuestion> ").strip(), "")
    for q in questions:
        show(q, router.route(q), args.answer)


if __name__ == "__main__":
    main()
