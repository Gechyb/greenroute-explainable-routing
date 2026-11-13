"""Labeling run: every model answers every question; we record whether it was right.

This is the evidence base the router searches. It is resumable: each result is
appended to data/labels/labels.jsonl the moment it is graded, and on restart any
(question, model) pair already in the file is skipped.

Models run one at a time, smallest first, so each model stays loaded in memory
and the cheap models finish early (partial results are usable).

Run:
  .venv/bin/python -m scripts.label                # full run
  .venv/bin/python -m scripts.label --limit 10     # quick test on 10 questions
  .venv/bin/python -m scripts.label --models qwen2.5:7b   # one model only
"""

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from greenroute.config import MODELS
from greenroute.grading import SYSTEM, answer_confidence, build_prompt, is_correct
from greenroute.llm import available_models, generate

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "questions.jsonl"
LABELS = ROOT / "data" / "labels" / "labels.jsonl"
CONFIDENCE = ROOT / "data" / "labels" / "confidence.jsonl"


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line cut off by a crash; it will be redone
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    ap.add_argument("--models", nargs="*", default=[m["name"] for m in MODELS])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--confidence", action="store_true",
                    help="also record token log-probs and answer confidence (cascade run); "
                         "defaults to the tiny and small models and data/labels/confidence.jsonl")
    args = ap.parse_args()
    if args.confidence and args.models == [m["name"] for m in MODELS]:
        args.models = [m["name"] for m in MODELS[:-1]]  # the largest model is never cascaded past
    args.out = args.out or (CONFIDENCE if args.confidence else LABELS)

    questions = load_jsonl(QUESTIONS)[: args.limit]
    if not questions:
        sys.exit("No questions found. Run: .venv/bin/python -m scripts.prepare_data")
    missing = set(args.models) - set(available_models())
    if missing:
        sys.exit(f"Missing models: {sorted(missing)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["model"]) for r in load_jsonl(args.out)}

    for model in args.models:
        todo = [q for q in questions if (q["id"], model) not in done]
        if not todo:
            print(f"{model}: already complete")
            continue
        n_ok = 0
        t0 = time.time()
        with args.out.open("a") as f:
            bar = tqdm(todo, desc=model, unit="q")
            for i, q in enumerate(bar, 1):
                g = generate(model, build_prompt(q["question"], q["fmt"], q["choices"]), system=SYSTEM,
                             logprobs=args.confidence)
                ok, pred = is_correct(g.text, q["fmt"], q["gold"])
                n_ok += ok
                rec = {
                    "id": q["id"], "model": model, "correct": bool(ok), "pred": pred,
                    "gold": q["gold"], "latency_s": round(g.latency_s, 3),
                    "prompt_tokens": g.prompt_tokens, "output_tokens": g.output_tokens,
                    "response": g.text,
                }
                if args.confidence:
                    rec.update(answer_confidence(g.token_logprobs))
                f.write(json.dumps(rec) + "\n")
                f.flush()
                bar.set_postfix(acc=f"{n_ok / i:.2f}")
        print(f"{model}: {n_ok}/{len(todo)} correct in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
