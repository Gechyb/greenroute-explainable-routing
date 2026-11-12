"""Build the benchmark question set GreenRoute learns from.

Mix (1,000 questions, chosen to span easy -> hard so routing has room to work):
  - ARC-Easy        150  grade-school science, multiple choice (easy)
  - ARC-Challenge   150  harder science, multiple choice
  - MMLU            400  57 academic subjects, multiple choice (mixed)
  - GSM8K           300  grade-school math word problems, numeric answer

Output: data/questions.jsonl, one question per line:
  {"id", "source", "subject", "fmt", "question", "choices", "gold"}

Run:  .venv/bin/python -m scripts.prepare_data
"""

import json
import random
import re
from pathlib import Path

from datasets import load_dataset

SEED = 42
OUT = Path(__file__).resolve().parent.parent / "data" / "questions.jsonl"
COUNTS = {"arc_easy": 150, "arc_challenge": 150, "mmlu": 400, "gsm8k": 300}
LETTERS = "ABCDEFGHIJ"


def sample(rows, n, rng):
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    return [rows[i] for i in idx[:n]]


def arc(config, n, rng, tag):
    ds = load_dataset("allenai/ai2_arc", config, split="test")
    ds = [r for r in ds if len(r["choices"]["text"]) == 4]  # keep format uniform
    out = []
    for i, r in enumerate(sample(ds, n, rng)):
        labels = r["choices"]["label"]
        out.append({
            "id": f"{tag}-{i:04d}", "source": tag, "subject": "science", "fmt": "mc",
            "question": r["question"], "choices": r["choices"]["text"],
            "gold": LETTERS[labels.index(r["answerKey"])],
        })
    return out


def mmlu(n, rng):
    ds = load_dataset("cais/mmlu", "all", split="test")
    out = []
    for i, r in enumerate(sample(ds, n, rng)):
        out.append({
            "id": f"mmlu-{i:04d}", "source": "mmlu", "subject": r["subject"], "fmt": "mc",
            "question": r["question"], "choices": r["choices"], "gold": LETTERS[r["answer"]],
        })
    return out


def gsm8k(n, rng):
    ds = load_dataset("openai/gsm8k", "main", split="test")
    out = []
    for i, r in enumerate(sample(ds, n, rng)):
        gold = r["answer"].split("####")[-1].strip().replace(",", "")
        out.append({
            "id": f"gsm8k-{i:04d}", "source": "gsm8k", "subject": "math", "fmt": "numeric",
            "question": r["question"], "choices": None, "gold": float(re.sub(r"[^\d.\-]", "", gold)),
        })
    return out


def main():
    rng = random.Random(SEED)
    rows = (arc("ARC-Easy", COUNTS["arc_easy"], rng, "arc_easy")
            + arc("ARC-Challenge", COUNTS["arc_challenge"], rng, "arc_challenge")
            + mmlu(COUNTS["mmlu"], rng)
            + gsm8k(COUNTS["gsm8k"], rng))
    rng.shuffle(rows)  # interleave sources so a partial run is still a fair sample
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    by_src = {}
    for r in rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    print(f"Wrote {len(rows)} questions to {OUT}: {by_src}")


if __name__ == "__main__":
    main()
