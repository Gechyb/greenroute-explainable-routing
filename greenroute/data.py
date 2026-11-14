"""Load benchmark questions and the per-model correctness labels."""

import json
from pathlib import Path

import numpy as np

from .config import MODELS

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = ROOT / "data" / "questions.jsonl"
LABELS = ROOT / "data" / "labels" / "labels.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        with path.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # half-written line from an in-progress run
    return rows


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return _read_jsonl(path)


def question_text(q: dict) -> str:
    """Text used for similarity search: the question plus its answer options."""
    if q.get("choices"):
        return q["question"] + "\nOptions: " + " | ".join(q["choices"])
    return q["question"]


def load_outcomes(ids: list[str], path: Path = LABELS) -> np.ndarray:
    """Matrix [n_questions, n_tiers]: 1.0 correct, 0.0 wrong, NaN not yet labeled.

    Columns follow config.MODELS order (smallest -> largest).
    """
    col = {m["name"]: j for j, m in enumerate(MODELS)}
    row = {qid: i for i, qid in enumerate(ids)}
    out = np.full((len(ids), len(MODELS)), np.nan)
    for r in _read_jsonl(path):
        i, j = row.get(r["id"]), col.get(r["model"])
        if i is not None and j is not None:
            out[i, j] = 1.0 if r["correct"] else 0.0
    return out
