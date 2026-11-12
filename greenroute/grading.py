"""Prompt templates and automatic answer checking.

Two question formats cover the benchmark mix:
  - "mc":      multiple choice, gold answer is a letter (A-D)
  - "numeric": math word problems, gold answer is a number
"""

import re

SYSTEM = "You are a careful assistant. Follow the answer format exactly."


def build_prompt(question: str, fmt: str, choices: list[str] | None = None) -> str:
    if fmt == "mc":
        letters = "ABCDEFGHIJ"
        opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices or []))
        return (
            f"{question}\n\n{opts}\n\n"
            "Think briefly if needed, then finish with a final line of the form "
            "'Answer: X' where X is the letter of the correct option."
        )
    if fmt == "numeric":
        return (
            f"{question}\n\n"
            "Solve step by step, then finish with a final line of the form "
            "'Answer: N' where N is just the number."
        )
    raise ValueError(f"unknown format {fmt!r}")


_ANSWER_LINE = re.compile(r"answer\s*(?:is)?\s*[:：]?\s*\(?\**\s*([A-J])\b", re.I)
_LONE_LETTER = re.compile(r"(?:^|[\s(\[*])([A-J])(?:[)\].:*]|\s|$)")
_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_mc(text: str) -> str | None:
    hits = _ANSWER_LINE.findall(text)
    if hits:
        return hits[-1].upper()
    hits = _LONE_LETTER.findall(text.strip())
    return hits[-1].upper() if hits else None


def extract_numeric(text: str) -> float | None:
    m = re.findall(r"answer\s*[:：]?\s*(-?\s*\$?\s*-?\d[\d,]*\.?\d*)", text, re.I)
    candidates = m or _NUMBER.findall(text)
    for raw in reversed(candidates):
        try:
            return float(re.sub(r"[\s$,]", "", raw).rstrip("."))
        except ValueError:
            continue
    return None


def is_correct(text: str, fmt: str, gold) -> tuple[bool, object]:
    """Return (correct, parsed_prediction)."""
    if fmt == "mc":
        pred = extract_mc(text)
        return pred == str(gold).upper(), pred
    if fmt == "numeric":
        pred = extract_numeric(text)
        return pred is not None and abs(pred - float(gold)) < 1e-6, pred
    raise ValueError(f"unknown format {fmt!r}")
