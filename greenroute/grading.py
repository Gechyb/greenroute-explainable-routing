"""Prompt templates and automatic answer checking.

Two question formats cover the benchmark mix:
  - "mc":      multiple choice, gold answer is a letter (A-D)
  - "numeric": math word problems, gold answer is a number
"""

import math
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
    if fmt == "open":  # free-form questions typed into the demo
        return (
            f"{question}\n\n"
            "Think briefly if needed, then finish with a final line of the form "
            "'Answer: <short answer>'."
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


# ---------------------------------------------------------------- confidence

_ANSWER_CUE = re.compile(r"answer\s*(?:is)?\s*[:：]?", re.I)


def answer_confidence(token_logprobs: list[tuple[str, float]]) -> dict:
    """How sure the model was of its final answer, from its own token probabilities.

    answer_prob: joint probability of the answer tokens (the letter or the digits)
                 after the last "Answer:" cue; 0.0 if the model never gave one.
    mean_logprob: average log-probability over the whole response (fallback signal).
    """
    if not token_logprobs:
        return {"answer_prob": 0.0, "mean_logprob": None, "answer_found": False}
    toks = [t for t, _ in token_logprobs]
    lps = [lp for _, lp in token_logprobs]
    mean_lp = sum(lps) / len(lps)

    starts, pos = [], 0
    for t in toks:
        starts.append(pos)
        pos += len(t)
    text = "".join(toks)
    cues = list(_ANSWER_CUE.finditer(text))
    if not cues:
        return {"answer_prob": 0.0, "mean_logprob": mean_lp, "answer_found": False}

    # Answer tokens: alphanumeric tokens after the cue, up to the end of that line.
    cue_end = cues[-1].end()
    line_end = text.find("\n", cue_end)
    line_end = len(text) if line_end == -1 else line_end
    total, n = 0.0, 0
    for t, st, lp in zip(toks, starts, lps):
        if st + len(t) <= cue_end or st >= line_end:
            continue
        if any(ch.isalnum() for ch in t):
            total += lp
            n += 1
    if n == 0:
        return {"answer_prob": 0.0, "mean_logprob": mean_lp, "answer_found": False}
    return {"answer_prob": math.exp(total), "mean_logprob": mean_lp, "answer_found": True}
