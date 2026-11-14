"""GreenRoute cascade: try a small model first, keep its answer only if the evidence says so.

For each stage (e.g. tiny, then small):
  1. The model answers and we read how confident it was in its final answer
     (joint probability of the answer tokens; see grading.answer_confidence).
  2. Evidence: the k benchmark questions on which this same model had the
     closest confidence, and whether it got each of them right.
  3. Keep the answer if at least `min_correct` of those k were right;
     otherwise escalate to the next model. The last model's answer is always kept.

Faithfulness by construction: `accept()` sees only the cited evidence (the
correctness of those k past answers), so the explanation is the decision rule.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from .config import MODELS, TIERS

BIG = len(TIERS) - 1


@dataclass
class Stage:
    tier: str
    model: str
    confidence: float
    evidence_ids: list[str]
    evidence_confidence: tuple[float, float]  # range of confidences among the evidence
    n_correct: int
    k: int
    min_correct: int
    accepted: bool
    answer: str | None = None


@dataclass
class CascadeDecision:
    tier: str
    model: str
    stages: list[Stage] = field(default_factory=list)
    explanation: str = ""
    answer: str | None = None
    # one entry per model actually run: (tier, prompt+output tokens, seconds)
    runs: list[tuple[str, int, float]] = field(default_factory=list)

    def compute_vs_largest(self) -> float:
        """Compute used (2 x params x tokens, summed over every model run) relative to
        answering with only the largest model. When the largest model never ran, its
        token count is estimated from the final answer's."""
        params = {m["tier"]: m["params_b"] for m in MODELS}
        used = sum(params[t] * n for t, n, _ in self.runs)
        big = next((n for t, n, _ in self.runs if t == TIERS[BIG]), self.runs[-1][1])
        return used / (params[TIERS[BIG]] * big)


def accept(evidence_correct: np.ndarray, min_correct: int) -> bool:
    """The whole acceptance policy, as a pure function of the cited evidence."""
    return int(np.sum(evidence_correct)) >= min_correct


def explain(d: CascadeDecision) -> str:
    parts = []
    for s in d.stages:
        lo, hi = s.evidence_confidence
        ev = (f"on the {s.k} benchmark questions where {s.tier} was similarly confident "
              f"({lo:.0%}-{hi:.0%}), it was right {s.n_correct}/{s.k} times "
              f"(bar: {s.min_correct}/{s.k})")
        verdict = "so its answer was kept" if s.accepted else "so the question was escalated"
        parts.append(f"{s.tier} ({s.model}) answered with {s.confidence:.0%} confidence; {ev}, {verdict}.")
    if not d.stages or not d.stages[-1].accepted:
        parts.append(f"{d.tier} ({d.model}) gave the final answer as the safest option.")
    return " ".join(p[0].upper() + p[1:] for p in parts)


class CascadeRouter:
    """Holds the calibration evidence: per cheap tier, each benchmark question's
    confidence and correctness."""

    def __init__(self, ids: list[str], confidence: np.ndarray, correct: np.ndarray,
                 stages=(0, 1), k: int = 50, min_accuracy: float = 0.9):
        self.ids = list(ids)
        self.conf = confidence   # [n, n_cheap_tiers]
        self.correct = correct   # [n, n_cheap_tiers] bool
        self.stages, self.k = tuple(stages), k
        self.min_correct = math.ceil(min_accuracy * k - 1e-9)  # guard: 0.9 * 50 = 45.000000000000004
        self._row = {qid: i for i, qid in enumerate(self.ids)}

    def evidence(self, tier: int, c: float, exclude_id: str | None = None) -> np.ndarray:
        dist = np.abs(self.conf[:, tier] - c)
        if exclude_id in self._row:
            dist[self._row[exclude_id]] = np.inf
        return np.argsort(dist, kind="stable")[: self.k]

    def judge(self, tier: int, c: float, exclude_id: str | None = None) -> Stage:
        idx = self.evidence(tier, c, exclude_id)
        ev = self.correct[idx, tier]
        cs = self.conf[idx, tier]
        return Stage(TIERS[tier], MODELS[tier]["name"], float(c), [self.ids[i] for i in idx],
                     (float(cs.min()), float(cs.max())), int(ev.sum()), self.k,
                     self.min_correct, accept(ev, self.min_correct))

    def run(self, question: str, fmt: str = "open", choices=None) -> CascadeDecision:
        """Answer a live question, escalating through the cascade."""
        from .grading import SYSTEM, answer_confidence, build_prompt
        from .llm import generate

        prompt = build_prompt(question, fmt, choices)
        d = CascadeDecision(tier=TIERS[BIG], model=MODELS[BIG]["name"])
        for t in self.stages:
            g = generate(MODELS[t]["name"], prompt, system=SYSTEM, logprobs=True)
            st = self.judge(t, answer_confidence(g.token_logprobs)["answer_prob"])
            st.answer = g.text
            d.stages.append(st)
            d.runs.append((st.tier, g.prompt_tokens + g.output_tokens, g.latency_s))
            if st.accepted:
                d.tier, d.model, d.answer = st.tier, st.model, g.text
                break
        else:
            g = generate(MODELS[BIG]["name"], prompt, system=SYSTEM)
            d.answer = g.text
            d.runs.append((TIERS[BIG], g.prompt_tokens + g.output_tokens, g.latency_s))
        d.explanation = explain(d)
        return d


def build_cascade(**kwargs) -> CascadeRouter:
    from .data import load_questions
    from .evaluation import load_confidence
    ids = [q["id"] for q in load_questions()]
    conf, correct = load_confidence(ids)
    return CascadeRouter(ids, conf, correct, **kwargs)
