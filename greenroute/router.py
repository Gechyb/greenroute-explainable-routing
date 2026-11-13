"""GreenRoute: k-nearest-neighbour router with evidence-based explanations.

Decision rule (for a new question):
  1. Find the k most similar benchmark questions (cosine similarity).
  2. If even the closest one is not very similar, the evidence is weak:
     send to the largest model ("unfamiliar").
  3. Otherwise, walk the tiers smallest -> largest and pick the first one that
     answered at least `min_correct` of the k neighbours correctly ("evidence").
  4. If no smaller tier clears that bar, play it safe with the largest model
     ("mixed_evidence").

Faithfulness by construction: `decide()` receives only the neighbour evidence
it reports in the explanation. Nothing else (no hidden score, no extra model)
can influence the outcome, so the explanation *is* the decision procedure.
`decide()` is public so experiments can perturb the evidence and verify the
decision changes accordingly.
"""

from dataclasses import dataclass, field

import numpy as np

from .config import LARGEST_TIER, MODELS, TIERS

MODEL_OF = {m["tier"]: m["name"] for m in MODELS}


@dataclass
class Neighbor:
    id: str
    text: str
    similarity: float
    outcomes: dict[str, bool | None]  # tier -> correct? (None = not labeled yet)


@dataclass
class Decision:
    tier: str
    model: str
    reason: str  # "evidence" | "mixed_evidence" | "unfamiliar"
    tally: dict[str, tuple[int, int]]  # tier -> (n_correct, n_labeled) among neighbours
    neighbors: list[Neighbor] = field(default_factory=list)
    min_correct: int = 0
    min_similarity: float = 0.0
    explanation: str = ""


def decide(neighbors: list[Neighbor], min_correct: int, min_similarity: float) -> Decision:
    """The entire routing policy, as a pure function of the evidence."""
    tally = {}
    for t in TIERS:
        vals = [n.outcomes.get(t) for n in neighbors]
        known = [v for v in vals if v is not None]
        tally[t] = (sum(known), len(known))

    k = len(neighbors)
    top_sim = max((n.similarity for n in neighbors), default=0.0)

    if top_sim < min_similarity:
        tier, reason = LARGEST_TIER, "unfamiliar"
    else:
        tier, reason = LARGEST_TIER, "mixed_evidence"
        for t in TIERS[:-1]:
            ok, n = tally[t]
            if n == k and ok >= min_correct:  # require full evidence for a cheap tier
                tier, reason = t, "evidence"
                break

    d = Decision(tier=tier, model=MODEL_OF[tier], reason=reason, tally=tally,
                 neighbors=neighbors, min_correct=min_correct, min_similarity=min_similarity)
    d.explanation = explain(d)
    return d


def explain(d: Decision) -> str:
    k = len(d.neighbors)
    scores = ", ".join(f"{t} {ok}/{n}" for t, (ok, n) in d.tally.items() if n)
    if d.reason == "evidence":
        ok, _ = d.tally[d.tier]
        smaller = [t for t in TIERS[: TIERS.index(d.tier)]]
        why_not = ""
        if smaller:
            why_not = " Smaller models fell short: " + ", ".join(
                f"{t} {d.tally[t][0]}/{d.tally[t][1]}" for t in smaller) + "."
        return (f"Sent to {d.tier} ({d.model}). On the {k} most similar benchmark questions, "
                f"{d.tier} got {ok}/{k} right, meeting the bar of {d.min_correct}/{k}.{why_not} "
                f"[All results: {scores}]")
    if d.reason == "mixed_evidence":
        return (f"Sent to {d.tier} ({d.model}) as a safeguard. No smaller model got at least "
                f"{d.min_correct}/{k} of the similar questions right ({scores}), so the evidence "
                f"was too mixed to trust a smaller model.")
    top = max((n.similarity for n in d.neighbors), default=0.0)
    return (f"Sent to {d.tier} ({d.model}) as a safeguard. The closest benchmark question is "
            f"only {top:.2f} similar (bar: {d.min_similarity:.2f}), so past results are not "
            f"good evidence for this question.")


class KNNRouter:
    def __init__(self, ids: list[str], texts: list[str], embeddings: np.ndarray,
                 outcomes: np.ndarray, k: int = 5, min_correct: int = 4,
                 min_similarity: float = 0.30):
        assert len(ids) == len(texts) == len(embeddings) == len(outcomes)
        self.ids, self.texts = list(ids), list(texts)
        self.emb = embeddings
        self.outcomes = outcomes  # [n, n_tiers], 1/0/NaN
        self.k, self.min_correct, self.min_similarity = k, min_correct, min_similarity
        self._row = {qid: i for i, qid in enumerate(self.ids)}

    def neighbors(self, q_emb: np.ndarray, exclude_ids: set[str] = frozenset()) -> list[Neighbor]:
        sims = self.emb @ q_emb
        for qid in exclude_ids:  # leave-one-out evaluation: never use the question itself
            if qid in self._row:
                sims[self._row[qid]] = -np.inf
        top = np.argsort(-sims)[: self.k]
        out = []
        for i in top:
            outcomes = {t: (None if np.isnan(v) else bool(v)) for t, v in zip(TIERS, self.outcomes[i])}
            out.append(Neighbor(self.ids[i], self.texts[i], float(sims[i]), outcomes))
        return out

    def route_embedding(self, q_emb: np.ndarray, exclude_ids: set[str] = frozenset()) -> Decision:
        return decide(self.neighbors(q_emb, exclude_ids), self.min_correct, self.min_similarity)

    def route(self, text: str) -> Decision:
        from .embed import embed
        return self.route_embedding(embed([text])[0])


def build_router(**kwargs) -> KNNRouter:
    """Router over the full labeled benchmark (whatever has been labeled so far)."""
    from .data import load_outcomes, load_questions, question_text
    from .embed import embed_cached

    qs = load_questions()
    ids = [q["id"] for q in qs]
    texts = [question_text(q) for q in qs]
    return KNNRouter(ids, texts, embed_cached(texts), load_outcomes(ids), **kwargs)
