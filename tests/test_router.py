"""Unit tests for the routing policy, including counterfactual faithfulness checks."""

import numpy as np

from greenroute.router import KNNRouter, Neighbor, decide

K, MIN_OK, MIN_SIM = 5, 4, 0.30


def nbrs(tiny, small, medium=None, sim=0.8):
    """Build K neighbours from per-tier counts of correct answers."""
    medium = K if medium is None else medium
    return [Neighbor(f"q{i}", "q", sim, {"tiny": i < tiny, "small": i < small, "medium": i < medium})
            for i in range(K)]


def test_picks_smallest_tier_that_clears_bar():
    assert decide(nbrs(tiny=5, small=5), MIN_OK, MIN_SIM).tier == "tiny"
    assert decide(nbrs(tiny=4, small=5), MIN_OK, MIN_SIM).tier == "tiny"
    assert decide(nbrs(tiny=3, small=4), MIN_OK, MIN_SIM).tier == "small"


def test_mixed_evidence_falls_back_to_largest():
    d = decide(nbrs(tiny=2, small=3), MIN_OK, MIN_SIM)
    assert (d.tier, d.reason) == ("medium", "mixed_evidence")


def test_unfamiliar_question_falls_back_even_if_tiny_looks_good():
    d = decide(nbrs(tiny=5, small=5, sim=0.10), MIN_OK, MIN_SIM)
    assert (d.tier, d.reason) == ("medium", "unfamiliar")


def test_missing_labels_never_justify_a_cheap_tier():
    n = nbrs(tiny=5, small=5)
    n[0].outcomes["tiny"] = None  # 4/4 known correct, but evidence incomplete
    assert decide(n, MIN_OK, MIN_SIM).tier == "small"


def test_explanation_reports_the_exact_evidence():
    d = decide(nbrs(tiny=4, small=5), MIN_OK, MIN_SIM)
    assert "4/5" in d.explanation and d.tally["tiny"] == (4, 5)


def test_faithfulness_flipping_evidence_flips_decision():
    """Counterfactual: remove one tiny success and the decision must move up a tier.
    Add it back and it must return. The decision depends on the shown evidence only."""
    base = nbrs(tiny=4, small=5)
    assert decide(base, MIN_OK, MIN_SIM).tier == "tiny"
    base[3].outcomes["tiny"] = False
    assert decide(base, MIN_OK, MIN_SIM).tier == "small"
    base[3].outcomes["tiny"] = True
    assert decide(base, MIN_OK, MIN_SIM).tier == "tiny"


def test_knn_router_excludes_query_itself():
    emb = np.eye(6, dtype=np.float32)
    outcomes = np.ones((6, 3))
    r = KNNRouter([f"q{i}" for i in range(6)], ["t"] * 6, emb, outcomes, k=3)
    ids = [n.id for n in r.neighbors(emb[0], exclude_ids={"q0"})]
    assert "q0" not in ids and len(ids) == 3
