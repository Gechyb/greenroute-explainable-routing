"""Unit tests for the confidence cascade, including counterfactual faithfulness."""

import numpy as np

from greenroute.cascade import CascadeDecision, CascadeRouter, accept, explain
from greenroute.grading import answer_confidence


def make_router(min_accuracy=0.8, k=5, stages=(0, 1)):
    # 20 questions; tiny is right only when confident (>0.5); small is right on all.
    conf = np.column_stack([np.linspace(0, 1, 20), np.linspace(0, 1, 20)])
    correct = np.column_stack([conf[:, 0] > 0.5, np.ones(20, bool)])
    return CascadeRouter([f"q{i}" for i in range(20)], conf, correct, stages, k, min_accuracy)


def test_min_correct_has_no_float_rounding_error():
    assert make_router(min_accuracy=0.9, k=50).min_correct == 45


def test_confident_tiny_answer_is_kept_unsure_one_escalates():
    r = make_router()
    assert r.judge(0, 0.95).accepted
    assert not r.judge(0, 0.10).accepted
    assert r.judge(1, 0.10).accepted  # small is always right in this toy pool


def test_leave_one_out_excludes_the_question_itself():
    r = make_router()
    st = r.judge(0, r.conf[19, 0], exclude_id="q19")
    assert "q19" not in st.evidence_ids and len(st.evidence_ids) == 5


def test_faithfulness_flipping_cited_evidence_flips_decision():
    ev = np.array([True, True, True, True, False])
    assert accept(ev, 4)
    ev[0] = False
    assert not accept(ev, 4)


def test_explanation_cites_counts_and_outcome():
    r = make_router()
    d = CascadeDecision(tier="large", model="qwen2.5:7b", stages=[r.judge(0, 0.1)])
    text = explain(d)
    assert "escalated" in text and "/5" in text and "safest option" in text


def test_answer_confidence_reads_answer_tokens_only():
    lp = [("Reasoning", -3.0), ("\n", 0.0), ("Answer", -0.1), (":", 0.0), (" C", np.log(0.8))]
    assert abs(answer_confidence(lp)["answer_prob"] - 0.8) < 1e-9
    assert answer_confidence([("no", -1.0), (" format", -1.0)])["answer_prob"] == 0.0
