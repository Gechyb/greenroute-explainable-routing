"""GreenRoute live demo: type a question, see which model answered it and why.

  .venv/bin/python -m scripts.demo          # opens http://127.0.0.1:7860

Needs Ollama running with the models pulled, and the labeled benchmark
(data/labels/*.jsonl) as the evidence base.
"""

import json
import re
from pathlib import Path

import gradio as gr
import numpy as np

from greenroute.cascade import CascadeRouter
from greenroute.config import MODELS, TIERS
from greenroute.data import load_questions
from greenroute.evaluation import STAGE_SETS, load_confidence
from greenroute.router import build_router

ROOT = Path(__file__).resolve().parent.parent
SIZE = {m["tier"]: f"{m['params_b']:g}B" for m in MODELS}

# ---------------------------------------------------------------- evidence
QS = load_questions()
IDS = [q["id"] for q in QS]
QTEXT = {q["id"]: q["question"] for q in QS}
CONF, CORRECT = load_confidence(IDS, strict=False)
READY = [t for t in range(CONF.shape[1]) if not np.isnan(CONF[:, t]).any()]  # fully labeled tiers
ORDERS = [name for name, st in STAGE_SETS.items() if all(t in READY for t in st)]

DEFAULT = {"order": ORDERS[-1] if ORDERS else None, "k": 50, "min_accuracy": 0.8}
metrics = ROOT / "results" / "metrics.json"
if metrics.exists():
    c = json.loads(metrics.read_text())["config"]["cascade"]
    if c["stages"] in ORDERS:
        DEFAULT = {"order": c["stages"], "k": c["k"], "min_accuracy": c["min_accuracy"]}

SIM_ROUTER = None  # built lazily: loads the embedding model


def final_line(text: str) -> str:
    m = list(re.finditer(r"answer\s*[:：]\s*(.+)", text or "", re.I))
    return m[-1].group(1).strip() if m else (text or "").strip().split("\n")[-1]


# ----------------------------------------------------------------- cascade
def run_cascade(question, order, k, min_accuracy):
    if not question.strip():
        return "Type a question first.", "", None, ""
    router = CascadeRouter(IDS, CONF, CORRECT, STAGE_SETS[order], int(k), float(min_accuracy))
    d = router.run(question)

    secs = sum(s for _, _, s in d.runs)
    summary = (f"### Answered by **{d.tier}** ({d.model}, {SIZE[d.tier]})\n\n"
               f"**Answer:** {final_line(d.answer)}\n\n"
               f"Models run: {' → '.join(t for t, _, _ in d.runs)}  ·  "
               f"time {secs:.1f}s  ·  compute ≈ **{d.compute_vs_largest():.0%}** of answering with "
               f"the {SIZE[TIERS[-1]]} model alone")

    rows = []
    for s in d.stages:
        lo, hi = s.evidence_confidence
        rows.append([f"{s.tier} ({SIZE[s.tier]})", f"{s.confidence:.1%}", f"{lo:.1%} – {hi:.1%}",
                     f"{s.n_correct}/{s.k}", f"{s.min_correct}/{s.k}",
                     "✅ kept" if s.accepted else "⬆️ escalated"])
    if not d.stages or not d.stages[-1].accepted:
        rows.append([f"{TIERS[-1]} ({SIZE[TIERS[-1]]})", "—", "—", "—", "—", "🛡️ final answer"])

    examples = ""
    if d.stages:
        s = d.stages[-1]
        ex = [i for i in s.evidence_ids[:5]]
        examples = (f"**Examples of the {s.k} past questions where {s.tier} was similarly confident** "
                    f"(✓ = it was right):\n\n" + "\n".join(
                        f"- {'✓' if CORRECT[IDS.index(i), TIERS.index(s.tier)] else '✗'} "
                        f"{QTEXT[i][:140]}{'…' if len(QTEXT[i]) > 140 else ''}" for i in ex))
    trace = "\n\n".join(f"**{s.tier}** ({s.confidence:.0%} confident):\n\n{s.answer}" for s in d.stages)
    if not d.stages or not d.stages[-1].accepted:
        trace += f"\n\n**{d.tier}** (final):\n\n{d.answer}"
    return summary, d.explanation, rows, examples + "\n\n---\n\n**Full model outputs**\n\n" + trace


# --------------------------------------------------------- similarity router
def run_similarity(question):
    global SIM_ROUTER
    if not question.strip():
        return "Type a question first.", None
    if SIM_ROUTER is None:
        SIM_ROUTER = build_router()
    d = SIM_ROUTER.route(question)
    rows = [[f"{n.similarity:.2f}", *["✓" if v else ("✗" if v is False else "·") for v in n.outcomes.values()],
             n.text.split("\n")[0][:110]] for n in d.neighbors]
    return f"### Would send to **{d.tier}** ({d.model})\n\n{d.explanation}", rows


# ---------------------------------------------------------------------- UI
EXAMPLES = [
    "What is the chemical symbol for gold?",
    "A baker makes 24 muffins and sells them in boxes of 4 at $3 per box. How much money does she make?",
    "Why does ice float on water?",
    "In a perfectly competitive market in long-run equilibrium, how does price compare to average total cost?",
    "A train leaves at 2:45 pm and the trip takes 3 hours 50 minutes. When does it arrive?",
]

with gr.Blocks(title="GreenRoute") as app:
    gr.Markdown(
        "# 🌱 GreenRoute\n"
        "Answers each question with the **smallest model that the evidence says will get it right**, "
        "and shows that evidence. Small models answer first; an answer is kept only if, on past "
        "benchmark questions where that model was about as confident, it was usually right. "
        "Otherwise the question moves up to a bigger model.")
    with gr.Tab("Ask GreenRoute"):
        q = gr.Textbox(label="Your question", lines=2, placeholder="Ask anything…")
        with gr.Accordion("Router settings", open=False):
            order = gr.Dropdown(ORDERS, value=DEFAULT["order"], label="Models to try, in order")
            bar = gr.Slider(0.5, 1.0, value=DEFAULT["min_accuracy"], step=0.02,
                            label="Accuracy bar: keep a small model's answer only if it was right on at "
                                  "least this share of similar-confidence past questions")
            k = gr.Slider(10, 100, value=DEFAULT["k"], step=5, label="Past questions used as evidence (k)")
        go = gr.Button("Route and answer", variant="primary")
        gr.Examples(EXAMPLES, inputs=q)
        summary = gr.Markdown()
        explanation = gr.Textbox(label="Why (the explanation is the decision rule)", lines=3)
        table = gr.Dataframe(headers=["model", "its confidence", "evidence confidence range",
                                      "evidence: right", "bar", "decision"], label="Decision trail")
        with gr.Accordion("Evidence and full model outputs", open=False):
            details = gr.Markdown()
        go.click(run_cascade, [q, order, k, bar], [summary, explanation, table, details])
        q.submit(run_cascade, [q, order, k, bar], [summary, explanation, table, details])

    with gr.Tab("Compare: similar-questions router"):
        gr.Markdown("The first design: route *before* answering, from the most similar benchmark "
                    "questions. In our evaluation, topic similarity barely predicted difficulty.")
        q2 = gr.Textbox(label="Your question", lines=2)
        go2 = gr.Button("Show routing decision")
        out2 = gr.Markdown()
        tab2 = gr.Dataframe(headers=["similarity", *TIERS, "similar benchmark question"])
        go2.click(run_similarity, q2, [out2, tab2])

    gr.Markdown("<small>Evidence base: 1,000 benchmark questions (ARC, MMLU, GSM8K) answered by "
                "Qwen2.5 models running locally via Ollama. Confidence evidence comes from benchmark-style "
                "prompts, so it is a guide for free-form questions, not a guarantee.</small>")

if __name__ == "__main__":
    app.launch()
