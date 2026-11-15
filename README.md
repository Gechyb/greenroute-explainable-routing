# GreenRoute: Explainable, Energy-Aware Routing for LLMs

Many applications send every question to a large language model, even when a much smaller model would answer just as well. That wastes energy and money. Systems that try to save by using smaller models usually don't explain their choices, so nobody can check whether a hard question was sent to a model too weak for it.

GreenRoute answers each question with the **smallest model that the evidence says will get it right**, and shows that evidence. The explanation is the decision rule itself, so it is faithful by construction. We test that claim with counterfactual experiments.

📄 **Paper:** [GreenRoute: Faithful-by-Construction Explanations for Energy-Aware LLM Routing](paper/greenroute_paper.md)

## How it works

Four models from one family (Qwen2.5, run locally with [Ollama](https://ollama.com)), so differences come from size alone:

| Tier   | Model          | Params |
|--------|----------------|--------|
| tiny   | `qwen2.5:0.5b` | 0.5B   |
| small  | `qwen2.5:1.5b` | 1.5B   |
| medium | `qwen2.5:3b`   | 3B     |
| large  | `qwen2.5:7b`   | 7B     |

Every model answered the same 1,000 benchmark questions (ARC-Easy, ARC-Challenge, MMLU, GSM8K). For each answer we recorded whether it was correct and how confident the model was in it. These records are the evidence base.

**Cascade (main design).** A small model answers first. We read its confidence in its final answer, then look up the 50 benchmark questions where that model was about equally confident. If it got enough of those right (the accuracy bar), its answer is kept. Otherwise the question moves up to the next model. The largest model's answer is always kept.

> *"Small answered with 96% confidence; on the 50 benchmark questions where small was similarly confident (94%–98%), it was right 47/50 times (bar: 45/50), so its answer was kept."*

**Similar-questions router (first design, kept for comparison).** Before answering, find the 5 most similar benchmark questions and use the smallest model that got at least 4 of them right. If the evidence is mixed or the question is unfamiliar, use the largest model.

## Reproduce everything

```bash
# 1. Models
brew install ollama && brew services start ollama
for m in qwen2.5:0.5b qwen2.5:1.5b qwen2.5:3b qwen2.5:7b; do ollama pull $m; done

# 2. Python environment
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests           # unit tests
.venv/bin/python -m scripts.smoke_test        # every model answers a few questions

# 3. Data and labels (already in the repo; rerun only to regenerate, several hours)
.venv/bin/python -m scripts.prepare_data      # data/questions.jsonl
.venv/bin/python -m scripts.label             # correctness of every model on every question
.venv/bin/python -m scripts.label --confidence --models qwen2.5:0.5b qwen2.5:1.5b qwen2.5:3b

# 4. Evaluation and figures (about a minute)
.venv/bin/python -m scripts.evaluate          # results/*.json, *.csv
.venv/bin/python -m scripts.make_figures      # figures/*.png, *.svg

# 5. Live demo
.venv/bin/python -m scripts.demo              # http://127.0.0.1:7860
.venv/bin/python -m scripts.route "Your question"   # similar-questions router, command line
```

Decoding is deterministic (temperature 0, fixed seed), so reruns reproduce the labels.

## Evaluation protocol

- **Leave-one-out evidence.** Each benchmark question is routed using only the other 999 as evidence, never its own label.
- **Dev/test split.** Thresholds are tuned on 300 dev questions. All reported numbers are on the other 700 test questions.
- **Compute.** FLOPs proxy: 2 × parameters × (prompt + output tokens), summed over every model a question passed through, relative to always using the largest model. Wall-clock time is reported too.
- **Baselines.** Always-one-model, a perfect oracle (cheapest model that is correct), a black-box classifier router, and a raw-confidence cascade without explanations.
- **Faithfulness.** Flip the outcomes of the evidence an explanation cites and check whether the decision changes; flip the same number of uncited outcomes and check that it does not.

## Results

700 test questions. Compute and time are relative to always using the 7B model.

| Router | Accuracy | Compute | Time |
|---|---|---|---|
| Always tiny (0.5B) | 43.7% | 8.5% | 21% |
| Always small (1.5B) | 65.3% | 25.7% | 49% |
| Always medium (3B) | 77.1% | 48.0% | 71% |
| Always large (7B) | 85.6% | 100% | 100% |
| Perfect oracle | 92.6% | 26.1% | 44% |
| Similar-questions router | 83.0% | 89.0% | 91% |
| Black-box classifier router | 83.1% | 81.1% | 88% |
| Raw-confidence cascade (no explanation) | 84.3% | 103.1% | 114% |
| **GreenRoute cascade (3B → 7B)** | **82.4%** | **74.0%** | **94%** |

**Findings**

1. **The cascade saves about a quarter of the compute for about 3 points of accuracy.** That is 35% of the savings a perfect oracle would get. Across settings it reaches within 2 points of the 7B at 75% compute, and within 5 points at 60% (test-set frontier, `results/cascade_sweep.csv`).
2. **Topic similarity barely predicts difficulty; confidence does better.** Neighbour evidence predicts correctness with AUC 0.57–0.61, against 0.61–0.72 for the model's own confidence (`fig2_signal`). The similar-questions router never beats a random mix of fixed models.
3. **Small models are overconfident, and the evidence step corrects for it.** The 1.5B model is more than 99.9% confident on half the questions, but right on only 79% of those. Raw-confidence thresholds therefore save almost nothing. GreenRoute's evidence-based cascade, which checks how often the model was actually right at that confidence, does much better at the same accuracy. In this design the explanation also improves efficiency.
4. **The explanations are faithful.** Flipping the outcomes of the evidence an explanation cites changes the decision 100% of the time; flipping uncited evidence changes it 0% of the time. A black-box router shown with the same kind of explanation changes only 49% and 7% of the time, so its explanation does not reflect its decision (`fig4_faithfulness`).
5. **Failures.** 36 of 700 test questions (5%) were answered wrongly by the 3B when the 7B would have been right. These cluster in MMLU and GSM8K (`fig3_where`). On 14 questions the 3B was right where the 7B was wrong.
6. **Compute savings are not time savings on this laptop.** The cascade saves 26% of FLOPs but only 6% of wall-clock time: the 3B runs at 71% of the 7B's time here, and escalated questions pay for both models. Savings depend on the hardware.

## Limitations

- Results come from four benchmarks and one model family. They show the idea in a controlled setting, not readiness for deployment.
- Compute is a FLOPs proxy measured on one laptop, not metered energy.
- Confidence evidence comes from benchmark-style prompts, so for free-form questions in the demo it is a guide, not a guarantee.
- No human participants: the study measures system behavior, not how people use the explanations.

## Layout

```
greenroute/   config.py (model ladder), llm.py (Ollama client), grading.py (prompts, answer checking, confidence)
              router.py (similar-questions router), cascade.py (confidence cascade)
              evaluation.py (protocol, sweeps, baselines), faithfulness.py (counterfactual tests)
scripts/      prepare_data, label, evaluate, make_figures, demo, route, smoke_test
tests/        unit tests, including faithfulness checks
data/         questions.jsonl, labels/labels.jsonl, labels/confidence.jsonl
results/      metrics and sweeps        figures/  poster figures
```
