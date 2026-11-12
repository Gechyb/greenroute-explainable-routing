# GreenRoute: Explainable, Energy-Aware Routing for LLMs

GreenRoute sends each question to the smallest of three language models (tiny, small, medium) that is likely to answer it correctly. Each decision comes with its evidence: the most similar benchmark questions and which models got them right. When the evidence is mixed, GreenRoute falls back to the largest model.

## Models

One family at three sizes (Qwen2.5 via Ollama), so accuracy differences come from size alone:

| Tier   | Model          | Params |
|--------|----------------|--------|
| tiny   | `qwen2.5:0.5b` | 0.5B   |
| small  | `qwen2.5:1.5b` | 1.5B   |
| medium | `qwen2.5:7b`   | 7B     |

## Setup (macOS)

```bash
brew install ollama && brew services start ollama
ollama pull qwen2.5:0.5b && ollama pull qwen2.5:1.5b && ollama pull qwen2.5:7b
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m scripts.smoke_test
```

## Layout

```
greenroute/   config.py (model ladder), llm.py (Ollama client), grading.py (prompts + answer checking)
scripts/      smoke_test.py, and later the labeling / evaluation scripts
data/labels/  per-question correctness for each model (the expensive output, kept in git)
results/      metrics tables
figures/      poster figures
```
