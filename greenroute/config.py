"""Central configuration: the model ladder and generation settings.

All tiers come from one model family (Qwen2.5) so that differences in
accuracy come from size, not from training data or prompt format.
"""

OLLAMA_URL = "http://localhost:11434"

# Ordered smallest -> largest. The router always prefers earlier entries.
# `params_b` (billions of parameters) is the compute proxy: cost of a query is
# roughly params x tokens processed, so relative cost between tiers is the
# params ratio. Latency is measured separately during labeling.
MODELS = [
    {"tier": "tiny",   "name": "qwen2.5:0.5b", "params_b": 0.5},
    {"tier": "small",  "name": "qwen2.5:1.5b", "params_b": 1.5},
    {"tier": "medium", "name": "qwen2.5:3b",   "params_b": 3.0},
    {"tier": "large",  "name": "qwen2.5:7b",   "params_b": 7.0},
]
TIERS = [m["tier"] for m in MODELS]
LARGEST_TIER = TIERS[-1]

# Deterministic decoding so labels are reproducible across reruns.
GEN_OPTIONS = {
    "temperature": 0.0,
    "seed": 42,
    "num_ctx": 2048,
    "num_predict": 384,  # enough for short chain-of-thought on math
}
REQUEST_TIMEOUT_S = 180
