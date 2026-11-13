"""Thin client for the local Ollama server with timing and token accounting."""

import time
from dataclasses import dataclass

import requests

from .config import GEN_OPTIONS, OLLAMA_URL, REQUEST_TIMEOUT_S


@dataclass
class Generation:
    model: str
    text: str
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    # Per output token: (token text, log-probability). Only filled when requested.
    token_logprobs: list[tuple[str, float]] | None = None


def generate(model: str, prompt: str, system: str | None = None, retries: int = 2,
             logprobs: bool = False) -> Generation:
    payload = {"model": model, "prompt": prompt, "stream": False, "options": GEN_OPTIONS}
    if logprobs:
        payload["logprobs"] = True
    if system:
        payload["system"] = system
    last_err = None
    for attempt in range(retries + 1):
        try:
            t0 = time.perf_counter()
            r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=REQUEST_TIMEOUT_S)
            r.raise_for_status()
            data = r.json()
            return Generation(
                model=model,
                text=data.get("response", ""),
                prompt_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
                latency_s=time.perf_counter() - t0,
                token_logprobs=([(t["token"], t["logprob"]) for t in data.get("logprobs") or []]
                                if logprobs else None),
            )
        except (requests.RequestException, ValueError) as e:  # network / bad JSON
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Ollama call to {model} failed after {retries + 1} tries: {last_err}")


def available_models() -> list[str]:
    r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]
