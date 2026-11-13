"""Sentence embeddings for similarity search, cached to disk.

Uses all-MiniLM-L6-v2: small (~90 MB), fast on CPU, and its embedding cost is
negligible next to running even the tiny LLM, so routing overhead stays small.
"""

import hashlib
from pathlib import Path

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer  # slow import; defer
        _model = SentenceTransformer(EMBED_MODEL, device="cpu")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """Unit-normalized embeddings, so dot product == cosine similarity."""
    vecs = _get_model().encode(texts, batch_size=64, normalize_embeddings=True,
                               show_progress_bar=len(texts) > 200)
    return np.asarray(vecs, dtype=np.float32)


def embed_cached(texts: list[str]) -> np.ndarray:
    key = hashlib.sha256((EMBED_MODEL + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{key}.npy"
    if path.exists():
        return np.load(path)
    vecs = embed(texts)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs
