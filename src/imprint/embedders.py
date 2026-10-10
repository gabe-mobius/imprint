"""Embedder interface.

Imprint needs one thing from an embedder: turn text into a fixed-length vector.
The default is a small local sentence-transformer (all-MiniLM-L6-v2), which runs
on CPU and never sends text anywhere. Any object with `embed(list[str]) ->
array (n, d)` and a stable `name` works.
"""
from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # (n, d)
        ...


class MiniLMEmbedder:
    """Local all-MiniLM-L6-v2 via sentence-transformers (optional dependency)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover - depends on environment
            raise ImportError(
                "The default embedder needs sentence-transformers: "
                "pip install 'imprint-layer[minilm]', or pass your own embedder="
            ) from e
        self.name = f"sentence-transformers/{model_name}"
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(self._model.encode(list(texts), normalize_embeddings=True),
                          dtype=np.float32)


def default_embedder() -> Embedder:
    return MiniLMEmbedder()
