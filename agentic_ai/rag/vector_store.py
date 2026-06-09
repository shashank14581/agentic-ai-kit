"""In-memory vector store with cosine similarity search."""

from __future__ import annotations
import math


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b + 1e-9)


class VectorStore:
    """Simple in-memory vector store.

    Usage::

        store = VectorStore()
        store.add(chunks, embeddings)
        results = store.search(query_embedding, top_k=3)
    """

    def __init__(self) -> None:
        self._texts: list[str] = []
        self._vectors: list[list[float]] = []

    def add(self, texts: list[str], vectors: list[list[float]]) -> None:
        """Add a batch of texts and their corresponding embedding vectors."""
        assert len(texts) == len(vectors)
        self._texts.extend(texts)
        self._vectors.extend(vectors)

    def search(self, query_vector: list[float], top_k: int = 3) -> list[dict]:
        """Return the top-k most similar chunks.

        Returns:
            List of dicts with keys ``text`` and ``score``.
        """
        scored = [
            {"text": t, "score": _cosine(query_vector, v)}
            for t, v in zip(self._texts, self._vectors)
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self._texts)

    def __repr__(self) -> str:
        return f"<VectorStore chunks={len(self)}>"
