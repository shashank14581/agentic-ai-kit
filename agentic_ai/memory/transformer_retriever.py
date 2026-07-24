"""Local transformer-backed retrieval for conversational memory.

The retriever stores original user messages rather than asking a generative
LLM to rewrite them as facts.

One pretrained Sentence Transformer is used to encode both memories and
queries.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TransformerMemoryRetriever:
    """Store and retrieve raw conversational memories semantically."""

    # Share one loaded model across all agent instances.
    _model_cache: dict[str, Any] = {}

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
    ) -> None:
        self.model_name = model_name
        self._vectors: dict[str, np.ndarray] = {}

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize whitespace without rewriting the message."""
        return " ".join(str(text).strip().split())

    @classmethod
    def memory_key(cls, text: str) -> str:
        """Create a stable key for an exact normalized memory."""
        normalized = cls.normalize_text(text).lower()

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    def _get_model(self) -> Any:
        """Load the transformer lazily and reuse it across agents."""
        cached_model = self._model_cache.get(self.model_name)

        if cached_model is not None:
            return cached_model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for transformer memory."
            ) from exc

        model = SentenceTransformer(self.model_name)
        self._model_cache[self.model_name] = model

        return model

    def encode(
        self,
        texts: str | list[str],
    ) -> np.ndarray:
        """Encode text into normalized sentence embeddings."""
        single_input = isinstance(texts, str)
        values = [texts] if single_input else texts

        embeddings = self._get_model().encode(
            values,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        return embeddings[0] if single_input else embeddings

    def add(self, text: str) -> None:
        """Embed and store one message."""
        normalized = self.normalize_text(text)

        if not normalized:
            return

        key = self.memory_key(normalized)

        if key not in self._vectors:
            self._vectors[key] = self.encode(normalized)

    def retrieve(
        self,
        query: str,
        records: list[dict[str, object]],
        top_k: int = 8,
        exclude_latest_query: bool = True,
    ) -> list[dict[str, object]]:
        """Return the records most semantically relevant to the query."""
        if not records or top_k <= 0:
            return []

        normalized_query = self.normalize_text(query)

        if not normalized_query:
            return []

        query_vector = self.encode(normalized_query)

        latest_index = len(records) - 1
        candidates: list[dict[str, object]] = []

        for index, item in enumerate(records):
            text = self.normalize_text(
                str(item.get("fact", ""))
            )

            if not text:
                continue

            # BaseAgent.think() stores the new message before build_context().
            # Do not return the current message as a retrieved memory.
            if (
                exclude_latest_query
                and index == latest_index
                and text.lower() == normalized_query.lower()
            ):
                continue

            key = self.memory_key(text)

            if key not in self._vectors:
                self._vectors[key] = self.encode(text)

            score = float(
                np.dot(
                    query_vector,
                    self._vectors[key],
                )
            )

            candidate = dict(item)
            candidate["retrieval_score"] = score

            candidates.append(candidate)

        candidates.sort(
            key=lambda item: float(
                item.get("retrieval_score", 0.0)
            ),
            reverse=True,
        )

        return candidates[: min(top_k, len(candidates))]

    def prune(
        self,
        active_texts: list[str],
    ) -> None:
        """Remove vectors for records removed by max_facts."""
        active_keys = {
            self.memory_key(text)
            for text in active_texts
            if self.normalize_text(text)
        }

        self._vectors = {
            key: vector
            for key, vector in self._vectors.items()
            if key in active_keys
        }

    def clear(self) -> None:
        """Clear this agent's indexed vectors."""
        self._vectors.clear()
