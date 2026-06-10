"""Embedding helpers using Google's text-embedding model."""

from __future__ import annotations
import os
from google import genai


def embed_texts(texts: list[str], api_key: str | None = None) -> list[list[float]]:
    """Embed a list of strings using Google's embedding model.

    Args:
        texts: List of strings to embed.
        api_key: Gemini API key (falls back to GEMINI_API_KEY env var).

    Returns:
        List of embedding vectors (one per input string).
    """
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("No Gemini API key found. Set GEMINI_API_KEY.")

    client = genai.Client(api_key=key)
    embeddings = []
    for text in texts:
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text,
        )
        embeddings.append(result.embeddings[0].values)
    return embeddings
