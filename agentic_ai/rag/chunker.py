"""Text chunking utilities for RAG pipelines."""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping character-level chunks.

    Args:
        text: The source text.
        chunk_size: Target characters per chunk.
        overlap: Characters to repeat between adjacent chunks.

    Returns:
        List of chunk strings.
    """
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]
