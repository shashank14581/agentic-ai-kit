"""RAGAgent: retrieves relevant chunks before answering."""

from __future__ import annotations
from agentic_ai.agents.base import BaseAgent
from agentic_ai.rag.embedder import embed_texts
from agentic_ai.rag.vector_store import VectorStore
from agentic_ai.rag.chunker import chunk_text


class RAGAgent(BaseAgent):
    """An agent that retrieves context from a vector store before generating a response.

    Usage::

        agent = RAGAgent("Docs Bot", "You answer questions using the provided context.")
        agent.ingest("Long document text here...")
        agent.think("What does this document say about X?")
    """

    def __init__(self, name: str, sys_prompt: str, top_k: int = 3, **kwargs):
        super().__init__(name, sys_prompt, **kwargs)
        self.top_k = top_k
        self._store = VectorStore()

    def ingest(self, text: str, chunk_size: int = 500, overlap: int = 50) -> int:
        """Chunk and embed a document, adding it to the vector store.

        Returns:
            Number of chunks ingested.
        """
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        vectors = embed_texts(chunks, api_key=None)
        self._store.add(chunks, vectors)
        return len(chunks)

    def think(self, input_text: str, use_memory: bool = True, stream: bool = True) -> str:
        """Retrieve relevant chunks, inject them as context, then generate a response."""
        query_vec = embed_texts([input_text])[0]
        hits = self._store.search(query_vec, top_k=self.top_k)
        context = "\n\n---\n\n".join(h["text"] for h in hits)
        augmented = f"CONTEXT:\n{context}\n\nQUESTION:\n{input_text}"
        return super().think(augmented, use_memory=use_memory, stream=stream)
