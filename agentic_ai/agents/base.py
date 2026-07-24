"""
BaseAgent: the foundation for all agents in agentic_ai.

Every agent has:
- A name and system prompt that define its persona
- Short-term memory for recent context
- Optional max-turn retention at the interface level
- LLM-based fact extraction for lightweight user memory
- A `think(input_text)` method that streams a Gemini response
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types

from agentic_ai.memory.transformer_retriever import (
    TransformerMemoryRetriever,
)


class BaseAgent:
    """Core agent backed by a Gemini model."""

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
        memory_window: int = 3,
        max_turns: int | None = None,
        max_facts: int = 50,
        extract_memory: bool = True,
        thinking_budget: int = 0,
    ):
        self.name = name
        self.sys_prompt = sys_prompt
        self.model = model
        self.memory_window = memory_window
        self.max_turns = max_turns
        self.max_facts = max_facts
        self.extract_memory = extract_memory
        self.thinking_budget = thinking_budget

        self.memory: list[tuple[str, str]] = []
        self.facts_store: list[dict[str, object]] = []

        # Transformer-backed raw conversational memory.
        self.memory_top_k = 8
        self._memory_retriever = TransformerMemoryRetriever()

        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "No Gemini API key found. Pass api_key= or set GEMINI_API_KEY."
            )

        self.client = genai.Client(api_key=key)

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def extract_facts(self, text: str) -> None:
        """Store the raw message as locally retrievable memory.

        The method name is retained for backward compatibility.
        No generative model is used for extraction.
        """

        if not self.extract_memory:
            return

        normalized_text = self._memory_retriever.normalize_text(text)

        if not normalized_text:
            return

        if self._fact_exists(normalized_text):
            return

        try:
            self._memory_retriever.add(normalized_text)

            self.facts_store.append(
                {
                    "fact": normalized_text,
                    "confidence": 1.0,
                    "source": text,
                }
            )

            self.trim_facts()

        except Exception:
            # Memory processing must never stop the main agent response.
            return

    def _fact_exists(self, new_fact: str) -> bool:
        """Check whether a fact already exists in memory."""

        normalized_new = new_fact.lower().strip().rstrip(".")

        for item in self.facts_store:
            existing = str(item.get("fact", "")).lower().strip().rstrip(".")
            if existing == normalized_new:
                return True

        return False

    def build_context(self, input_text: str) -> str:
        """Assemble retrieved memories, recent history, and the new message."""

        recent_turns = self.memory[-self.memory_window :]

        history = "\n".join(
            f"user: {user_input}\nassistant: {agent_output}"
            for user_input, agent_output in recent_turns
        )

        try:
            selected_facts = self._memory_retriever.retrieve(
                query=input_text,
                records=self.facts_store,
                top_k=self.memory_top_k,
                exclude_latest_query=True,
            )

        except Exception:
            # Safe fallback if transformer retrieval is unavailable.
            fallback_records = list(self.facts_store)

            if fallback_records:
                latest_fact = str(
                    fallback_records[-1].get("fact", "")
                ).strip()

                if latest_fact.lower() == input_text.strip().lower():
                    fallback_records = fallback_records[:-1]

            selected_facts = fallback_records[-self.memory_top_k :]

        if selected_facts:
            facts_text = "\n".join(
                f"- {item['fact']}"
                for item in selected_facts
            )
        else:
            facts_text = "(none)"

        return (
            f"FACTS:\n{facts_text}\n\n"
            f"HISTORY:\n{history}\n\n"
            f"NEW MESSAGE:\n{input_text}"
        )

    def trim_memory(self) -> None:
        """Limit retained conversation turns at the interface level."""

        if self.max_turns is not None:
            if self.max_turns < 0:
                raise ValueError("max_turns must be None or a non-negative integer.")

            self.memory = self.memory[-self.max_turns :]

    def trim_facts(self) -> None:
        """Limit the number of stored memory records."""

        if self.max_facts < 0:
            raise ValueError(
                "max_facts must be a non-negative integer."
            )

        if self.max_facts == 0:
            self.facts_store.clear()
        else:
            self.facts_store = self.facts_store[-self.max_facts :]

        self._memory_retriever.prune(
            [
                str(item.get("fact", ""))
                for item in self.facts_store
            ]
        )

    def clear_memory(self) -> None:
        """Wipe short-term memory and transformer memory."""

        self.memory.clear()
        self.facts_store.clear()
        self._memory_retriever.clear()

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def think(
        self,
        input_text: str,
        use_memory: bool = True,
        stream: bool = True,
    ) -> str:
        """Send input to Gemini and return the full response string."""

        self.extract_facts(input_text)
        prompt = self.build_context(input_text) if use_memory else input_text

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            ),
            system_instruction=[
                types.Part.from_text(text=self.sys_prompt)
            ],
        )

        response_text = ""

        if stream:
            for chunk in self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response_text += chunk.text

            print()

        else:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            response_text = response.text or ""

        self.memory.append((input_text, response_text))
        self.trim_memory()

        return response_text

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.name!r} "
            f"model={self.model!r} "
            f"memory_window={self.memory_window!r} "
            f"max_turns={self.max_turns!r} "
            f"max_facts={self.max_facts!r}>"
        )