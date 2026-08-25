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
        transport: str = "generate_content",
        thinking_level: str | None = None,
        client: object | None = None,
    ):
        self.name = name
        self.sys_prompt = sys_prompt
        self.model = model
        self.memory_window = memory_window
        self.max_turns = max_turns
        self.max_facts = max_facts
        self.extract_memory = extract_memory
        self.thinking_budget = thinking_budget

        if transport not in {"generate_content", "interactions"}:
            raise ValueError(
                "transport must be 'generate_content' or 'interactions'."
            )

        self.transport = transport
        self.thinking_level = thinking_level

        self.memory: list[tuple[str, str]] = []
        self.facts_store: list[dict[str, object]] = []

        self.last_interaction: object | None = None
        self.last_interaction_id: str | None = None
        self.last_steps: list[object] = []
        self.last_usage: object | None = None

        # Transformer-backed raw conversational memory.
        self.memory_top_k = 8
        self._memory_retriever = TransformerMemoryRetriever()

        if client is not None:
            self.client = client
        else:
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

        self.last_interaction = None
        self.last_interaction_id = None
        self.last_steps = []
        self.last_usage = None

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _think_with_generate_content(
        self,
        prompt: str,
        stream: bool,
    ) -> str:
        """Generate a response through the generateContent API."""

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

        return response_text

    def _record_interaction(self, interaction: object) -> None:
        """Expose metadata from the most recent interaction."""

        self.last_interaction = interaction
        self.last_interaction_id = getattr(interaction, "id", None)
        self.last_steps = list(
            getattr(interaction, "steps", None) or []
        )
        self.last_usage = getattr(interaction, "usage", None)

    @staticmethod
    def _interaction_status(interaction: object) -> str:
        """Normalize an interaction status to lowercase text."""

        status = getattr(interaction, "status", None)
        status = getattr(status, "value", status)
        return str(status).lower()

    def _raise_for_interaction_status(
        self,
        interaction: object,
    ) -> None:
        """Raise when an interaction did not complete successfully."""

        self._record_interaction(interaction)
        status = self._interaction_status(interaction)

        if status == "completed":
            return

        errors = getattr(interaction, "errors", None) or []
        if not isinstance(errors, (list, tuple)):
            errors = [errors]

        details = "; ".join(
            str(getattr(error, "message", error))
            for error in errors
        )

        message = (
            f"Interaction {self.last_interaction_id!r} "
            f"ended with status {status!r}."
        )

        if details:
            message += f" {details}"

        raise RuntimeError(message)

    def _think_with_interactions(
        self,
        prompt: str,
        stream: bool,
    ) -> str:
        """Generate a response through the Interactions API."""

        request: dict[str, object] = {
            "model": self.model,
            "input": prompt,
            "system_instruction": self.sys_prompt,
            "store": False,
            "stream": stream,
        }

        if self.thinking_level is not None:
            request["generation_config"] = {
                "thinking_level": self.thinking_level
            }

        result = self.client.interactions.create(**request)

        if not stream:
            self._raise_for_interaction_status(result)
            return getattr(result, "output_text", None) or ""

        response_parts: list[str] = []
        final_interaction: object | None = None

        for event in result:
            raw_event_type = getattr(event, "event_type", "")
            event_type = str(
                getattr(raw_event_type, "value", raw_event_type)
            ).lower()

            if event_type == "step.delta":
                delta = getattr(event, "delta", None)
                raw_delta_type = getattr(delta, "type", "")
                delta_type = str(
                    getattr(raw_delta_type, "value", raw_delta_type)
                ).lower()

                if delta_type == "text":
                    text = getattr(delta, "text", None) or ""
                    if text:
                        print(text, end="", flush=True)
                        response_parts.append(text)

            elif event_type == "error":
                error = getattr(event, "error", None)
                code = getattr(error, "code", None)
                message = getattr(error, "message", None)

                details = message or str(error or "unknown error")
                if code:
                    details = f"{code}: {details}"

                raise RuntimeError(
                    f"Interactions stream error: {details}"
                )

            elif event_type in {
                "interaction.completed",
                "interaction.failed",
                "interaction.cancelled",
            }:
                final_interaction = getattr(
                    event,
                    "interaction",
                    None,
                )

        print()

        if final_interaction is None:
            raise RuntimeError(
                "Interactions stream ended without a final interaction."
            )

        self._raise_for_interaction_status(final_interaction)

        response_text = "".join(response_parts)
        if not response_text:
            response_text = (
                getattr(final_interaction, "output_text", None) or ""
            )

        return response_text

    def think(
        self,
        input_text: str,
        use_memory: bool = True,
        stream: bool = True,
    ) -> str:
        """Send input through the configured Gemini transport."""

        self.extract_facts(input_text)
        prompt = self.build_context(input_text) if use_memory else input_text

        self.last_interaction = None
        self.last_interaction_id = None
        self.last_steps = []
        self.last_usage = None

        if self.transport == "interactions":
            response_text = self._think_with_interactions(
                prompt,
                stream,
            )
        else:
            response_text = self._think_with_generate_content(
                prompt,
                stream,
            )

        self.memory.append((input_text, response_text))
        self.trim_memory()

        return response_text

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.name!r} "
            f"model={self.model!r} "
            f"transport={self.transport!r} "
            f"memory_window={self.memory_window!r} "
            f"max_turns={self.max_turns!r} "
            f"max_facts={self.max_facts!r}>"
        )