"""
BaseAgent: the foundation for all agents in agentic_ai.

Every agent has:
- A name and system prompt that define its persona
- Short-term memory (last N turns)
- A simple fact store for extracting persistent knowledge
- A `think(input_text)` method that streams a Gemini response
"""

import os
from google import genai
from google.genai import types


class BaseAgent:
    """Core agent backed by a Gemini model.

    Args:
        name (str): Display name of the agent.
        sys_prompt (str): System instruction that defines the agent's persona.
        model (str): Gemini model string. Defaults to ``gemini-2.5-flash-lite``.
        api_key (str | None): Gemini API key. Falls back to the
            ``GEMINI_API_KEY`` environment variable.
        memory_window (int): Number of recent turns to include in context.
        thinking_budget (int): Token budget for Gemini's thinking mode.
            Set to ``0`` to disable.
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        model: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
        memory_window: int = 3,
        thinking_budget: int = 0,
    ):
        self.name = name
        self.sys_prompt = sys_prompt
        self.model = model
        self.memory_window = memory_window
        self.thinking_budget = thinking_budget

        self.memory: list[tuple[str, str]] = []   # (input, output) pairs
        self.facts_store: list[str] = []

        _key = api_key or os.environ.get("GEMINI_API_KEY")
        if not _key:
            raise ValueError(
                "No Gemini API key found. Pass api_key= or set GEMINI_API_KEY."
            )
        self.client = genai.Client(api_key=_key)

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def extract_facts(self, text: str) -> None:
        """Store simple facts from user input (override to customise)."""
        if any(kw in text for kw in ("I am", "I like", "I have", "I want")):
            if text not in self.facts_store:
                self.facts_store.append(text)

    def build_context(self, input_text: str) -> str:
        """Assemble facts + recent history + new message into a single prompt."""
        history = ""
        for inp, out in self.memory[-self.memory_window:]:
            history += f"user: {inp}\nassistant: {out}\n"

        facts_text = "\n".join(self.facts_store) if self.facts_store else "(none)"

        return (
            f"FACTS:\n{facts_text}\n\n"
            f"HISTORY:\n{history}\n"
            f"NEW MESSAGE:\n{input_text}"
        )

    def clear_memory(self) -> None:
        """Wipe short-term memory and facts store."""
        self.memory.clear()
        self.facts_store.clear()

    # ------------------------------------------------------------------
    # Core reasoning
    # ------------------------------------------------------------------

    def think(self, input_text: str, use_memory: bool = True, stream: bool = True) -> str:
        """Send ``input_text`` to Gemini and return the full response string.

        Args:
            input_text: The message to process.
            use_memory: Whether to inject memory/facts into the prompt.
            stream: Stream tokens to stdout as they arrive.

        Returns:
            The complete response text.
        """
        self.extract_facts(input_text)
        prompt = self.build_context(input_text) if use_memory else input_text

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
            system_instruction=[types.Part.from_text(text=self.sys_prompt)],
        )

        response_text = ""
        if stream:
            for chunk in self.client.models.generate_content_stream(
                model=self.model, contents=contents, config=config
            ):
                if chunk.text:
                    print(chunk.text, end="", flush=True)
                    response_text += chunk.text
            print()
        else:
            resp = self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
            response_text = resp.text or ""

        self.memory.append((input_text, response_text))
        return response_text

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} model={self.model!r}>"
