"""
JsonAgent: an agent whose ``think()`` always returns a parsed Python dict.

The system prompt is automatically suffixed with a strict JSON-only instruction,
and the raw text response is cleaned and parsed before being returned.

Usage::

    agent = JsonAgent(
        "Extractor",
        "You extract structured data from text.",
        schema={"name": "string", "age": "number"},
    )
    result: dict = agent.think("My name is Alex and I am 30 years old.")
    print(result["name"])  # "Alex"
"""

import json
import re
from agentic_ai.agents.base import BaseAgent


_JSON_SUFFIX = (
    "\n\nIMPORTANT: You must respond ONLY with a valid JSON object. "
    "No markdown fences, no preamble, no explanation — just raw JSON."
)


class JsonAgent(BaseAgent):
    """Agent that guarantees its output is a parsed ``dict``.

    Args:
        name: Display name.
        sys_prompt: System instruction (JSON constraint is appended automatically).
        schema: Optional dict describing the expected output shape (for docs/prompting).
        model: Gemini model string.
        api_key: Optional API key.
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        schema: dict | None = None,
        model: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
    ):
        schema_hint = ""
        if schema:
            schema_hint = f"\n\nExpected JSON schema:\n{json.dumps(schema, indent=2)}"
        super().__init__(
            name,
            sys_prompt + schema_hint + _JSON_SUFFIX,
            model=model,
            api_key=api_key,
        )
        self.schema = schema

    def think(self, input_text: str, use_memory: bool = True, stream: bool = False) -> dict:  # type: ignore[override]
        """Process input and return a parsed ``dict``.

        Raises:
            ValueError: If the model's response cannot be parsed as JSON.
        """
        raw = super().think(input_text, use_memory=use_memory, stream=False)
        # Strip markdown fences if the model added them anyway
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"JsonAgent received non-JSON response:\n{raw}"
            ) from exc
