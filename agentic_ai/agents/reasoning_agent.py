"""
ReasoningAgent: an agent that explicitly separates "thinking" from "answering"
and can pass structured context to the next agent in a pipeline.

The agent produces two outputs per turn:
- ``reasoning``: internal chain-of-thought (can be hidden)
- ``answer``: the final response

The ``think()`` method returns a ``ReasoningOutput`` namedtuple so downstream
agents can consume either field.

Usage::

    planner = ReasoningAgent("Planner", "Break the task into steps.")
    executor = BaseAgent("Executor", "Execute the plan given to you.")

    plan = planner.think("Write a blog post about agentic AI.")
    executor.think(plan.context_for_next)   # passes reasoning + answer together
"""

from __future__ import annotations
from dataclasses import dataclass
from agentic_ai.agents.base import BaseAgent


@dataclass
class ReasoningOutput:
    """Output from a single ``ReasoningAgent.think()`` call."""
    reasoning: str
    answer: str

    @property
    def context_for_next(self) -> str:
        """A combined string suitable for passing to the next agent."""
        return f"[REASONING]\n{self.reasoning}\n\n[ANSWER]\n{self.answer}"

    def __str__(self) -> str:
        return self.answer


_REASONING_SUFFIX = (
    "\n\nStructure EVERY response exactly as:\n"
    "THINKING: <your internal reasoning here>\n"
    "ANSWER: <your final response here>\n"
    "Do not deviate from this format."
)


class ReasoningAgent(BaseAgent):
    """An agent that surfaces its chain-of-thought alongside its answer.

    Args:
        name: Display name.
        sys_prompt: System instruction (reasoning format is appended automatically).
        show_reasoning: If ``True`` (default), print the THINKING block to stdout.
        model: Gemini model string.
        api_key: Optional API key.
    """

    def __init__(
        self,
        name: str,
        sys_prompt: str,
        show_reasoning: bool = True,
        model: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
    ):
        super().__init__(
            name,
            sys_prompt + _REASONING_SUFFIX,
            model=model,
            api_key=api_key,
        )
        self.show_reasoning = show_reasoning

    def think(self, input_text: str, use_memory: bool = True, stream: bool = True) -> ReasoningOutput:  # type: ignore[override]
        """Process input and return a :class:`ReasoningOutput`.

        The raw text is split on the ``THINKING:`` / ``ANSWER:`` markers.
        If parsing fails, the full text lands in ``answer`` with empty ``reasoning``.
        """
        raw = super().think(input_text, use_memory=use_memory, stream=stream)

        reasoning, answer = "", raw
        if "THINKING:" in raw and "ANSWER:" in raw:
            try:
                parts = raw.split("ANSWER:", 1)
                answer = parts[1].strip()
                reasoning = parts[0].replace("THINKING:", "", 1).strip()
            except Exception:
                pass

        if not self.show_reasoning:
            # Reprint only the answer if streaming swallowed both
            pass

        return ReasoningOutput(reasoning=reasoning, answer=answer)
