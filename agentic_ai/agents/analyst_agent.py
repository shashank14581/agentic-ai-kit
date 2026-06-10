"""
AnalystAgent: specialist agent for analytics, experimentation,
marketing measurement, segmentation, forecasting, and business insights.
"""

from __future__ import annotations

from agentic_ai.agents.base import BaseAgent


_ANALYST_PROMPT = """
You are an elite Decision Scientist.

Your job is to help users solve analytics and business problems.

For every response:

1. Clarify the business objective.
2. Identify relevant metrics and dimensions.
3. Highlight assumptions.
4. Separate observations from interpretations.
5. Identify data quality risks and confounders.
6. Recommend next analyses.
7. End with a decision-oriented summary.

Preferred structure:

BUSINESS QUESTION
KEY OBSERVATIONS
ANALYTICAL RISKS
RECOMMENDED ANALYSIS
DECISION SUMMARY

Avoid generic consulting language.
Think like a strong analyst.
"""


class AnalystAgent(BaseAgent):
    """Analytics-focused specialist agent."""

    def __init__(
        self,
        name: str = "Analyst",
        domain_context: str | None = None,
        model: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
        memory_window: int = 3,
        max_turns: int | None = None,
        thinking_budget: int = 0,
    ):
        prompt = _ANALYST_PROMPT

        if domain_context:
            prompt += f"\n\nDOMAIN CONTEXT:\n{domain_context}"

        super().__init__(
            name=name,
            sys_prompt=prompt,
            model=model,
            api_key=api_key,
            memory_window=memory_window,
            max_turns=max_turns,
            thinking_budget=thinking_budget,
        )

