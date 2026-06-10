"""
MLEAgent: specialist agent for machine learning engineering.
"""

from __future__ import annotations

from agentic_ai.agents.base import BaseAgent


_MLE_PROMPT = """
You are an elite Machine Learning Engineer.

Your role is to design production-grade ML systems.

For every response:

1. Define prediction target.
2. Identify labels and features.
3. Check leakage risks.
4. Recommend baseline models first.
5. Define evaluation strategy.
6. Discuss deployment architecture.
7. Discuss monitoring and drift.
8. End with implementation steps.

Preferred structure:

PROBLEM DEFINITION
DATA REQUIREMENTS
MODELING STRATEGY
EVALUATION PLAN
DEPLOYMENT CONSIDERATIONS
MONITORING
IMPLEMENTATION PLAN

Prioritize practical solutions over academic complexity.
"""


class MLEAgent(BaseAgent):
    """Machine-learning-engineering specialist agent."""

    def __init__(
        self,
        name: str = "MLE",
        project_context: str | None = None,
        model: str = "gemini-2.5-flash-lite",
        api_key: str | None = None,
        memory_window: int = 3,
        max_turns: int | None = None,
        thinking_budget: int = 0,
    ):
        prompt = _MLE_PROMPT

        if project_context:
            prompt += f"\n\nPROJECT CONTEXT:\n{project_context}"

        super().__init__(
            name=name,
            sys_prompt=prompt,
            model=model,
            api_key=api_key,
            memory_window=memory_window,
            max_turns=max_turns,
            thinking_budget=thinking_budget,
        )

