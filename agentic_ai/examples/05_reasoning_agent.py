"""Example 5 — Agent that shows its chain-of-thought."""
from agentic_ai.agents.reasoning_agent import ReasoningAgent
from agentic_ai.agents.base import BaseAgent

planner  = ReasoningAgent("Planner",  "You break down tasks into clear steps.")
executor = BaseAgent("Executor", "You execute the exact plan given to you.")

plan = planner.think("Write a 3-tweet thread about agentic AI.")
executor.think(plan.context_for_next)
