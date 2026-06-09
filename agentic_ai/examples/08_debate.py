"""Example 8 — Two agents debate; a judge picks a winner."""
from agentic_ai.agents.base import BaseAgent
from agentic_ai.patterns.debate import run_debate

pro   = BaseAgent("ProAgent",  "You argue strongly IN FAVOUR of the motion.")
con   = BaseAgent("ConAgent",  "You argue strongly AGAINST the motion.")
judge = BaseAgent("Judge",     "You are an impartial judge. Evaluate arguments fairly.")

outcome = run_debate(
    pro, con, judge,
    topic="AI agents will replace most knowledge workers within 10 years.",
    rounds=2,
)

print("\n=== VERDICT ===")
print(outcome["verdict"])
