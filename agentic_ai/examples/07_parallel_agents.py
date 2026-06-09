"""Example 7 — Fan-out: same task sent to multiple agents in parallel."""
from agentic_ai.agents.base import BaseAgent
from agentic_ai.patterns.parallel import run_parallel

optimist  = BaseAgent("Optimist",  "You see only the positive side of everything.")
pessimist = BaseAgent("Pessimist", "You see only the risks and downsides.")
realist   = BaseAgent("Realist",   "You give a balanced, evidence-based view.")

results = run_parallel(
    [optimist, pessimist, realist],
    task="Should a small startup adopt AI agents in their product today?",
)

for name, response in results.items():
    print(f"\n=== {name} ===\n{response}")
