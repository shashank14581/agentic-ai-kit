"""Example 6 — Supervisor delegates to specialist workers."""
from agentic_ai.agents.base import BaseAgent
from agentic_ai.patterns.orchestrator import run_supervisor

supervisor = BaseAgent("Manager",   "You are a project manager who delegates tasks.")
researcher = BaseAgent("Researcher","You research topics and summarise findings.")
writer     = BaseAgent("Writer",    "You write engaging content based on research.")

results = run_supervisor(
    supervisor,
    workers={"Researcher": researcher, "Writer": writer},
    task="Create a short blog post about the future of AI agents.",
)

for name, output in results.items():
    print(f"\n=== {name} ===\n{output}")
