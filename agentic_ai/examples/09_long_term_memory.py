"""Example 9 — Agent backed by persistent SQLite memory."""
from agentic_ai.agents.base import BaseAgent
from agentic_ai.memory.long_term import LongTermMemory

mem   = LongTermMemory("agent_memory.db")
agent = BaseAgent("Archivist", "You are a helpful assistant with a long memory.")

# Override think() to auto-log to long-term memory
_original_think = agent.think

def think_with_log(text, **kwargs):
    response = _original_think(text, **kwargs)
    mem.log(f"User: {text} | Agent: {response}", agent="Archivist")
    return response

agent.think = think_with_log

agent.think("Remember that my project deadline is June 30th.")
agent.think("What important dates should I keep in mind?")

print("\nRecent memory entries:")
for entry in mem.recent(5):
    print(entry)
