"""Example 2 — Agent with personality and short-term memory."""
from agentic_ai.agents.base import BaseAgent

agent = BaseAgent(
    "Alfred",
    "You are Alfred, a witty British butler who remembers everything the user tells you.",
    memory_window=5,
)

agent.think("My name is Shashank and I love cricket.")
agent.think("I also own a golden retriever named Bruno.")
agent.think("What do you know about me so far?")
