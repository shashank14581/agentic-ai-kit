"""Example 4 — Agent that always returns structured JSON."""
from agentic_ai.agents.json_agent import JsonAgent

agent = JsonAgent(
    "Extractor",
    "Extract structured information from the user's message.",
    schema={"name": "string", "age": "number", "city": "string"},
)

result = agent.think("Hi, I'm Priya, 28 years old, living in Bangalore.")
print(result)  # {'name': 'Priya', 'age': 28, 'city': 'Bangalore'}
