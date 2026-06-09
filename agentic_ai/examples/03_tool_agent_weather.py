"""Example 3 — Agent that calls a real weather tool."""
from agentic_ai.agents.tool_agent import ToolAgent
from agentic_ai.tools.builtins import get_weather, add_numbers

agent = ToolAgent("WeatherBot", "You are a helpful weather assistant.")
agent.register_tool(get_weather)
agent.register_tool(add_numbers)

agent.think("What's the weather in Chennai and London?")
agent.think("What is 42 plus 58?")
