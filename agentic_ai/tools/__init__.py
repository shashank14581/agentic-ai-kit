"""
agentic_ai.tools — built-in tools and a registration decorator.
"""

from agentic_ai.tools.registry import tool, ToolRegistry
from agentic_ai.tools.builtins import get_weather, add_numbers

__all__ = ["tool", "ToolRegistry", "get_weather", "add_numbers"]
