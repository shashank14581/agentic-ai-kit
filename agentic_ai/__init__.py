"""
agentic_ai — A progressive framework for building agentic AI systems with Google Gemini.
"""

from agentic_ai.agents.base import BaseAgent
from agentic_ai.agents.tool_agent import ToolAgent
from agentic_ai.agents.json_agent import JsonAgent
from agentic_ai.agents.reasoning_agent import ReasoningAgent
from agentic_ai.memory.short_term import ShortTermMemory
from agentic_ai.memory.long_term import LongTermMemory
from agentic_ai.memory.shared import SharedMemory

__version__ = "0.1.0"
__all__ = [
    "BaseAgent",
    "ToolAgent",
    "JsonAgent",
    "ReasoningAgent",
    "ShortTermMemory",
    "LongTermMemory",
    "SharedMemory",
]
