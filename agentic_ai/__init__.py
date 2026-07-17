"""agentic_ai - A progressive framework for building agentic AI systems."""

from agentic_ai.agents.auto_model_agent import AutoModelAgent
from agentic_ai.agents.base import BaseAgent
from agentic_ai.agents.json_agent import JsonAgent
from agentic_ai.agents.reasoning_agent import ReasoningAgent
from agentic_ai.agents.tool_agent import ToolAgent
from agentic_ai.memory.emotion import (
    EmotionConfig,
    EmotionMemorySystem,
    LinearQLearner,
    RLStateEncoder,
    TreeAttentionEngine,
)
from agentic_ai.memory.long_term import LongTermMemory
from agentic_ai.memory.shared import SharedMemory
from agentic_ai.memory.short_term import ShortTermMemory

__version__ = "0.2.1"

__all__ = [
    "AutoModelAgent",
    "BaseAgent",
    "EmotionConfig",
    "EmotionMemorySystem",
    "JsonAgent",
    "LinearQLearner",
    "LongTermMemory",
    "RLStateEncoder",
    "ReasoningAgent",
    "SharedMemory",
    "ShortTermMemory",
    "ToolAgent",
    "TreeAttentionEngine",
]
