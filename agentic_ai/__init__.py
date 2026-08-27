"""agentic_ai - a progressive framework for building agentic AI systems."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.5.1"

_LAZY_IMPORTS = {
    "AutoModelAgent": ("agentic_ai.agents.auto_model_agent", "AutoModelAgent"),
    "BaseAgent": ("agentic_ai.agents.base", "BaseAgent"),
    "JsonAgent": ("agentic_ai.agents.json_agent", "JsonAgent"),
    "ReasoningAgent": ("agentic_ai.agents.reasoning_agent", "ReasoningAgent"),
    "ToolAgent": ("agentic_ai.agents.tool_agent", "ToolAgent"),
    "EmotionConfig": ("agentic_ai.memory.emotion", "EmotionConfig"),
    "EmotionMemorySystem": ("agentic_ai.memory.emotion", "EmotionMemorySystem"),
    "LinearQLearner": ("agentic_ai.memory.emotion", "LinearQLearner"),
    "RLStateEncoder": ("agentic_ai.memory.emotion", "RLStateEncoder"),
    "TreeAttentionEngine": ("agentic_ai.memory.emotion", "TreeAttentionEngine"),
    "LongTermMemory": ("agentic_ai.memory.long_term", "LongTermMemory"),
    "SharedMemory": ("agentic_ai.memory.shared", "SharedMemory"),
    "ShortTermMemory": ("agentic_ai.memory.short_term", "ShortTermMemory"),
}

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


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from exc

    module = import_module(module_name)
    value = getattr(module, attribute_name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
