"""Agent implementations exposed lazily to keep imports lightweight."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_IMPORTS = {
    "BaseAgent": ("agentic_ai.agents.base", "BaseAgent"),
    "ToolAgent": ("agentic_ai.agents.tool_agent", "ToolAgent"),
    "JsonAgent": ("agentic_ai.agents.json_agent", "JsonAgent"),
    "ReasoningAgent": ("agentic_ai.agents.reasoning_agent", "ReasoningAgent"),
    "AnalystAgent": ("agentic_ai.agents.analyst_agent", "AnalystAgent"),
    "MLEAgent": ("agentic_ai.agents.mle_agent", "MLEAgent"),
    "AutoModelAgent": ("agentic_ai.agents.auto_model_agent", "AutoModelAgent"),
}

__all__ = list(_LAZY_IMPORTS)


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
