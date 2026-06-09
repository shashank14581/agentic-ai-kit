"""
Tool registry: a lightweight decorator and container for agent-callable functions.

Usage::

    from agentic_ai.tools.registry import tool

    @tool(description="Multiply two numbers", params={"a": "NUMBER", "b": "NUMBER"})
    def multiply(a: float, b: float) -> float:
        return a * b
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Any


@dataclass
class ToolSpec:
    """Metadata for a registered tool."""
    fn: Callable
    name: str
    description: str
    params: dict[str, str]          # param_name -> Gemini type string
    required: list[str]

    def to_function_declaration_dict(self) -> dict:
        """Return a dict compatible with ``types.FunctionDeclaration``."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "OBJECT",
                "properties": {k: {"type": v} for k, v in self.params.items()},
                "required": self.required,
            },
        }

    def call(self, **kwargs: Any) -> Any:
        return self.fn(**kwargs)


class ToolRegistry:
    """Container that holds multiple :class:`ToolSpec` instances."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        fn: Callable,
        description: str | None = None,
        params: dict[str, str] | None = None,
        required: list[str] | None = None,
    ) -> ToolSpec:
        import inspect

        name = fn.__name__
        desc = description or (fn.__doc__ or "").strip().split("\n")[0]

        if params is None:
            hints = inspect.get_annotations(fn)
            _type_map = {str: "STRING", int: "NUMBER", float: "NUMBER", bool: "BOOLEAN"}
            params = {
                k: _type_map.get(v, "STRING")
                for k, v in hints.items()
                if k != "return"
            }

        req = required if required is not None else list((params or {}).keys())
        spec = ToolSpec(fn=fn, name=name, description=desc, params=params or {}, required=req)
        self._tools[name] = spec
        return spec

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def call(self, name: str, **kwargs: Any) -> Any:
        spec = self._tools.get(name)
        if spec is None:
            raise KeyError(f"No tool named '{name}'")
        return spec.call(**kwargs)

    def all_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={list(self._tools.keys())}>"


# Module-level default registry
_default_registry = ToolRegistry()


def tool(
    description: str | None = None,
    params: dict[str, str] | None = None,
    required: list[str] | None = None,
):
    """Decorator that registers a function with the default :class:`ToolRegistry`.

    Can be used with or without arguments::

        @tool
        def my_fn(x: str) -> str: ...

        @tool(description="Explicit description", params={"x": "STRING"})
        def my_fn(x: str) -> str: ...
    """
    def decorator(fn: Callable) -> Callable:
        _default_registry.register(fn, description=description, params=params, required=required)
        fn._tool_spec = _default_registry.get(fn.__name__)
        return fn

    # Called as @tool (no parentheses)
    if callable(description):
        fn, description = description, None  # type: ignore[assignment]
        return decorator(fn)  # type: ignore[arg-type]

    return decorator
