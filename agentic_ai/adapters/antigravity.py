from __future__ import annotations

import asyncio
from typing import Any, Callable

from agentic_ai.adapters.base import RuntimeAdapter


class AntigravityAdapter:
    """Run an AAK agent through the Antigravity runtime."""

    def __init__(
        self,
        agent: Any,
        *,
        config_factory: Callable[..., Any] | None = None,
        agent_factory: Callable[..., Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
    ):
        self.agent = agent
        self.config_kwargs = dict(config_kwargs or {})

        if config_factory is None or agent_factory is None:
            try:
                from google.antigravity import Agent, LocalAgentConfig

            except ImportError as exc:
                raise ImportError(
                    "Antigravity support requires `google-antigravity`. "
                    "Install it with `pip install agentic-ai-kit[antigravity]`."
                ) from exc

            config_factory = config_factory or LocalAgentConfig
            agent_factory = agent_factory or Agent

        self._config_factory = config_factory
        self._agent_factory = agent_factory

    def _build_config(self) -> Any:
        kwargs = dict(self.config_kwargs)

        kwargs.setdefault(
            "system_instructions",
            getattr(
                self.agent,
                "sys_prompt",
                "You are a helpful AI assistant.",
            ),
        )

        model = getattr(self.agent, "model", None)
        if model is not None:
            kwargs.setdefault("model", model)

        return self._config_factory(**kwargs)

    async def arun(self, prompt: str) -> str:
        if not prompt.strip():
            raise ValueError("prompt cannot be empty.")

        config = self._build_config()

        async with self._agent_factory(config) as runtime:
            response = await runtime.chat(prompt)
            text = await response.text()

        return str(text).strip()

    def run(self, prompt: str) -> str:
        try:
            asyncio.get_running_loop()

        except RuntimeError:
            return asyncio.run(
                self.arun(prompt)
            )

        raise RuntimeError(
            "AntigravityAdapter.run() cannot be used inside an active "
            "asyncio loop. Use `await adapter.arun(...)` instead."
        )


__all__ = [
    "AntigravityAdapter",
    "RuntimeAdapter",
]
