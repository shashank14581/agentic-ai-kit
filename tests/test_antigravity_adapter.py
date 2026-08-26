import asyncio

import pytest

from agentic_ai.adapters import AntigravityAdapter


class FakeAAKAgent:
    sys_prompt = "You are Alfred."


class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeResponse:
    async def text(self):
        return "Antigravity response"


class FakeRuntime:
    def __init__(self, config):
        self.config = config

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type,
        exc,
        traceback,
    ):
        return False

    async def chat(self, prompt):
        assert prompt == "Hello"
        return FakeResponse()


def test_antigravity_adapter_maps_system_prompt():
    created = {}

    def config_factory(**kwargs):
        config = FakeConfig(**kwargs)
        created["config"] = config
        return config

    adapter = AntigravityAdapter(
        FakeAAKAgent(),
        config_factory=config_factory,
        agent_factory=FakeRuntime,
    )

    result = adapter.run("Hello")

    assert result == "Antigravity response"
    assert (
        created["config"].kwargs["system_instructions"]
        == "You are Alfred."
    )


def test_antigravity_adapter_config_override():
    created = {}

    def config_factory(**kwargs):
        config = FakeConfig(**kwargs)
        created["config"] = config
        return config

    adapter = AntigravityAdapter(
        FakeAAKAgent(),
        config_factory=config_factory,
        agent_factory=FakeRuntime,
        config_kwargs={
            "system_instructions": "Override",
        },
    )

    result = adapter.run("Hello")

    assert result == "Antigravity response"
    assert (
        created["config"].kwargs["system_instructions"]
        == "Override"
    )


def test_antigravity_adapter_rejects_empty_prompt():
    adapter = AntigravityAdapter(
        FakeAAKAgent(),
        config_factory=FakeConfig,
        agent_factory=FakeRuntime,
    )

    with pytest.raises(
        ValueError,
        match="prompt cannot be empty",
    ):
        asyncio.run(
            adapter.arun("   ")
        )
