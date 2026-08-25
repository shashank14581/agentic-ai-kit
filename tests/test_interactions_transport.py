from types import SimpleNamespace

import pytest

from agentic_ai.agents.base import BaseAgent


class FakeInteractions:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class FakeClient:
    def __init__(self, result):
        self.interactions = FakeInteractions(result)


def completed_interaction(text="Hello", interaction_id="int_1"):
    return SimpleNamespace(
        status="completed",
        output_text=text,
        id=interaction_id,
        steps=[SimpleNamespace(id="step_1")],
        usage={"total_tokens": 3},
        errors=None,
    )


def test_interactions_nonstream_request_and_metadata():
    interaction = completed_interaction()
    client = FakeClient(interaction)
    agent = BaseAgent(
        name="test",
        sys_prompt="Be concise.",
        model="gemini-test",
        transport="interactions",
        thinking_level="low",
        client=client,
        extract_memory=False,
    )

    result = agent.think("Hi", use_memory=False, stream=False)

    assert result == "Hello"
    assert client.interactions.calls == [
        {
            "model": "gemini-test",
            "input": "Hi",
            "system_instruction": "Be concise.",
            "store": False,
            "stream": False,
            "generation_config": {"thinking_level": "low"},
        }
    ]
    assert agent.last_interaction is interaction
    assert agent.last_interaction_id == "int_1"
    assert agent.last_steps == interaction.steps
    assert agent.last_usage == interaction.usage
    assert agent.memory[-1] == ("Hi", "Hello")


def test_interactions_stream_accumulates_text(capsys):
    interaction = completed_interaction(interaction_id="")
    interaction.steps = []
    events = [
        SimpleNamespace(
            event_type="step.delta",
            delta=SimpleNamespace(type="text", text="Hel"),
        ),
        SimpleNamespace(
            event_type="step.delta",
            delta=SimpleNamespace(type="text", text="lo"),
        ),
        SimpleNamespace(
            event_type="interaction.completed",
            interaction=interaction,
        ),
    ]
    client = FakeClient(events)
    agent = BaseAgent(
        name="test",
        sys_prompt="Be concise.",
        transport="interactions",
        client=client,
        extract_memory=False,
    )

    result = agent.think("Hi", use_memory=False, stream=True)

    assert result == "Hello"
    assert capsys.readouterr().out == "Hello\n"
    assert client.interactions.calls[0]["stream"] is True
    assert client.interactions.calls[0]["store"] is False
    assert "generation_config" not in client.interactions.calls[0]
    assert agent.last_interaction is interaction
    assert agent.last_interaction_id is None
    assert agent.last_steps == []
    assert agent.last_usage == interaction.usage


def test_interactions_failure_raises_runtime_error():
    failed = SimpleNamespace(
        status="failed",
        output_text="",
        id="int_bad",
        steps=[],
        usage=None,
        errors=[SimpleNamespace(message="boom")],
    )
    agent = BaseAgent(
        name="test",
        sys_prompt="Be concise.",
        transport="interactions",
        client=FakeClient(failed),
        extract_memory=False,
    )

    with pytest.raises(RuntimeError, match="failed"):
        agent.think("Hi", use_memory=False, stream=False)


def test_interactions_stream_error_raises_runtime_error():
    events = [
        SimpleNamespace(
            event_type="error",
            error=SimpleNamespace(
                code="internal",
                message="boom",
            ),
        )
    ]
    agent = BaseAgent(
        name="test",
        sys_prompt="Be concise.",
        transport="interactions",
        client=FakeClient(events),
        extract_memory=False,
    )

    with pytest.raises(RuntimeError, match="boom"):
        agent.think("Hi", use_memory=False, stream=True)


def test_invalid_transport_is_rejected():
    with pytest.raises(ValueError, match="transport"):
        BaseAgent(
            name="test",
            sys_prompt="Be concise.",
            transport="bogus",
            client=FakeClient(None),
            extract_memory=False,
        )