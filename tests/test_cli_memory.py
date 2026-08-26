from types import SimpleNamespace

import agentic_ai.cli.main as cli


def _args(command):
    return SimpleNamespace(
        command=command,
        name="AAK",
        system="Be concise.",
        model="gemini-2.5-flash-lite",
        transport="generate_content",
    )


def test_run_disables_memory_extraction(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        cli,
        "BaseAgent",
        FakeAgent,
    )

    cli._create_agent(_args("run"))

    assert captured["extract_memory"] is False


def test_chat_enables_memory_extraction(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        cli,
        "BaseAgent",
        FakeAgent,
    )

    cli._create_agent(_args("chat"))

    assert captured["extract_memory"] is True
