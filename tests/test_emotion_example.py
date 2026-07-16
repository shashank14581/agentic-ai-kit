from __future__ import annotations

import importlib.util
from pathlib import Path


def test_emotion_memory_example_runs_without_api_key():
    example_path = (
        Path(__file__).resolve().parents[1]
        / "agentic_ai"
        / "examples"
        / "12_emotion_memory.py"
    )
    spec = importlib.util.spec_from_file_location("emotion_memory_example", example_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    result = module.run_demo()

    assert list(result["labels_before"].values()) == [
        "failure",
        "wound",
        "trauma",
    ]
    assert list(result["labels_after"].values()) == ["failure", "wound"]
    # All three events share a trajectory and identical semantic similarity.
    # The retriever therefore deduplicates them to the earliest stable tie-break.
    assert result["retrieved_memory"] == "planning-failure-1"
    assert result["highest_attention_node"] is not None
    assert result["policy_memory_q"] < 0.0
    assert result["td_target"] == -1.0
    assert result["active_event_ids"] == (
        "planning-failure-1",
        "planning-failure-3",
    )
