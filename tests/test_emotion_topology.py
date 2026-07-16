from __future__ import annotations

import pytest

from agentic_ai.memory.emotion import (
    AppraisalEngine,
    EpisodicMemoryTimeline,
    EvidenceDirection,
    EvidenceItem,
    Event,
    IdentityState,
    OutcomeEvidence,
)


def failure_evidence(event_id: str) -> OutcomeEvidence:
    return OutcomeEvidence(
        items=(
            EvidenceItem(
                evidence_id=f"{event_id}:evidence",
                direction=EvidenceDirection.FAILURE,
                magnitude=1.0,
                reliability=1.0,
                provenance="environment",
            ),
        )
    )


def append_failure(timeline, engine, identity, timestep):
    event = Event(
        event_id=f"failure-{timestep}",
        timestep=timestep,
        context_id="planning",
        action_id="invalid-plan",
        embedding=(1.0, 0.0, 0.0, 0.0),
        expected_valence=0.0,
    )
    evidence = failure_evidence(event.event_id)
    transition = engine.evaluate(
        identity,
        event,
        evidence,
        timeline.history_before(timestep),
    )
    return timeline.append(event, evidence, transition)


def test_timeline_links_timesteps_and_roots_event_trees():
    engine = AppraisalEngine()
    identity = IdentityState(supplied_vector=(-0.29, 0.9570266, 0.0, 0.0))
    timeline = EpisodicMemoryTimeline()

    for timestep in (1, 2, 3):
        append_failure(timeline, engine, identity, timestep)

    assert timeline.head_timestep == 1
    assert timeline.tail_timestep == 3
    assert timeline.get(1).next_timestep == 2
    assert timeline.get(2).previous_timestep == 1
    assert timeline.get(2).next_timestep == 3
    assert timeline.get(3).previous_timestep == 2
    assert [node.timestep for node in timeline.iter_nodes()] == [1, 2, 3]

    tree = timeline.get(2).tree
    node_types = {node.node_type for node in tree.nodes}
    assert {
        "event-root",
        "context",
        "action",
        "expectation",
        "evidence",
        "failure-evidence",
        "snapshot",
    } <= node_types
    assert tree.get("failure-2:snapshot").as_mapping()["label"] == "wound"


def test_invalidated_timestep_is_removed_from_active_history():
    engine = AppraisalEngine()
    identity = IdentityState(supplied_vector=(-0.29, 0.9570266, 0.0, 0.0))
    timeline = EpisodicMemoryTimeline()
    for timestep in (1, 2, 3):
        append_failure(timeline, engine, identity, timestep)

    assert len(timeline.history_before(4)) == 3
    timeline.invalidate("failure-2", "deleted")
    assert len(timeline) == 2
    assert len(timeline.history_before(4)) == 2
    assert [node.timestep for node in timeline.iter_nodes()] == [1, 3]
    assert [node.timestep for node in timeline.iter_nodes(include_inactive=True)] == [1, 2, 3]


def test_timeline_requires_strictly_increasing_timesteps():
    engine = AppraisalEngine()
    identity = IdentityState(supplied_vector=(1.0, 0.0, 0.0, 0.0))
    timeline = EpisodicMemoryTimeline()
    append_failure(timeline, engine, identity, 2)

    with pytest.raises(ValueError, match="strictly increasing"):
        append_failure(timeline, engine, identity, 1)
