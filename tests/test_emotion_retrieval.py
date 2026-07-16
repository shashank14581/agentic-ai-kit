from __future__ import annotations

import math

from agentic_ai.memory.emotion import (
    AppraisalEngine,
    EvidenceDirection,
    EvidenceItem,
    Event,
    HistoricalOutcome,
    IdentityState,
    MemoryCandidate,
    OutcomeEvidence,
    OutcomeLabel,
    OutcomeStatus,
    RelevanceFirstRetriever,
)


def transition(event_id: str, success: bool = True):
    direction = EvidenceDirection.SUCCESS if success else EvidenceDirection.FAILURE
    evidence = OutcomeEvidence(
        items=(EvidenceItem(event_id, direction, 1.0, 1.0, "env"),)
    )
    event = Event(
        event_id,
        1,
        "context",
        "action",
        (1.0, 0.0, 0.0, 0.0),
        0.0,
    )
    result = AppraisalEngine().evaluate(
        IdentityState(supplied_vector=(1.0, 0.0, 0.0, 0.0)),
        event,
        evidence,
    )
    return result


def trauma_transition():
    evidence = OutcomeEvidence(
        items=(EvidenceItem("trauma", EvidenceDirection.FAILURE, 1.0, 1.0, "env"),)
    )
    event = Event(
        "trauma",
        3,
        "planning",
        "invalid-plan",
        (1.0, 0.0, 0.0, 0.0),
        0.0,
    )
    history = (
        HistoricalOutcome("f1", 1, event.recurrence_vector, OutcomeStatus.FAILURE, 1.0, 1.0),
        HistoricalOutcome("f2", 2, event.recurrence_vector, OutcomeStatus.FAILURE, 1.0, 1.0),
    )
    result = AppraisalEngine().evaluate(
        IdentityState(supplied_vector=(-0.29, 0.9570266, 0.0, 0.0)),
        event,
        evidence,
        history,
    )
    assert result.state.operational_label is OutcomeLabel.TRAUMA
    return result


def test_irrelevant_salient_memory_cannot_bypass_relevance_gate():
    retriever = RelevanceFirstRetriever()
    relevant = MemoryCandidate(
        memory_id="relevant-ordinary",
        trajectory_id="relevant",
        root_embedding=(0.0, 1.0, 0.0, 0.0),
        transition=transition("relevant"),
        age=100,
    )
    irrelevant_trauma = MemoryCandidate(
        memory_id="irrelevant-trauma",
        trajectory_id="irrelevant",
        root_embedding=(1.0, 0.0, 0.0, 0.0),
        transition=trauma_transition(),
        age=0,
    )

    trace = retriever.retrieve(
        (0.0, 1.0, 0.0, 0.0),
        (relevant, irrelevant_trauma),
    )

    assert trace.eligible_memory_ids == ("relevant-ordinary",)
    assert tuple(item.memory_id for item in trace.selections) == ("relevant-ordinary",)
    assert ("irrelevant-trauma", "below-relevance-threshold") in trace.rejected


def test_trajectory_deduplication_limits_final_slots():
    retriever = RelevanceFirstRetriever()
    candidates = (
        MemoryCandidate("a1", "trajectory-a", (1.0, 0.0, 0.0, 0.0), transition("a1")),
        MemoryCandidate("a2", "trajectory-a", (0.99, 0.1, 0.0, 0.0), transition("a2")),
        MemoryCandidate("a3", "trajectory-a", (0.98, 0.2, 0.0, 0.0), transition("a3")),
        MemoryCandidate("b1", "trajectory-b", (0.95, 0.31, 0.0, 0.0), transition("b1")),
    )
    trace = retriever.retrieve((1.0, 0.0, 0.0, 0.0), candidates)

    trajectories = [item.trajectory_id for item in trace.selections]
    assert trajectories.count("trajectory-a") == 1
    assert trajectories.count("trajectory-b") == 1
    assert trace.deduplicated_memory_ids == ("a1", "b1")


def test_deleted_memory_is_rejected_and_adjustment_is_bounded():
    retriever = RelevanceFirstRetriever()
    active = MemoryCandidate(
        "active",
        "active-trajectory",
        (1.0, 0.0, 0.0, 0.0),
        transition("active"),
        age=1000,
    )
    deleted = MemoryCandidate(
        "deleted",
        "deleted-trajectory",
        (1.0, 0.0, 0.0, 0.0),
        transition("deleted"),
        active=False,
    )
    trace = retriever.retrieve((1.0, 0.0, 0.0, 0.0), (active, deleted))

    assert ("deleted", "inactive") in trace.rejected
    assert len(trace.selections) == 1
    assert abs(trace.selections[0].bounded_adjustment) <= 0.05
