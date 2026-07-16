from __future__ import annotations

from agentic_ai.memory.emotion import (
    AppraisalEngine,
    EvidenceDirection,
    EvidenceItem,
    Event,
    HistoricalOutcome,
    IdentityState,
    OutcomeEvidence,
    OutcomeStatus,
    RetentionEngine,
    RetentionMode,
)


def failure_evidence() -> OutcomeEvidence:
    return OutcomeEvidence(
        items=(EvidenceItem("failure", EvidenceDirection.FAILURE, 1.0, 1.0, "env"),)
    )


def trauma_transition():
    engine = AppraisalEngine()
    identity = IdentityState(supplied_vector=(-0.29, 0.9570266, 0.0, 0.0))
    history = (
        HistoricalOutcome("f1", 1, (1.0, 0.0, 0.0, 0.0), OutcomeStatus.FAILURE, 1.0, 1.0),
        HistoricalOutcome("f2", 2, (1.0, 0.0, 0.0, 0.0), OutcomeStatus.FAILURE, 1.0, 1.0),
    )
    event = Event("f3", 3, "planning", "invalid", (1.0, 0.0, 0.0, 0.0), 0.0)
    return engine.evaluate(identity, event, failure_evidence(), history)


def test_emotion_retention_protects_trauma_from_identity_decay():
    transition = trauma_transition()
    engine = RetentionEngine()
    identity = engine.evaluate(transition, 150, RetentionMode.IDENTITY)
    emotion = engine.evaluate(transition, 150, RetentionMode.EMOTION)

    assert emotion.retention > identity.retention
    assert emotion.retention_floor > 0.0


def test_retention_modes_and_deletion_override():
    transition = trauma_transition()
    engine = RetentionEngine()

    assert engine.evaluate(transition, 150, "none").retention == 1.0
    assert engine.evaluate(transition, 150, "time").retention < 1.0
    assert engine.evaluate(transition, 150, "reward").retention > 0.0
    assert engine.evaluate(transition, 150, "emotion").retention > 0.0
    assert engine.evaluate(transition, 150, "emotion", active=False).retention == 0.0


def test_time_and_identity_retention_decrease_with_age():
    transition = trauma_transition()
    engine = RetentionEngine()
    for mode in (RetentionMode.TIME, RetentionMode.IDENTITY):
        early = engine.evaluate(transition, 10, mode).retention
        late = engine.evaluate(transition, 100, mode).retention
        assert late < early
