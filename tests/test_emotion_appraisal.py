from __future__ import annotations

import math

import pytest

from agentic_ai.memory.emotion import (
    AppraisalEngine,
    EvidenceDirection,
    EvidenceItem,
    Event,
    HistoricalOutcome,
    IdentityFact,
    IdentityState,
    OutcomeEvidence,
    OutcomeLabel,
    OutcomeStatus,
)


def failure_evidence(magnitude: float = 1.0) -> OutcomeEvidence:
    return OutcomeEvidence(
        items=(
            EvidenceItem(
                evidence_id="failure",
                direction=EvidenceDirection.FAILURE,
                magnitude=magnitude,
                reliability=1.0,
                provenance="environment",
            ),
        )
    )


def test_same_event_different_identity_changes_appraisal_not_evidence():
    engine = AppraisalEngine()
    event = Event(
        event_id="same-event",
        timestep=1,
        context_id="tools",
        action_id="unsupported-tool",
        embedding=(1.0, 0.0, 0.0, 0.0),
        expected_valence=0.0,
    )
    evidence = failure_evidence(0.9)
    aligned = IdentityState(
        positive_facts=(IdentityFact("p", (1.0, 0.0, 0.0, 0.0)),),
        negative_facts=(IdentityFact("n", (0.0, 1.0, 0.0, 0.0)),),
    )
    conflicted = IdentityState(
        positive_facts=(IdentityFact("p", (0.0, 1.0, 0.0, 0.0)),),
        negative_facts=(IdentityFact("n", (1.0, 0.0, 0.0, 0.0)),),
    )

    result_a = engine.evaluate(aligned, event, evidence)
    result_b = engine.evaluate(conflicted, event, evidence)

    assert result_a.appraisal.success_confidence == result_b.appraisal.success_confidence
    assert result_a.appraisal.failure_confidence == result_b.appraisal.failure_confidence
    assert result_a.appraisal.identity_alignment == pytest.approx(0.894427, abs=1e-6)
    assert result_b.appraisal.identity_alignment == pytest.approx(-0.447214, abs=1e-6)
    assert result_b.state.avoidance > result_a.state.avoidance
    distance = sum(abs(a - b) for a, b in zip(result_a.state.vector, result_b.state.vector))
    assert distance >= 0.10


def test_recurrence_progresses_failure_to_wound_to_trauma():
    engine = AppraisalEngine()
    identity = IdentityState(supplied_vector=(-0.29, 0.9570266, 0.0, 0.0))
    evidence = failure_evidence()
    history: list[HistoricalOutcome] = []
    results = []

    for timestep in (1, 2, 3):
        event = Event(
            event_id=f"failure-{timestep}",
            timestep=timestep,
            context_id="planning",
            action_id="invalid-plan",
            embedding=(1.0, 0.0, 0.0, 0.0),
            expected_valence=0.0,
        )
        result = engine.evaluate(identity, event, evidence, tuple(history))
        results.append(result)
        history.append(
            HistoricalOutcome(
                event_id=event.event_id,
                timestep=timestep,
                context_action_embedding=event.recurrence_vector,
                status=OutcomeStatus.FAILURE,
                failure_confidence=result.appraisal.failure_confidence,
                evidence_quality=result.appraisal.evidence_quality,
            )
        )

    assert [result.appraisal.occurrence_count for result in results] == [1, 2, 3]
    assert [result.state.operational_label for result in results] == [
        OutcomeLabel.FAILURE,
        OutcomeLabel.WOUND,
        OutcomeLabel.TRAUMA,
    ]
    assert results[0].state.injury_intensity < results[1].state.injury_intensity
    assert results[1].state.injury_intensity < results[2].state.injury_intensity
    assert not results[0].negative_identity_update
    assert results[1].negative_identity_update
    assert results[2].negative_identity_update


def test_extreme_success_updates_positive_identity_tail():
    engine = AppraisalEngine()
    result = engine.evaluate(
        IdentityState(supplied_vector=(0.8, 0.6, 0.0, 0.0)),
        Event(
            event_id="success",
            timestep=1,
            context_id="tools",
            action_id="correct-tool",
            embedding=(1.0, 0.0, 0.0, 0.0),
            expected_valence=-0.5,
        ),
        OutcomeEvidence(
            items=(
                EvidenceItem(
                    evidence_id="success",
                    direction=EvidenceDirection.SUCCESS,
                    magnitude=1.0,
                    reliability=1.0,
                    provenance="environment",
                ),
            )
        ),
    )

    assert result.state.operational_label is OutcomeLabel.SUCCESS
    assert result.state.repeat > 0.0
    assert result.positive_identity_update
    assert not result.negative_identity_update


def test_disputed_evidence_cannot_create_tail_update():
    engine = AppraisalEngine()
    evidence = OutcomeEvidence(
        items=(
            EvidenceItem(
                "success",
                EvidenceDirection.SUCCESS,
                0.9,
                1.0,
                "evaluator-a",
            ),
            EvidenceItem(
                "failure",
                EvidenceDirection.FAILURE,
                0.9,
                1.0,
                "evaluator-b",
            ),
        )
    )
    result = engine.evaluate(
        IdentityState(supplied_vector=(1.0, 0.0, 0.0, 0.0)),
        Event(
            event_id="disputed",
            timestep=1,
            context_id="qa",
            action_id="answer",
            embedding=(1.0, 0.0, 0.0, 0.0),
            expected_valence=0.0,
        ),
        evidence,
    )

    assert result.appraisal.outcome_status is OutcomeStatus.DISPUTED
    assert result.state.operational_label is OutcomeLabel.ORDINARY
    assert result.state.repeat == 0.0
    assert result.state.avoidance == 0.0
    assert not result.positive_identity_update
    assert not result.negative_identity_update


def test_transition_is_deterministic():
    engine = AppraisalEngine()
    identity = IdentityState(supplied_vector=(1.0, 0.0, 0.0, 0.0))
    event = Event(
        event_id="deterministic",
        timestep=1,
        context_id="qa",
        action_id="answer",
        embedding=(1.0, 0.0, 0.0, 0.0),
        expected_valence=0.0,
    )
    evidence = failure_evidence(0.9)

    first = engine.evaluate(identity, event, evidence)
    second = engine.evaluate(identity, event, evidence)

    assert first == second
