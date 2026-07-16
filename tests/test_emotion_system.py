from __future__ import annotations

from agentic_ai.memory.emotion import (
    EmotionMemorySystem,
    EpisodeOperation,
    EvidenceDirection,
    EvidenceItem,
    Event,
    IdentityState,
    OutcomeEvidence,
    OutcomeLabel,
)


def evidence(direction: EvidenceDirection | None) -> OutcomeEvidence:
    if direction is None:
        return OutcomeEvidence()
    return OutcomeEvidence(
        items=(
            EvidenceItem(
                evidence_id=f"evidence-{direction.value}",
                direction=direction,
                magnitude=1.0,
                reliability=1.0,
                provenance="environment",
            ),
        )
    )


def failure_event(event_id: str, timestep: int) -> Event:
    return Event(
        event_id=event_id,
        timestep=timestep,
        context_id="planning",
        action_id="invalid-plan",
        embedding=(1.0, 0.0, 0.0, 0.0),
        expected_valence=0.0,
        trajectory_id="planning-trajectory",
    )


def system() -> EmotionMemorySystem:
    return EmotionMemorySystem(
        IdentityState(supplied_vector=(-0.29, 0.9570266, 0.0, 0.0))
    )


def test_deleting_early_failure_recomputes_downstream_state():
    memory = system()
    for timestep in (1, 2, 3):
        memory.ingest(
            failure_event(f"failure-{timestep}", timestep),
            evidence(EvidenceDirection.FAILURE),
        )

    assert memory.timeline.get_event("failure-3").transition.state.operational_label is OutcomeLabel.TRAUMA
    memory.delete("failure-2", "invalid evaluator result")

    counterfactual = system()
    counterfactual.ingest(failure_event("failure-1", 1), evidence(EvidenceDirection.FAILURE))
    counterfactual.ingest(failure_event("failure-3", 3), evidence(EvidenceDirection.FAILURE))

    assert memory.snapshot() == counterfactual.snapshot()
    assert memory.timeline.get_event("failure-3").transition.state.operational_label is OutcomeLabel.WOUND
    assert memory.policy_state("planning", "invalid-plan") == counterfactual.policy_state(
        "planning", "invalid-plan"
    )


def test_correcting_failure_to_unconfirmed_removes_derived_effects():
    memory = EmotionMemorySystem(
        IdentityState(supplied_vector=(-1.0, 0.0, 0.0, 0.0))
    )
    event = Event(
        event_id="corrected-event",
        timestep=1,
        context_id="tools",
        action_id="unsupported-tool",
        embedding=(1.0, 0.0, 0.0, 0.0),
        expected_valence=0.0,
        extreme_consequence=1.0,
    )
    first = memory.ingest(event, evidence(EvidenceDirection.FAILURE))
    assert first.negative_identity_update
    assert memory.snapshot().negative_fact_count == 1

    corrected = memory.correct(
        event.event_id,
        OutcomeEvidence(),
        "environment withdrew the failure result",
    )
    assert corrected.state.operational_label is OutcomeLabel.ORDINARY
    assert memory.snapshot().negative_fact_count == 0
    assert memory.policy_state("tools", "unsupported-tool").avoidance == 0.0
    assert [entry.operation for entry in memory.journal] == [
        EpisodeOperation.INGEST,
        EpisodeOperation.CORRECT,
    ]


def test_deleted_or_expired_events_are_not_retrievable():
    memory = system()
    memory.ingest(failure_event("delete-me", 1), evidence(EvidenceDirection.FAILURE))
    memory.ingest(failure_event("expire-me", 2), evidence(EvidenceDirection.FAILURE))

    memory.delete("delete-me", "user deletion")
    memory.expire("expire-me", "retention expiry")
    trace = memory.retrieve((1.0, 0.0, 0.0, 0.0), current_timestep=10)

    assert trace.eligible_memory_ids == ()
    assert trace.selections == ()
    assert memory.snapshot().active_event_ids == ()


def test_journal_and_audit_are_append_only_and_versioned():
    memory = system()
    event = failure_event("versioned", 1)
    memory.ingest(event, evidence(EvidenceDirection.FAILURE))
    memory.correct("versioned", evidence(None), "correction")
    memory.delete("versioned", "deletion")

    assert [entry.version for entry in memory.journal] == [1, 2, 3]
    assert [entry.operation for entry in memory.audit_log] == [
        EpisodeOperation.INGEST,
        EpisodeOperation.CORRECT,
        EpisodeOperation.DELETE,
    ]
    assert [entry.sequence for entry in memory.audit_log] == [0, 1, 2]
    assert memory.audit_log[-1].active_event_count == 0
