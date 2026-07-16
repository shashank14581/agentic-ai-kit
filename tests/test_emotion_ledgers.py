from __future__ import annotations

from agentic_ai.memory.emotion import (
    AppraisalEngine,
    EvidenceDirection,
    EvidenceItem,
    Event,
    HistoricalOutcome,
    IdentityFact,
    IdentityLedger,
    IdentityState,
    OutcomeEvidence,
    OutcomeStatus,
    PolicyLedger,
)


def evidence(direction: EvidenceDirection) -> OutcomeEvidence:
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


def test_identity_ledger_adds_and_invalidates_tail_facts():
    engine = AppraisalEngine()
    initial = IdentityState(
        positive_facts=(IdentityFact("initial", (0.0, 1.0, 0.0, 0.0)),)
    )
    ledger = IdentityLedger(initial)
    event = Event(
        event_id="extreme-success",
        timestep=1,
        context_id="tools",
        action_id="correct-tool",
        embedding=(1.0, 0.0, 0.0, 0.0),
        expected_valence=-0.5,
    )
    transition = engine.evaluate(
        IdentityState(supplied_vector=(0.8, 0.6, 0.0, 0.0)),
        event,
        evidence(EvidenceDirection.SUCCESS),
    )

    fact_id = ledger.apply(event, transition)
    assert fact_id == "extreme-success:positive"
    assert len(ledger.state().positive_facts) == 2
    assert len(ledger.state().negative_facts) == 0
    assert ledger.apply(event, transition) == fact_id
    assert len(ledger.state().positive_facts) == 2

    assert ledger.invalidate_source_event(event.event_id, "corrected") == 1
    assert len(ledger.state().positive_facts) == 1
    assert len(ledger.events) == 2


def test_policy_ledger_replays_after_invalidation():
    engine = AppraisalEngine()
    ledger = PolicyLedger()
    identity = IdentityState(supplied_vector=(1.0, 0.0, 0.0, 0.0))
    events = []

    for timestep in (1, 2, 3):
        event = Event(
            event_id=f"success-{timestep}",
            timestep=timestep,
            context_id="tools",
            action_id="correct-tool",
            embedding=(1.0, 0.0, 0.0, 0.0),
            expected_valence=0.0,
        )
        transition = engine.evaluate(
            identity,
            event,
            evidence(EvidenceDirection.SUCCESS),
        )
        ledger.apply(event, transition)
        events.append(event)

    before = ledger.state("tools", "correct-tool")
    assert before.repeat > 0.0
    assert before.memory_q > 0.0

    ledger.invalidate_source_event(events[1].event_id, "deleted")
    after = ledger.state("tools", "correct-tool")

    replay = PolicyLedger()
    for event in (events[0], events[2]):
        transition = engine.evaluate(
            identity,
            event,
            evidence(EvidenceDirection.SUCCESS),
        )
        replay.apply(event, transition)

    assert after == replay.state("tools", "correct-tool")
    assert after.repeat < before.repeat


def test_recurring_failure_adds_negative_identity_and_policy_evidence():
    engine = AppraisalEngine()
    identity = IdentityState(supplied_vector=(-0.29, 0.9570266, 0.0, 0.0))
    identity_ledger = IdentityLedger(
        IdentityState(
            positive_facts=(IdentityFact("initial", (0.0, 1.0, 0.0, 0.0)),)
        )
    )
    policy_ledger = PolicyLedger()
    history = []
    policy_scores = []

    for timestep in (1, 2, 3):
        event = Event(
            event_id=f"failure-{timestep}",
            timestep=timestep,
            context_id="planning",
            action_id="invalid-plan",
            embedding=(1.0, 0.0, 0.0, 0.0),
            expected_valence=0.0,
        )
        transition = engine.evaluate(
            identity,
            event,
            evidence(EvidenceDirection.FAILURE),
            tuple(history),
        )
        identity_ledger.apply(event, transition)
        policy_ledger.apply(event, transition)
        policy_scores.append(policy_ledger.state("planning", "invalid-plan"))
        history.append(
            HistoricalOutcome(
                event.event_id,
                timestep,
                event.recurrence_vector,
                OutcomeStatus.FAILURE,
                transition.appraisal.failure_confidence,
                transition.appraisal.evidence_quality,
            )
        )

    assert len(identity_ledger.state().negative_facts) == 2
    assert policy_scores[0].avoidance < policy_scores[1].avoidance
    assert policy_scores[1].avoidance < policy_scores[2].avoidance
    assert policy_scores[2].memory_q < policy_scores[0].memory_q
