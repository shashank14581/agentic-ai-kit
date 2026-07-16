"""Append-only identity and policy ledgers with deterministic replay."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agentic_ai.memory.emotion.appraisal import build_identity_vector
from agentic_ai.memory.emotion.config import EmotionConfig
from agentic_ai.memory.emotion.models import (
    Event,
    IdentityFact,
    IdentityState,
    TransitionResult,
    Vector,
)


class LedgerOperation(str, Enum):
    ADD = "add"
    INVALIDATE = "invalidate"


class IdentityPolarity(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass(frozen=True)
class IdentityLedgerEvent:
    sequence: int
    operation: LedgerOperation
    fact_id: str
    source_event_id: str
    polarity: IdentityPolarity | None = None
    fact: IdentityFact | None = None
    reason: str | None = None


@dataclass(frozen=True)
class PolicyEvidence:
    evidence_id: str
    source_event_id: str
    context_id: str
    action_id: str
    repeat: float
    avoidance: float
    evidence_quality: float


@dataclass(frozen=True)
class PolicyLedgerEvent:
    sequence: int
    operation: LedgerOperation
    evidence_id: str
    source_event_id: str
    evidence: PolicyEvidence | None = None
    reason: str | None = None


@dataclass(frozen=True)
class PolicyState:
    repeat: float
    avoidance: float

    @property
    def memory_q(self) -> float:
        return self.repeat - self.avoidance


class IdentityLedger:
    """Event-sourced positive and negative identity fact ledger."""

    def __init__(
        self,
        initial_state: IdentityState,
        config: EmotionConfig | None = None,
    ) -> None:
        self.config = config or EmotionConfig()
        positive_facts = list(initial_state.positive_facts)
        if initial_state.supplied_vector is not None:
            positive_facts.append(
                IdentityFact(
                    fact_id="__initial_supplied_identity__",
                    embedding=initial_state.supplied_vector,
                    weight=1.0,
                    provenance="supplied",
                )
            )
        self._initial_state = IdentityState(
            positive_facts=tuple(positive_facts),
            negative_facts=initial_state.negative_facts,
        )
        self._events: list[IdentityLedgerEvent] = []

    @property
    def events(self) -> tuple[IdentityLedgerEvent, ...]:
        return tuple(self._events)

    def apply(self, event: Event, transition: TransitionResult) -> str | None:
        """Append a positive or negative fact when the transition crosses a tail."""
        if event.event_id != transition.event_id:
            raise ValueError("Event and transition identifiers must match.")

        if transition.positive_identity_update:
            fact_id = f"{event.event_id}:positive"
            weight = (
                transition.appraisal.evidence_quality
                * transition.state.positive_tail_gate
                * transition.state.repeat
            )
            polarity = IdentityPolarity.POSITIVE
        elif transition.negative_identity_update:
            fact_id = f"{event.event_id}:negative"
            transition_config = self.config.identity_transition
            weight = min(
                transition_config.maximum_negative_fact_weight,
                transition.appraisal.evidence_quality
                * transition.state.negative_tail_gate
                * transition.state.avoidance
                * (
                    1.0
                    + transition_config.recurrence_weight_multiplier
                    * transition.appraisal.recurrence
                ),
            )
            polarity = IdentityPolarity.NEGATIVE
        else:
            return None

        if fact_id in self._active_facts():
            return fact_id

        fact = IdentityFact(
            fact_id=fact_id,
            embedding=event.embedding,
            weight=max(weight, 1.0e-12),
            provenance=event.provenance,
        )
        self._events.append(
            IdentityLedgerEvent(
                sequence=len(self._events),
                operation=LedgerOperation.ADD,
                fact_id=fact_id,
                source_event_id=event.event_id,
                polarity=polarity,
                fact=fact,
            )
        )
        return fact_id

    def invalidate_source_event(self, source_event_id: str, reason: str) -> int:
        """Append invalidations for every active fact derived from an event."""
        active = self._active_facts()
        targets = [
            fact_id
            for fact_id, ledger_event in active.items()
            if ledger_event.source_event_id == source_event_id
        ]
        for fact_id in targets:
            self._events.append(
                IdentityLedgerEvent(
                    sequence=len(self._events),
                    operation=LedgerOperation.INVALIDATE,
                    fact_id=fact_id,
                    source_event_id=source_event_id,
                    reason=reason,
                )
            )
        return len(targets)

    def state(self) -> IdentityState:
        active = self._active_facts().values()
        positives = list(self._initial_state.positive_facts)
        negatives = list(self._initial_state.negative_facts)
        for ledger_event in active:
            if ledger_event.fact is None:
                continue
            if ledger_event.polarity is IdentityPolarity.POSITIVE:
                positives.append(ledger_event.fact)
            elif ledger_event.polarity is IdentityPolarity.NEGATIVE:
                negatives.append(ledger_event.fact)
        return IdentityState(
            positive_facts=tuple(positives),
            negative_facts=tuple(negatives),
            supplied_vector=self._initial_state.supplied_vector,
        )

    def vector(self) -> Vector:
        return build_identity_vector(
            self.state(),
            self.config.identity.contrastive_weight,
        )

    def _active_facts(self) -> dict[str, IdentityLedgerEvent]:
        active: dict[str, IdentityLedgerEvent] = {}
        for ledger_event in self._events:
            if ledger_event.operation is LedgerOperation.ADD:
                active[ledger_event.fact_id] = ledger_event
            elif ledger_event.operation is LedgerOperation.INVALIDATE:
                active.pop(ledger_event.fact_id, None)
        return active


class PolicyLedger:
    """Event-sourced repeat and avoidance evidence for context-action keys."""

    def __init__(self, config: EmotionConfig | None = None) -> None:
        self.config = config or EmotionConfig()
        self._events: list[PolicyLedgerEvent] = []

    @property
    def events(self) -> tuple[PolicyLedgerEvent, ...]:
        return tuple(self._events)

    def apply(self, event: Event, transition: TransitionResult) -> str | None:
        if event.event_id != transition.event_id:
            raise ValueError("Event and transition identifiers must match.")
        if transition.state.repeat <= 0.0 and transition.state.avoidance <= 0.0:
            return None

        evidence_id = f"{event.event_id}:policy"
        if evidence_id in self._active_evidence():
            return evidence_id

        evidence = PolicyEvidence(
            evidence_id=evidence_id,
            source_event_id=event.event_id,
            context_id=event.context_id,
            action_id=event.action_id,
            repeat=transition.state.repeat,
            avoidance=transition.state.avoidance,
            evidence_quality=transition.appraisal.evidence_quality,
        )
        self._events.append(
            PolicyLedgerEvent(
                sequence=len(self._events),
                operation=LedgerOperation.ADD,
                evidence_id=evidence_id,
                source_event_id=event.event_id,
                evidence=evidence,
            )
        )
        return evidence_id

    def invalidate_source_event(self, source_event_id: str, reason: str) -> int:
        active = self._active_evidence()
        targets = [
            evidence_id
            for evidence_id, ledger_event in active.items()
            if ledger_event.source_event_id == source_event_id
        ]
        for evidence_id in targets:
            self._events.append(
                PolicyLedgerEvent(
                    sequence=len(self._events),
                    operation=LedgerOperation.INVALIDATE,
                    evidence_id=evidence_id,
                    source_event_id=source_event_id,
                    reason=reason,
                )
            )
        return len(targets)

    def state(self, context_id: str, action_id: str) -> PolicyState:
        repeat = 0.0
        avoidance = 0.0
        persistence = self.config.policy.persistence
        ordered = sorted(
            self._active_evidence().values(),
            key=lambda ledger_event: ledger_event.sequence,
        )
        for ledger_event in ordered:
            evidence = ledger_event.evidence
            if evidence is None:
                continue
            if evidence.context_id != context_id or evidence.action_id != action_id:
                continue
            repeat = persistence * repeat + evidence.evidence_quality * evidence.repeat
            avoidance = (
                persistence * avoidance
                + evidence.evidence_quality * evidence.avoidance
            )
        return PolicyState(repeat=repeat, avoidance=avoidance)

    def _active_evidence(self) -> dict[str, PolicyLedgerEvent]:
        active: dict[str, PolicyLedgerEvent] = {}
        for ledger_event in self._events:
            if ledger_event.operation is LedgerOperation.ADD:
                active[ledger_event.evidence_id] = ledger_event
            elif ledger_event.operation is LedgerOperation.INVALIDATE:
                active.pop(ledger_event.evidence_id, None)
        return active
