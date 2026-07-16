"""Immutable domain models for emotion-aware episodic memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import TypeAlias


Vector: TypeAlias = tuple[float, ...]


def _validate_range(name: str, value: float, low: float, high: float) -> None:
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}.")


def _validate_vector(name: str, vector: Vector) -> None:
    if not vector:
        raise ValueError(f"{name} must not be empty.")


def _freeze_vector(vector: Vector) -> Vector:
    return tuple(float(value) for value in vector)


class EvidenceDirection(IntEnum):
    FAILURE = -1
    SUCCESS = 1


class OutcomeStatus(str, Enum):
    UNCONFIRMED = "unconfirmed"
    DISPUTED = "disputed"
    SUCCESS = "success"
    FAILURE = "failure"


class OutcomeLabel(str, Enum):
    ORDINARY = "ordinary"
    SUCCESS = "success"
    FAILURE = "failure"
    WOUND = "wound"
    TRAUMA = "trauma"


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    direction: EvidenceDirection
    magnitude: float
    reliability: float
    provenance: str
    source: str = "environment"
    embedding: Vector | None = None
    active: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "direction", EvidenceDirection(self.direction))
        if self.embedding is not None:
            object.__setattr__(self, "embedding", _freeze_vector(self.embedding))
        _validate_range("magnitude", self.magnitude, 0.0, 1.0)
        _validate_range("reliability", self.reliability, 0.0, 1.0)
        if not self.evidence_id:
            raise ValueError("evidence_id must not be empty.")
        if not self.provenance:
            raise ValueError("provenance must not be empty.")


@dataclass(frozen=True)
class OutcomeEvidence:
    items: tuple[EvidenceItem, ...] = ()
    source_quality: float = 1.0
    provenance_quality: float = 1.0
    corroboration_quality: float = 1.0
    integrity_quality: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        for name in (
            "source_quality",
            "provenance_quality",
            "corroboration_quality",
            "integrity_quality",
        ):
            _validate_range(name, getattr(self, name), 0.0, 1.0)


@dataclass(frozen=True)
class IdentityFact:
    fact_id: str
    embedding: Vector
    weight: float = 1.0
    provenance: str = "configured"
    active: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "embedding", _freeze_vector(self.embedding))
        _validate_vector("identity fact embedding", self.embedding)
        if self.weight <= 0.0:
            raise ValueError("Identity fact weight must be positive.")


@dataclass(frozen=True)
class IdentityState:
    positive_facts: tuple[IdentityFact, ...] = ()
    negative_facts: tuple[IdentityFact, ...] = ()
    supplied_vector: Vector | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "positive_facts", tuple(self.positive_facts))
        object.__setattr__(self, "negative_facts", tuple(self.negative_facts))
        if self.supplied_vector is not None:
            object.__setattr__(self, "supplied_vector", _freeze_vector(self.supplied_vector))
            _validate_vector("supplied identity vector", self.supplied_vector)
        elif not any(fact.active for fact in self.positive_facts):
            raise ValueError("Identity requires a supplied vector or an active positive fact.")


@dataclass(frozen=True)
class Event:
    event_id: str
    timestep: int
    context_id: str
    action_id: str
    embedding: Vector
    expected_valence: float
    trajectory_id: str | None = None
    extreme_consequence: float = 0.0
    context_action_embedding: Vector | None = None
    provenance: str = "environment"

    def __post_init__(self) -> None:
        object.__setattr__(self, "embedding", _freeze_vector(self.embedding))
        if self.context_action_embedding is not None:
            object.__setattr__(
                self,
                "context_action_embedding",
                _freeze_vector(self.context_action_embedding),
            )
        if self.timestep < 0:
            raise ValueError("timestep must be non-negative.")
        _validate_vector("event embedding", self.embedding)
        if self.context_action_embedding is not None:
            _validate_vector("context-action embedding", self.context_action_embedding)
        _validate_range("expected_valence", self.expected_valence, -1.0, 1.0)
        _validate_range("extreme_consequence", self.extreme_consequence, 0.0, 1.0)

    @property
    def recurrence_vector(self) -> Vector:
        return self.context_action_embedding or self.embedding


@dataclass(frozen=True)
class HistoricalOutcome:
    event_id: str
    timestep: int
    context_action_embedding: Vector
    status: OutcomeStatus
    failure_confidence: float
    evidence_quality: float
    active: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "context_action_embedding",
            _freeze_vector(self.context_action_embedding),
        )
        _validate_vector("historical context-action embedding", self.context_action_embedding)
        _validate_range("failure_confidence", self.failure_confidence, 0.0, 1.0)
        _validate_range("evidence_quality", self.evidence_quality, 0.0, 1.0)


@dataclass(frozen=True)
class Appraisal:
    identity_alignment: float
    valence: float
    expectation_violation: float
    salience: float
    recurrence: float
    weighted_recurrence: float
    occurrence_count: int
    success_confidence: float
    failure_confidence: float
    evidence_quality: float
    outcome_status: OutcomeStatus

    @property
    def vector(self) -> Vector:
        return (
            self.identity_alignment,
            self.valence,
            self.expectation_violation,
            self.salience,
            self.recurrence,
        )


@dataclass(frozen=True)
class AffectiveState:
    repeat: float
    avoidance: float
    identity_update_pressure: float
    retention_floor: float
    expectation_update: float
    success_intensity: float
    injury_intensity: float
    positive_tail_gate: float
    negative_tail_gate: float
    operational_label: OutcomeLabel

    @property
    def vector(self) -> Vector:
        return (
            self.repeat,
            self.avoidance,
            self.identity_update_pressure,
            self.retention_floor,
            self.expectation_update,
        )


@dataclass(frozen=True)
class TransitionResult:
    event_id: str
    identity_vector: Vector
    appraisal: Appraisal
    state: AffectiveState
    positive_identity_update: bool
    negative_identity_update: bool
    configuration_version: str
    trace: tuple[str, ...] = field(default_factory=tuple)
