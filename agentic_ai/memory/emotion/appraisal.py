"""Deterministic appraisal and functional state transition engine."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np

from agentic_ai.memory.emotion.config import EmotionConfig
from agentic_ai.memory.emotion.models import (
    AffectiveState,
    Appraisal,
    EvidenceDirection,
    Event,
    HistoricalOutcome,
    IdentityFact,
    IdentityState,
    OutcomeEvidence,
    OutcomeLabel,
    OutcomeStatus,
    TransitionResult,
    Vector,
)


def _clip(value: float, low: float, high: float) -> float:
    return float(min(high, max(low, value)))


def _positive(value: float) -> float:
    return max(0.0, value)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def normalize(vector: Vector) -> Vector:
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 1.0e-12:
        raise ValueError("Cannot normalize a zero vector.")
    return tuple(float(value) for value in array / norm)


def cosine(left: Vector, right: Vector) -> float:
    if len(left) != len(right):
        raise ValueError("Vector dimensions must match.")
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator <= 1.0e-12:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    return _clip(float(np.dot(left_array, right_array) / denominator), -1.0, 1.0)


def _weighted_mean(facts: Iterable[IdentityFact], dimension: int) -> np.ndarray:
    active = [fact for fact in facts if fact.active]
    if not active:
        return np.zeros(dimension, dtype=np.float64)
    for fact in active:
        if len(fact.embedding) != dimension:
            raise ValueError("All identity fact dimensions must match.")
    total_weight = sum(fact.weight for fact in active)
    return sum(
        fact.weight * np.asarray(fact.embedding, dtype=np.float64)
        for fact in active
    ) / total_weight


def build_identity_vector(identity: IdentityState, contrastive_weight: float) -> Vector:
    """Build the normalized contrastive identity vector."""
    if identity.supplied_vector is not None:
        return normalize(identity.supplied_vector)
    dimension = len(next(f.embedding for f in identity.positive_facts if f.active))
    positive_mean = _weighted_mean(identity.positive_facts, dimension)
    negative_mean = _weighted_mean(identity.negative_facts, dimension)
    combined = positive_mean - contrastive_weight * negative_mean
    return normalize(tuple(float(value) for value in combined))


class AppraisalEngine:
    """Pure deterministic transition from identity, event, evidence, and history."""

    def __init__(self, config: EmotionConfig | None = None):
        self.config = config or EmotionConfig()

    def evaluate(
        self,
        identity: IdentityState,
        event: Event,
        evidence: OutcomeEvidence,
        history: tuple[HistoricalOutcome, ...] = (),
    ) -> TransitionResult:
        identity_vector = build_identity_vector(
            identity,
            self.config.identity.contrastive_weight,
        )
        if len(identity_vector) != len(event.embedding):
            raise ValueError("Identity and event embedding dimensions must match.")

        success_confidence, failure_confidence = self._outcome_confidences(evidence)
        evidence_quality = self._evidence_quality(evidence)
        status = self._outcome_status(success_confidence, failure_confidence)
        valence = _clip(success_confidence - failure_confidence, -1.0, 1.0)
        alignment = cosine(event.embedding, identity_vector)
        expectation_violation = _clip(valence - event.expected_valence, -1.0, 1.0)
        weighted_recurrence, occurrence_count, recurrence = self._recurrence(
            event,
            status,
            history,
        )
        salience = self._salience(
            alignment,
            valence,
            expectation_violation,
            success_confidence,
            failure_confidence,
        )

        appraisal = Appraisal(
            identity_alignment=alignment,
            valence=valence,
            expectation_violation=expectation_violation,
            salience=salience,
            recurrence=recurrence,
            weighted_recurrence=weighted_recurrence,
            occurrence_count=occurrence_count,
            success_confidence=success_confidence,
            failure_confidence=failure_confidence,
            evidence_quality=evidence_quality,
            outcome_status=status,
        )
        state = self._state(appraisal, event.extreme_consequence)

        positive_update = (
            status is OutcomeStatus.SUCCESS
            and state.identity_update_pressure
            >= self.config.identity.positive_ledger_threshold
        )
        negative_update = (
            status is OutcomeStatus.FAILURE
            and state.identity_update_pressure
            <= -self.config.identity.negative_ledger_threshold
        )

        return TransitionResult(
            event_id=event.event_id,
            identity_vector=identity_vector,
            appraisal=appraisal,
            state=state,
            positive_identity_update=positive_update,
            negative_identity_update=negative_update,
            configuration_version=self.config.architecture_version,
            trace=(
                f"outcome_status={status.value}",
                f"occurrence_count={occurrence_count}",
                f"operational_label={state.operational_label.value}",
            ),
        )

    @staticmethod
    def _deduplicated_items(evidence: OutcomeEvidence):
        selected = {}
        for item in evidence.items:
            if not item.active:
                continue
            key = (item.direction, item.source, item.provenance)
            contribution = item.reliability * item.magnitude
            previous = selected.get(key)
            if previous is None or contribution > previous.reliability * previous.magnitude:
                selected[key] = item
        return tuple(selected.values())

    def _outcome_confidences(self, evidence: OutcomeEvidence) -> tuple[float, float]:
        items = self._deduplicated_items(evidence)

        def aggregate(direction: EvidenceDirection) -> float:
            product = 1.0
            for item in items:
                if item.direction is direction:
                    product *= 1.0 - item.reliability * item.magnitude
            return _clip(1.0 - product, 0.0, 1.0)

        return aggregate(EvidenceDirection.SUCCESS), aggregate(EvidenceDirection.FAILURE)

    @staticmethod
    def _evidence_quality(evidence: OutcomeEvidence) -> float:
        return _clip(
            evidence.source_quality
            * evidence.provenance_quality
            * evidence.corroboration_quality
            * evidence.integrity_quality,
            0.0,
            1.0,
        )

    def _outcome_status(self, success: float, failure: float) -> OutcomeStatus:
        threshold = self.config.evidence.confirmation_threshold
        margin = self.config.evidence.confirmation_margin
        if success >= threshold and success - failure >= margin:
            return OutcomeStatus.SUCCESS
        if failure >= threshold and failure - success >= margin:
            return OutcomeStatus.FAILURE
        if success > 0.0 or failure > 0.0:
            return OutcomeStatus.DISPUTED
        return OutcomeStatus.UNCONFIRMED

    def _recurrence(
        self,
        event: Event,
        status: OutcomeStatus,
        history: tuple[HistoricalOutcome, ...],
    ) -> tuple[float, int, float]:
        if status is not OutcomeStatus.FAILURE:
            return 0.0, 0, 0.0

        config = self.config.recurrence
        weighted = 0.0
        prior_count = 0
        for previous in history:
            if not previous.active or previous.status is not OutcomeStatus.FAILURE:
                continue
            similarity = cosine(event.recurrence_vector, previous.context_action_embedding)
            if similarity < config.semantic_similarity_threshold:
                continue
            age = max(0, event.timestep - previous.timestep)
            weighted += (
                previous.evidence_quality
                * previous.failure_confidence
                * math.exp(-config.temporal_decay * age)
            )
            prior_count += 1

        occurrence_count = prior_count + int(config.occurrence_count_includes_current)
        recurrence = 1.0 - math.exp(-weighted / config.normalization_scale)
        return weighted, occurrence_count, _clip(recurrence, 0.0, 1.0)

    def _salience(
        self,
        alignment: float,
        valence: float,
        expectation_violation: float,
        success: float,
        failure: float,
    ) -> float:
        weights = self.config.salience
        return _clip(
            weights.valence * abs(valence)
            + weights.expectation_violation * abs(expectation_violation)
            + weights.identity_conflict * _positive(-alignment)
            + weights.outcome_confidence * max(success, failure),
            0.0,
            1.0,
        )

    def _state(self, appraisal: Appraisal, extreme_consequence: float) -> AffectiveState:
        repeat_config = self.config.repeat
        avoidance_config = self.config.avoidance

        confirmed_success = (
            appraisal.success_confidence
            if appraisal.outcome_status is OutcomeStatus.SUCCESS
            else 0.0
        )
        confirmed_failure = (
            appraisal.failure_confidence
            if appraisal.outcome_status is OutcomeStatus.FAILURE
            else 0.0
        )

        repeat = confirmed_success * _sigmoid(
            repeat_config.positive_valence * _positive(appraisal.valence)
            + repeat_config.identity_term * _positive(appraisal.identity_alignment)
            + repeat_config.expectation_violation
            * _positive(appraisal.expectation_violation)
            + repeat_config.salience * appraisal.salience
            - repeat_config.bias
        )
        avoidance = confirmed_failure * _sigmoid(
            avoidance_config.positive_valence * _positive(-appraisal.valence)
            + avoidance_config.identity_term
            * _positive(-appraisal.identity_alignment)
            + avoidance_config.expectation_violation
            * _positive(-appraisal.expectation_violation)
            + avoidance_config.salience * appraisal.salience
            + avoidance_config.recurrence * appraisal.recurrence
            - avoidance_config.bias
        )

        gates = self.config.tail_gates
        positive_gate = _sigmoid(
            gates.positive_sharpness * (repeat - gates.positive_threshold)
        )
        negative_gate = _sigmoid(
            gates.negative_sharpness * (avoidance - gates.negative_threshold)
        )
        identity_pressure = _clip(
            positive_gate * repeat - negative_gate * avoidance,
            -1.0,
            1.0,
        )

        retention = self.config.retention
        retention_floor = _clip(
            appraisal.evidence_quality
            * (
                retention.repeat * repeat
                + retention.avoidance * avoidance
                + retention.absolute_identity_pressure * abs(identity_pressure)
                + retention.recurring_negative_tail
                * confirmed_failure
                * appraisal.recurrence
                * negative_gate
            ),
            0.0,
            retention.maximum_floor,
        )

        expectation = self.config.expectation
        learning_rate = _clip(
            expectation.base_learning_rate
            + expectation.salience_learning_rate * appraisal.salience
            + expectation.recurrence_learning_rate * appraisal.recurrence
            + expectation.identity_pressure_learning_rate * abs(identity_pressure),
            0.0,
            expectation.maximum_learning_rate,
        )
        expectation_update = (
            appraisal.evidence_quality
            * learning_rate
            * appraisal.expectation_violation
        )

        regions = self.config.outcome_regions
        success_intensity = _clip(
            regions.success_repeat * repeat
            + regions.success_identity_pressure * _positive(identity_pressure)
            + regions.success_salience * appraisal.salience,
            0.0,
            1.0,
        )
        injury_intensity = _clip(
            regions.injury_avoidance * avoidance
            + regions.injury_identity_pressure * _positive(-identity_pressure)
            + regions.injury_salience * appraisal.salience
            + regions.injury_recurrence * appraisal.recurrence,
            0.0,
            1.0,
        )
        label = self._label(appraisal, injury_intensity, extreme_consequence)

        return AffectiveState(
            repeat=repeat,
            avoidance=avoidance,
            identity_update_pressure=identity_pressure,
            retention_floor=retention_floor,
            expectation_update=expectation_update,
            success_intensity=success_intensity,
            injury_intensity=injury_intensity,
            positive_tail_gate=positive_gate,
            negative_tail_gate=negative_gate,
            operational_label=label,
        )

    def _label(
        self,
        appraisal: Appraisal,
        injury: float,
        extreme_consequence: float,
    ) -> OutcomeLabel:
        if appraisal.outcome_status is OutcomeStatus.SUCCESS:
            return OutcomeLabel.SUCCESS
        if appraisal.outcome_status is not OutcomeStatus.FAILURE:
            return OutcomeLabel.ORDINARY

        regions = self.config.outcome_regions
        trauma = (
            extreme_consequence >= regions.trauma_extreme_consequence_threshold
            or (
                injury >= regions.trauma_injury_threshold
                and appraisal.occurrence_count >= regions.trauma_recurrence_count
            )
        )
        if trauma:
            return OutcomeLabel.TRAUMA

        wound = injury >= regions.wound_injury_threshold and (
            _positive(-appraisal.identity_alignment)
            >= regions.wound_identity_conflict_threshold
            or appraisal.occurrence_count >= regions.wound_recurrence_count
            or extreme_consequence >= regions.wound_extreme_consequence_threshold
        )
        if wound:
            return OutcomeLabel.WOUND
        return OutcomeLabel.FAILURE
