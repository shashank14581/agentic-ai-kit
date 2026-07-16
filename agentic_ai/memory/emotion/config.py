"""Typed configuration for the deterministic emotion architecture."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class EvidenceConfig:
    confirmation_threshold: float = 0.60
    confirmation_margin: float = 0.10


@dataclass(frozen=True)
class IdentityConfig:
    contrastive_weight: float = 0.50
    positive_ledger_threshold: float = 0.55
    negative_ledger_threshold: float = 0.55


@dataclass(frozen=True)
class RecurrenceConfig:
    semantic_similarity_threshold: float = 0.80
    temporal_decay: float = 0.01
    normalization_scale: float = 2.0
    wound_count: int = 2
    trauma_count: int = 3
    occurrence_count_includes_current: bool = True


@dataclass(frozen=True)
class SalienceConfig:
    valence: float = 0.30
    expectation_violation: float = 0.25
    identity_conflict: float = 0.25
    outcome_confidence: float = 0.20


@dataclass(frozen=True)
class EvidenceSignalConfig:
    positive_valence: float
    identity_term: float
    expectation_violation: float
    salience: float
    recurrence: float = 0.0
    bias: float = 0.0


@dataclass(frozen=True)
class TailGateConfig:
    positive_threshold: float = 0.75
    negative_threshold: float = 0.70
    positive_sharpness: float = 12.0
    negative_sharpness: float = 12.0


@dataclass(frozen=True)
class OutcomeRegionConfig:
    success_repeat: float = 0.55
    success_identity_pressure: float = 0.30
    success_salience: float = 0.15
    injury_avoidance: float = 0.45
    injury_identity_pressure: float = 0.25
    injury_salience: float = 0.15
    injury_recurrence: float = 0.15
    wound_injury_threshold: float = 0.50
    wound_identity_conflict_threshold: float = 0.30
    wound_recurrence_count: int = 2
    wound_extreme_consequence_threshold: float = 0.70
    trauma_injury_threshold: float = 0.72
    trauma_recurrence_count: int = 3
    trauma_extreme_consequence_threshold: float = 0.90


@dataclass(frozen=True)
class RetentionConfig:
    identity_decay_lambda: float = 0.01
    time_decay_lambda: float = 0.03
    maximum_floor: float = 0.95
    repeat: float = 0.30
    avoidance: float = 0.25
    absolute_identity_pressure: float = 0.35
    recurring_negative_tail: float = 0.45
    reward_ordinary_floor: float = 0.0
    reward_success_floor: float = 0.25
    reward_failure_floor: float = 0.25
    reward_wound_floor: float = 0.25
    reward_trauma_floor: float = 0.34


@dataclass(frozen=True)
class RetrievalConfig:
    semantic_relevance_threshold: float = 0.55
    candidate_top_k: int = 50
    final_top_k: int = 6
    maximum_results_per_trajectory: int = 1
    retention_weight: float = 0.02
    salience_weight: float = 0.02
    maximum_adjustment: float = 0.05


@dataclass(frozen=True)
class TreeAttentionConfig:
    enabled: bool = True
    apply_after_semantic_eligibility: bool = True
    apply_after_trajectory_deduplication: bool = True
    preserve_typed_edges: bool = True
    preserve_provenance_paths: bool = True
    attention_weights_must_sum_to_one: bool = True
    allow_cross_tree_attention: bool = False
    root_similarity_weight: float = 0.50
    tree_similarity_weight: float = 0.50
    depth_bias_per_level: float = 0.0


@dataclass(frozen=True)
class IdentityTransitionConfig:
    maximum_negative_fact_weight: float = 2.50
    recurrence_weight_multiplier: float = 1.00


@dataclass(frozen=True)
class PolicyConfig:
    persistence: float = 0.95


@dataclass(frozen=True)
class ExpectationConfig:
    base_learning_rate: float = 0.05
    salience_learning_rate: float = 0.05
    recurrence_learning_rate: float = 0.10
    identity_pressure_learning_rate: float = 0.10
    maximum_learning_rate: float = 0.30


@dataclass(frozen=True)
class EmotionConfig:
    """All parameters needed by the deterministic transition engine."""

    architecture_version: str = "0.1"
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    recurrence: RecurrenceConfig = field(default_factory=RecurrenceConfig)
    salience: SalienceConfig = field(default_factory=SalienceConfig)
    repeat: EvidenceSignalConfig = field(
        default_factory=lambda: EvidenceSignalConfig(
            positive_valence=1.20,
            identity_term=0.60,
            expectation_violation=0.80,
            salience=0.50,
            bias=1.20,
        )
    )
    avoidance: EvidenceSignalConfig = field(
        default_factory=lambda: EvidenceSignalConfig(
            positive_valence=1.10,
            identity_term=0.70,
            expectation_violation=0.80,
            salience=0.50,
            recurrence=0.90,
            bias=1.40,
        )
    )
    tail_gates: TailGateConfig = field(default_factory=TailGateConfig)
    outcome_regions: OutcomeRegionConfig = field(default_factory=OutcomeRegionConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    tree_attention: TreeAttentionConfig = field(default_factory=TreeAttentionConfig)
    identity_transition: IdentityTransitionConfig = field(
        default_factory=IdentityTransitionConfig
    )
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    expectation: ExpectationConfig = field(default_factory=ExpectationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "EmotionConfig":
        """Load the registered architecture YAML file."""
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ImportError("PyYAML is required to load emotion configuration.") from exc

        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, Mapping):
            raise ValueError("Emotion configuration must contain a YAML mapping.")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EmotionConfig":
        """Build typed configuration from the versioned mapping."""
        evidence = raw.get("evidence", {})
        identity = raw.get("identity", {})
        appraisal = raw.get("appraisal", {})
        recurrence = appraisal.get("recurrence", {})
        salience = appraisal.get("salience_weights", {})
        affective = raw.get("affective_state", {})
        repeat = affective.get("repeat", {})
        repeat_weights = repeat.get("weights", {})
        avoidance = affective.get("avoidance", {})
        avoidance_weights = avoidance.get("weights", {})
        gates = affective.get("tail_gates", {})
        regions = raw.get("outcome_regions", {})
        success_weights = regions.get("success_intensity_weights", {})
        injury_weights = regions.get("injury_intensity_weights", {})
        wound = regions.get("wound", {})
        trauma = regions.get("trauma", {})
        retention = raw.get("retention", {})
        floor_weights = retention.get("floor_weights", {})
        reward_floor = retention.get("reward_floor", {})
        retrieval = raw.get("retrieval", {})
        tie_break = retrieval.get("bounded_tie_break", {})
        tree_attention = raw.get("tree_attention", {})
        identity_transition = raw.get("identity_transition", {})
        policy = raw.get("policy", {})
        expectation = raw.get("expectation", {})

        return cls(
            architecture_version=str(raw.get("architecture_version", "0.1")),
            evidence=EvidenceConfig(
                confirmation_threshold=float(evidence.get("confirmation_threshold", 0.60)),
                confirmation_margin=float(evidence.get("confirmation_margin", 0.10)),
            ),
            identity=IdentityConfig(
                contrastive_weight=float(identity.get("contrastive_weight", 0.50)),
                positive_ledger_threshold=float(identity.get("positive_ledger_threshold", 0.55)),
                negative_ledger_threshold=float(identity.get("negative_ledger_threshold", 0.55)),
            ),
            recurrence=RecurrenceConfig(
                semantic_similarity_threshold=float(recurrence.get("semantic_similarity_threshold", 0.80)),
                temporal_decay=float(recurrence.get("temporal_decay", 0.01)),
                normalization_scale=float(recurrence.get("normalization_scale", 2.0)),
                wound_count=int(recurrence.get("wound_count", 2)),
                trauma_count=int(recurrence.get("trauma_count", 3)),
                occurrence_count_includes_current=bool(recurrence.get("occurrence_count_includes_current", True)),
            ),
            salience=SalienceConfig(
                valence=float(salience.get("valence", 0.30)),
                expectation_violation=float(salience.get("expectation_violation", 0.25)),
                identity_conflict=float(salience.get("identity_conflict", 0.25)),
                outcome_confidence=float(salience.get("outcome_confidence", 0.20)),
            ),
            repeat=EvidenceSignalConfig(
                positive_valence=float(repeat_weights.get("positive_valence", 1.20)),
                identity_term=float(repeat_weights.get("positive_identity_alignment", 0.60)),
                expectation_violation=float(repeat_weights.get("positive_expectation_violation", 0.80)),
                salience=float(repeat_weights.get("salience", 0.50)),
                bias=float(repeat.get("bias", 1.20)),
            ),
            avoidance=EvidenceSignalConfig(
                positive_valence=float(avoidance_weights.get("negative_valence", 1.10)),
                identity_term=float(avoidance_weights.get("identity_conflict", 0.70)),
                expectation_violation=float(avoidance_weights.get("negative_expectation_violation", 0.80)),
                salience=float(avoidance_weights.get("salience", 0.50)),
                recurrence=float(avoidance_weights.get("recurrence", 0.90)),
                bias=float(avoidance.get("bias", 1.40)),
            ),
            tail_gates=TailGateConfig(
                positive_threshold=float(gates.get("positive_threshold", 0.75)),
                negative_threshold=float(gates.get("negative_threshold", 0.70)),
                positive_sharpness=float(gates.get("positive_sharpness", 12.0)),
                negative_sharpness=float(gates.get("negative_sharpness", 12.0)),
            ),
            outcome_regions=OutcomeRegionConfig(
                success_repeat=float(success_weights.get("repeat", 0.55)),
                success_identity_pressure=float(success_weights.get("positive_identity_pressure", 0.30)),
                success_salience=float(success_weights.get("salience", 0.15)),
                injury_avoidance=float(injury_weights.get("avoidance", 0.45)),
                injury_identity_pressure=float(injury_weights.get("negative_identity_pressure", 0.25)),
                injury_salience=float(injury_weights.get("salience", 0.15)),
                injury_recurrence=float(injury_weights.get("recurrence", 0.15)),
                wound_injury_threshold=float(wound.get("injury_threshold", 0.50)),
                wound_identity_conflict_threshold=float(wound.get("identity_conflict_threshold", 0.30)),
                wound_recurrence_count=int(wound.get("recurrence_count", 2)),
                wound_extreme_consequence_threshold=float(wound.get("extreme_consequence_threshold", 0.70)),
                trauma_injury_threshold=float(trauma.get("injury_threshold", 0.72)),
                trauma_recurrence_count=int(trauma.get("recurrence_count", 3)),
                trauma_extreme_consequence_threshold=float(trauma.get("extreme_consequence_threshold", 0.90)),
            ),
            retention=RetentionConfig(
                identity_decay_lambda=float(retention.get("identity_decay_lambda", 0.01)),
                time_decay_lambda=float(retention.get("time_decay_lambda", 0.03)),
                maximum_floor=float(retention.get("maximum_floor", 0.95)),
                repeat=float(floor_weights.get("repeat", 0.30)),
                avoidance=float(floor_weights.get("avoidance", 0.25)),
                absolute_identity_pressure=float(floor_weights.get("absolute_identity_pressure", 0.35)),
                recurring_negative_tail=float(floor_weights.get("recurring_negative_tail", 0.45)),
                reward_ordinary_floor=float(reward_floor.get("ordinary", 0.0)),
                reward_success_floor=float(reward_floor.get("success", 0.25)),
                reward_failure_floor=float(reward_floor.get("failure", 0.25)),
                reward_wound_floor=float(reward_floor.get("wound", 0.25)),
                reward_trauma_floor=float(reward_floor.get("trauma", 0.34)),
            ),
            retrieval=RetrievalConfig(
                semantic_relevance_threshold=float(
                    retrieval.get("semantic_relevance_threshold", 0.55)
                ),
                candidate_top_k=int(retrieval.get("candidate_top_k", 50)),
                final_top_k=int(retrieval.get("final_top_k", 6)),
                maximum_results_per_trajectory=int(
                    retrieval.get("maximum_results_per_trajectory", 1)
                ),
                retention_weight=float(tie_break.get("retention_weight", 0.02)),
                salience_weight=float(tie_break.get("salience_weight", 0.02)),
                maximum_adjustment=float(tie_break.get("maximum_adjustment", 0.05)),
            ),
            tree_attention=TreeAttentionConfig(
                enabled=bool(tree_attention.get("enabled", True)),
                apply_after_semantic_eligibility=bool(
                    tree_attention.get("apply_after_semantic_eligibility", True)
                ),
                apply_after_trajectory_deduplication=bool(
                    tree_attention.get("apply_after_trajectory_deduplication", True)
                ),
                preserve_typed_edges=bool(
                    tree_attention.get("preserve_typed_edges", True)
                ),
                preserve_provenance_paths=bool(
                    tree_attention.get("preserve_provenance_paths", True)
                ),
                attention_weights_must_sum_to_one=bool(
                    tree_attention.get("attention_weights_must_sum_to_one", True)
                ),
                allow_cross_tree_attention=bool(
                    tree_attention.get("allow_cross_tree_attention", False)
                ),
                root_similarity_weight=float(
                    tree_attention.get("root_similarity_weight", 0.50)
                ),
                tree_similarity_weight=float(
                    tree_attention.get("tree_similarity_weight", 0.50)
                ),
                depth_bias_per_level=float(
                    tree_attention.get("depth_bias_per_level", 0.0)
                ),
            ),
            identity_transition=IdentityTransitionConfig(
                maximum_negative_fact_weight=float(
                    identity_transition.get("maximum_negative_fact_weight", 2.50)
                ),
                recurrence_weight_multiplier=float(
                    identity_transition.get("recurrence_weight_multiplier", 1.00)
                ),
            ),
            policy=PolicyConfig(
                persistence=float(policy.get("persistence", 0.95)),
            ),
            expectation=ExpectationConfig(
                base_learning_rate=float(expectation.get("base_learning_rate", 0.05)),
                salience_learning_rate=float(expectation.get("salience_learning_rate", 0.05)),
                recurrence_learning_rate=float(expectation.get("recurrence_learning_rate", 0.10)),
                identity_pressure_learning_rate=float(expectation.get("identity_pressure_learning_rate", 0.10)),
                maximum_learning_rate=float(expectation.get("maximum_learning_rate", 0.30)),
            ),
        )
