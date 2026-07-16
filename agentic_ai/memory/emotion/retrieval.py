"""Relevance-first retrieval with trajectory deduplication."""

from __future__ import annotations

import math
from dataclasses import dataclass

from agentic_ai.memory.emotion.appraisal import cosine
from agentic_ai.memory.emotion.config import EmotionConfig
from agentic_ai.memory.emotion.models import TransitionResult, Vector
from agentic_ai.memory.emotion.retention import (
    RetentionEngine,
    RetentionMode,
)
from agentic_ai.memory.emotion.topology import TimestepNode


@dataclass(frozen=True)
class MemoryCandidate:
    memory_id: str
    trajectory_id: str
    root_embedding: Vector
    transition: TransitionResult
    age: float = 0.0
    active: bool = True
    accessible: bool = True
    expired: bool = False

    @classmethod
    def from_timestep(
        cls,
        node: TimestepNode,
        current_timestep: int,
    ) -> "MemoryCandidate":
        return cls(
            memory_id=node.event.event_id,
            trajectory_id=node.event.trajectory_id or node.event.event_id,
            root_embedding=node.event.embedding,
            transition=node.transition,
            age=max(0, current_timestep - node.timestep),
            active=node.active,
            accessible=True,
            expired=False,
        )


@dataclass(frozen=True)
class EligibleCandidate:
    candidate: MemoryCandidate
    semantic_similarity: float


@dataclass(frozen=True)
class RetrievalSelection:
    memory_id: str
    trajectory_id: str
    semantic_similarity: float
    retention: float
    salience: float
    bounded_adjustment: float
    final_score: float


@dataclass(frozen=True)
class RetrievalTrace:
    eligible_memory_ids: tuple[str, ...]
    deduplicated_memory_ids: tuple[str, ...]
    rejected: tuple[tuple[str, str], ...]
    selections: tuple[RetrievalSelection, ...]


class RelevanceGate:
    """Create the eligible set using access state and semantics only."""

    def __init__(self, config: EmotionConfig | None = None) -> None:
        self.config = config or EmotionConfig()

    def apply(
        self,
        query: Vector,
        candidates: tuple[MemoryCandidate, ...],
    ) -> tuple[tuple[EligibleCandidate, ...], tuple[tuple[str, str], ...]]:
        eligible = []
        rejected = []
        threshold = self.config.retrieval.semantic_relevance_threshold

        for candidate in candidates:
            if not candidate.active:
                rejected.append((candidate.memory_id, "inactive"))
                continue
            if not candidate.accessible:
                rejected.append((candidate.memory_id, "inaccessible"))
                continue
            if candidate.expired:
                rejected.append((candidate.memory_id, "expired"))
                continue
            similarity = cosine(query, candidate.root_embedding)
            if similarity < threshold:
                rejected.append((candidate.memory_id, "below-relevance-threshold"))
                continue
            eligible.append(EligibleCandidate(candidate, similarity))

        eligible.sort(key=lambda item: (-item.semantic_similarity, item.candidate.memory_id))
        limit = self.config.retrieval.candidate_top_k
        overflow = eligible[limit:]
        for item in overflow:
            rejected.append((item.candidate.memory_id, "candidate-limit"))
        return tuple(eligible[:limit]), tuple(rejected)


class TrajectoryDeduplicator:
    """Limit how many memories from one trajectory consume final slots."""

    def __init__(self, config: EmotionConfig | None = None) -> None:
        self.config = config or EmotionConfig()

    def apply(
        self,
        candidates: tuple[EligibleCandidate, ...],
    ) -> tuple[EligibleCandidate, ...]:
        counts: dict[str, int] = {}
        selected = []
        maximum = self.config.retrieval.maximum_results_per_trajectory
        for candidate in candidates:
            trajectory = candidate.candidate.trajectory_id
            count = counts.get(trajectory, 0)
            if count >= maximum:
                continue
            selected.append(candidate)
            counts[trajectory] = count + 1
        return tuple(selected)


class RelevanceFirstRetriever:
    """Retrieve with relevance gating before retention/salience tie-breaking."""

    def __init__(self, config: EmotionConfig | None = None) -> None:
        self.config = config or EmotionConfig()
        self.relevance_gate = RelevanceGate(self.config)
        self.deduplicator = TrajectoryDeduplicator(self.config)
        self.retention_engine = RetentionEngine(self.config)

    def retrieve(
        self,
        query: Vector,
        candidates: tuple[MemoryCandidate, ...],
        retention_mode: RetentionMode | str = RetentionMode.EMOTION,
    ) -> RetrievalTrace:
        eligible, rejected = self.relevance_gate.apply(query, candidates)
        deduplicated = self.deduplicator.apply(eligible)
        selections = []
        config = self.config.retrieval

        for item in deduplicated:
            candidate = item.candidate
            retention = self.retention_engine.evaluate(
                candidate.transition,
                candidate.age,
                retention_mode,
                active=candidate.active,
            ).retention
            salience = candidate.transition.appraisal.salience
            raw_adjustment = (
                config.retention_weight * math.log(retention + 1.0e-12)
                + config.salience_weight * salience
            )
            adjustment = min(
                config.maximum_adjustment,
                max(-config.maximum_adjustment, raw_adjustment),
            )
            selections.append(
                RetrievalSelection(
                    memory_id=candidate.memory_id,
                    trajectory_id=candidate.trajectory_id,
                    semantic_similarity=item.semantic_similarity,
                    retention=retention,
                    salience=salience,
                    bounded_adjustment=adjustment,
                    final_score=item.semantic_similarity + adjustment,
                )
            )

        selections.sort(key=lambda item: (-item.final_score, item.memory_id))
        selections = selections[: config.final_top_k]
        return RetrievalTrace(
            eligible_memory_ids=tuple(item.candidate.memory_id for item in eligible),
            deduplicated_memory_ids=tuple(
                item.candidate.memory_id for item in deduplicated
            ),
            rejected=rejected,
            selections=tuple(selections),
        )
