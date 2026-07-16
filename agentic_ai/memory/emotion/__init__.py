"""Deterministic emotion-aware episodic memory primitives."""

from agentic_ai.memory.emotion.appraisal import AppraisalEngine, build_identity_vector
from agentic_ai.memory.emotion.config import EmotionConfig
from agentic_ai.memory.emotion.models import (
    AffectiveState,
    Appraisal,
    EvidenceDirection,
    EvidenceItem,
    Event,
    HistoricalOutcome,
    IdentityFact,
    IdentityState,
    OutcomeEvidence,
    OutcomeLabel,
    OutcomeStatus,
    TransitionResult,
)

__all__ = [
    "AffectiveState",
    "Appraisal",
    "AppraisalEngine",
    "EmotionConfig",
    "EvidenceDirection",
    "EvidenceItem",
    "Event",
    "HistoricalOutcome",
    "IdentityFact",
    "IdentityState",
    "OutcomeEvidence",
    "OutcomeLabel",
    "OutcomeStatus",
    "TransitionResult",
    "build_identity_vector",
]
