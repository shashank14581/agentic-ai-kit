"""Deterministic emotion-aware episodic memory primitives."""

from agentic_ai.memory.emotion.appraisal import AppraisalEngine, build_identity_vector
from agentic_ai.memory.emotion.config import EmotionConfig
from agentic_ai.memory.emotion.ledgers import (
    IdentityLedger,
    IdentityPolarity,
    LedgerOperation,
    PolicyLedger,
    PolicyState,
)
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
from agentic_ai.memory.emotion.retention import (
    RetentionEngine,
    RetentionMode,
    RetentionResult,
)
from agentic_ai.memory.emotion.retrieval import (
    EligibleCandidate,
    MemoryCandidate,
    RelevanceFirstRetriever,
    RelevanceGate,
    RetrievalSelection,
    RetrievalTrace,
    TrajectoryDeduplicator,
)
from agentic_ai.memory.emotion.topology import (
    EpisodicMemoryTimeline,
    EventTree,
    EventTreeNode,
    TimestepNode,
)
from agentic_ai.memory.emotion.tree_attention import (
    NodeAttention,
    TreeAttentionEngine,
    TreeAttentionResult,
)
from agentic_ai.memory.emotion.system import (
    AuditEntry,
    EmotionMemorySystem,
    EpisodeJournalEntry,
    EpisodeOperation,
    SystemSnapshot,
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
    "IdentityLedger",
    "IdentityPolarity",
    "IdentityState",
    "LedgerOperation",
    "OutcomeEvidence",
    "OutcomeLabel",
    "OutcomeStatus",
    "PolicyLedger",
    "PolicyState",
    "RetentionEngine",
    "RetentionMode",
    "RetentionResult",
    "EligibleCandidate",
    "MemoryCandidate",
    "RelevanceFirstRetriever",
    "RelevanceGate",
    "RetrievalSelection",
    "RetrievalTrace",
    "TrajectoryDeduplicator",
    "EpisodicMemoryTimeline",
    "EventTree",
    "EventTreeNode",
    "TimestepNode",
    "NodeAttention",
    "TreeAttentionEngine",
    "TreeAttentionResult",
    "AuditEntry",
    "EmotionMemorySystem",
    "EpisodeJournalEntry",
    "EpisodeOperation",
    "SystemSnapshot",
    "TransitionResult",
    "build_identity_vector",
]
