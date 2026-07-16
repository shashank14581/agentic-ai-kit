"""Correction-safe orchestration for emotion-aware episodic memory."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agentic_ai.memory.emotion.appraisal import AppraisalEngine
from agentic_ai.memory.emotion.config import EmotionConfig
from agentic_ai.memory.emotion.ledgers import IdentityLedger, PolicyLedger, PolicyState
from agentic_ai.memory.emotion.models import (
    Event,
    IdentityState,
    OutcomeEvidence,
    TransitionResult,
    Vector,
)
from agentic_ai.memory.emotion.retention import RetentionMode
from agentic_ai.memory.emotion.retrieval import (
    MemoryCandidate,
    RelevanceFirstRetriever,
    RetrievalTrace,
)
from agentic_ai.memory.emotion.topology import EpisodicMemoryTimeline


class EpisodeOperation(str, Enum):
    INGEST = "ingest"
    CORRECT = "correct"
    DELETE = "delete"
    EXPIRE = "expire"


@dataclass(frozen=True)
class EpisodeJournalEntry:
    sequence: int
    operation: EpisodeOperation
    event_id: str
    version: int
    event: Event | None
    evidence: OutcomeEvidence | None
    reason: str | None
    configuration_version: str


@dataclass(frozen=True)
class AuditEntry:
    sequence: int
    operation: EpisodeOperation
    event_id: str
    version: int
    reason: str | None
    active_event_count: int
    positive_fact_count: int
    negative_fact_count: int
    policy_evidence_count: int
    resulting_label: str | None
    configuration_version: str


@dataclass(frozen=True)
class SystemSnapshot:
    active_event_ids: tuple[str, ...]
    identity_vector: Vector
    positive_fact_count: int
    negative_fact_count: int
    policy_evidence_count: int
    labels: tuple[tuple[str, str], ...]


class EmotionMemorySystem:
    """Compose appraisal, ledgers, timeline, replay, and retrieval.

    Raw episode operations are append-only. Every correction, deletion, or
    expiry rebuilds derived state from the active journal, so downstream
    appraisals cannot retain effects from invalid evidence.
    """

    def __init__(
        self,
        initial_identity: IdentityState,
        config: EmotionConfig | None = None,
    ) -> None:
        self.config = config or EmotionConfig()
        self.initial_identity = initial_identity
        self.appraisal_engine = AppraisalEngine(self.config)
        self.retriever = RelevanceFirstRetriever(self.config)
        self._journal: list[EpisodeJournalEntry] = []
        self._audit: list[AuditEntry] = []
        self.identity_ledger = IdentityLedger(initial_identity, self.config)
        self.policy_ledger = PolicyLedger(self.config)
        self.timeline = EpisodicMemoryTimeline()

    @property
    def journal(self) -> tuple[EpisodeJournalEntry, ...]:
        return tuple(self._journal)

    @property
    def audit_log(self) -> tuple[AuditEntry, ...]:
        return tuple(self._audit)

    def ingest(self, event: Event, evidence: OutcomeEvidence) -> TransitionResult:
        if self._has_ever_existed(event.event_id):
            raise ValueError(f"Event already exists in journal: {event.event_id}")
        if any(
            entry.event is not None and entry.event.timestep == event.timestep
            for entry in self._active_entries().values()
        ):
            raise ValueError(f"An active event already uses timestep {event.timestep}.")

        entry = self._append_journal(
            EpisodeOperation.INGEST,
            event.event_id,
            event,
            evidence,
            reason=None,
        )
        self._rebuild()
        self._append_audit(entry)
        return self.timeline.get_event(event.event_id).transition

    def correct(
        self,
        event_id: str,
        evidence: OutcomeEvidence,
        reason: str,
        event: Event | None = None,
    ) -> TransitionResult:
        current = self._require_active(event_id)
        corrected_event = event or current.event
        if corrected_event is None:
            raise RuntimeError("Active journal entry has no event.")
        if corrected_event.event_id != event_id:
            raise ValueError("A correction cannot change the event identifier.")
        if current.event is not None and corrected_event.timestep != current.event.timestep:
            raise ValueError("A correction cannot change the original timestep.")

        entry = self._append_journal(
            EpisodeOperation.CORRECT,
            event_id,
            corrected_event,
            evidence,
            reason,
        )
        self._rebuild()
        self._append_audit(entry)
        return self.timeline.get_event(event_id).transition

    def delete(self, event_id: str, reason: str) -> None:
        self._require_active(event_id)
        entry = self._append_journal(
            EpisodeOperation.DELETE,
            event_id,
            event=None,
            evidence=None,
            reason=reason,
        )
        self._rebuild()
        self._append_audit(entry)

    def expire(self, event_id: str, reason: str) -> None:
        self._require_active(event_id)
        entry = self._append_journal(
            EpisodeOperation.EXPIRE,
            event_id,
            event=None,
            evidence=None,
            reason=reason,
        )
        self._rebuild()
        self._append_audit(entry)

    def replay(self) -> SystemSnapshot:
        """Rebuild all derived state from the active append-only journal."""
        self._rebuild()
        return self.snapshot()

    def policy_state(self, context_id: str, action_id: str) -> PolicyState:
        return self.policy_ledger.state(context_id, action_id)

    def retrieve(
        self,
        query: Vector,
        current_timestep: int,
        retention_mode: RetentionMode | str = RetentionMode.EMOTION,
    ) -> RetrievalTrace:
        candidates = tuple(
            MemoryCandidate.from_timestep(node, current_timestep)
            for node in self.timeline.iter_nodes()
        )
        return self.retriever.retrieve(query, candidates, retention_mode)

    def snapshot(self) -> SystemSnapshot:
        identity_state = self.identity_ledger.state()
        labels = tuple(
            (
                node.event.event_id,
                node.transition.state.operational_label.value,
            )
            for node in self.timeline.iter_nodes()
        )
        return SystemSnapshot(
            active_event_ids=tuple(event_id for event_id, _ in labels),
            identity_vector=self.identity_ledger.vector(),
            positive_fact_count=len(identity_state.positive_facts),
            negative_fact_count=len(identity_state.negative_facts),
            policy_evidence_count=sum(
                event.operation.value == "add" for event in self.policy_ledger.events
            ),
            labels=labels,
        )

    def _append_journal(
        self,
        operation: EpisodeOperation,
        event_id: str,
        event: Event | None,
        evidence: OutcomeEvidence | None,
        reason: str | None,
    ) -> EpisodeJournalEntry:
        version = 1 + sum(entry.event_id == event_id for entry in self._journal)
        entry = EpisodeJournalEntry(
            sequence=len(self._journal),
            operation=operation,
            event_id=event_id,
            version=version,
            event=event,
            evidence=evidence,
            reason=reason,
            configuration_version=self.config.architecture_version,
        )
        self._journal.append(entry)
        return entry

    def _active_entries(self) -> dict[str, EpisodeJournalEntry]:
        active: dict[str, EpisodeJournalEntry] = {}
        for entry in self._journal:
            if entry.operation in (EpisodeOperation.INGEST, EpisodeOperation.CORRECT):
                active[entry.event_id] = entry
            elif entry.operation in (EpisodeOperation.DELETE, EpisodeOperation.EXPIRE):
                active.pop(entry.event_id, None)
        return active

    def _require_active(self, event_id: str) -> EpisodeJournalEntry:
        try:
            return self._active_entries()[event_id]
        except KeyError as exc:
            raise KeyError(f"No active event exists for {event_id!r}.") from exc

    def _has_ever_existed(self, event_id: str) -> bool:
        return any(entry.event_id == event_id for entry in self._journal)

    def _rebuild(self) -> None:
        self.identity_ledger = IdentityLedger(self.initial_identity, self.config)
        self.policy_ledger = PolicyLedger(self.config)
        self.timeline = EpisodicMemoryTimeline()

        active_entries = sorted(
            self._active_entries().values(),
            key=lambda entry: (
                entry.event.timestep if entry.event is not None else -1,
                entry.sequence,
            ),
        )
        for entry in active_entries:
            if entry.event is None or entry.evidence is None:
                continue
            transition = self.appraisal_engine.evaluate(
                self.identity_ledger.state(),
                entry.event,
                entry.evidence,
                self.timeline.history_before(entry.event.timestep),
            )
            self.timeline.append(entry.event, entry.evidence, transition)
            self.identity_ledger.apply(entry.event, transition)
            self.policy_ledger.apply(entry.event, transition)

    def _append_audit(self, journal_entry: EpisodeJournalEntry) -> None:
        identity_state = self.identity_ledger.state()
        resulting_label = None
        active = self._active_entries()
        if journal_entry.event_id in active:
            resulting_label = self.timeline.get_event(
                journal_entry.event_id
            ).transition.state.operational_label.value
        self._audit.append(
            AuditEntry(
                sequence=len(self._audit),
                operation=journal_entry.operation,
                event_id=journal_entry.event_id,
                version=journal_entry.version,
                reason=journal_entry.reason,
                active_event_count=len(self.timeline),
                positive_fact_count=len(identity_state.positive_facts),
                negative_fact_count=len(identity_state.negative_facts),
                policy_evidence_count=sum(
                    event.operation.value == "add"
                    for event in self.policy_ledger.events
                ),
                resulting_label=resulting_label,
                configuration_version=self.config.architecture_version,
            )
        )
