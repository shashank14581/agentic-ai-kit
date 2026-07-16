"""Linked timestep memory whose entries root typed event trees."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from agentic_ai.memory.emotion.models import (
    Event,
    HistoricalOutcome,
    OutcomeEvidence,
    OutcomeStatus,
    TransitionResult,
    Vector,
)


@dataclass(frozen=True)
class EventTreeNode:
    node_id: str
    node_type: str
    parent_id: str | None
    embedding: Vector | None
    provenance: str
    payload: tuple[tuple[str, Any], ...] = ()

    def as_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.payload))


class EventTree:
    """Typed tree containing raw evidence and a derived transition snapshot."""

    def __init__(self, root: EventTreeNode) -> None:
        if root.parent_id is not None:
            raise ValueError("Event tree root cannot have a parent.")
        self.root_id = root.node_id
        self._nodes: dict[str, EventTreeNode] = {root.node_id: root}

    @property
    def nodes(self) -> tuple[EventTreeNode, ...]:
        return tuple(self._nodes.values())

    def add(self, node: EventTreeNode) -> None:
        if node.node_id in self._nodes:
            raise ValueError(f"Duplicate event-tree node: {node.node_id}")
        if node.parent_id not in self._nodes:
            raise ValueError(f"Unknown parent node: {node.parent_id}")
        self._nodes[node.node_id] = node

    def get(self, node_id: str) -> EventTreeNode:
        return self._nodes[node_id]

    def children(self, node_id: str) -> tuple[EventTreeNode, ...]:
        if node_id not in self._nodes:
            raise KeyError(node_id)
        return tuple(node for node in self._nodes.values() if node.parent_id == node_id)

    @classmethod
    def from_transition(
        cls,
        event: Event,
        evidence: OutcomeEvidence,
        transition: TransitionResult,
    ) -> "EventTree":
        if event.event_id != transition.event_id:
            raise ValueError("Event and transition identifiers must match.")

        root_id = f"{event.event_id}:root"
        tree = cls(
            EventTreeNode(
                node_id=root_id,
                node_type="event-root",
                parent_id=None,
                embedding=event.embedding,
                provenance=event.provenance,
                payload=(
                    ("event_id", event.event_id),
                    ("timestep", event.timestep),
                    ("trajectory_id", event.trajectory_id or event.event_id),
                ),
            )
        )
        tree.add(
            EventTreeNode(
                node_id=f"{event.event_id}:context",
                node_type="context",
                parent_id=root_id,
                embedding=event.recurrence_vector,
                provenance=event.provenance,
                payload=(("context_id", event.context_id),),
            )
        )
        tree.add(
            EventTreeNode(
                node_id=f"{event.event_id}:action",
                node_type="action",
                parent_id=root_id,
                embedding=event.embedding,
                provenance=event.provenance,
                payload=(("action_id", event.action_id),),
            )
        )
        tree.add(
            EventTreeNode(
                node_id=f"{event.event_id}:expectation",
                node_type="expectation",
                parent_id=root_id,
                embedding=None,
                provenance=event.provenance,
                payload=(("expected_valence", event.expected_valence),),
            )
        )

        evidence_root = f"{event.event_id}:evidence"
        tree.add(
            EventTreeNode(
                node_id=evidence_root,
                node_type="evidence",
                parent_id=root_id,
                embedding=None,
                provenance=event.provenance,
                payload=(("evidence_quality", transition.appraisal.evidence_quality),),
            )
        )
        for index, item in enumerate(evidence.items):
            tree.add(
                EventTreeNode(
                    node_id=f"{event.event_id}:evidence:{index}",
                    node_type=(
                        "success-evidence" if int(item.direction) == 1 else "failure-evidence"
                    ),
                    parent_id=evidence_root,
                    embedding=None,
                    provenance=item.provenance,
                    payload=(
                        ("evidence_id", item.evidence_id),
                        ("magnitude", item.magnitude),
                        ("reliability", item.reliability),
                        ("active", item.active),
                    ),
                )
            )

        snapshot_id = f"{event.event_id}:snapshot"
        tree.add(
            EventTreeNode(
                node_id=snapshot_id,
                node_type="snapshot",
                parent_id=root_id,
                embedding=event.embedding,
                provenance=event.provenance,
                payload=(
                    ("appraisal", transition.appraisal.vector),
                    ("state", transition.state.vector),
                    ("label", transition.state.operational_label.value),
                    ("configuration_version", transition.configuration_version),
                ),
            )
        )
        return tree


@dataclass(frozen=True)
class TimestepNode:
    timestep: int
    event: Event
    evidence: OutcomeEvidence
    transition: TransitionResult
    tree: EventTree
    previous_timestep: int | None = None
    next_timestep: int | None = None
    active: bool = True
    invalidation_reason: str | None = None


class EpisodicMemoryTimeline:
    """In-memory linked timeline with one event tree per timestep."""

    def __init__(self) -> None:
        self._nodes: dict[int, TimestepNode] = {}
        self._event_index: dict[str, int] = {}
        self.head_timestep: int | None = None
        self.tail_timestep: int | None = None

    def __len__(self) -> int:
        return sum(node.active for node in self._nodes.values())

    def append(
        self,
        event: Event,
        evidence: OutcomeEvidence,
        transition: TransitionResult,
    ) -> TimestepNode:
        if event.event_id in self._event_index:
            raise ValueError(f"Duplicate event identifier: {event.event_id}")
        if self.tail_timestep is not None and event.timestep <= self.tail_timestep:
            raise ValueError("Timesteps must be appended in strictly increasing order.")

        tree = EventTree.from_transition(event, evidence, transition)
        previous = self.tail_timestep
        node = TimestepNode(
            timestep=event.timestep,
            event=event,
            evidence=evidence,
            transition=transition,
            tree=tree,
            previous_timestep=previous,
        )
        self._nodes[event.timestep] = node
        self._event_index[event.event_id] = event.timestep

        if previous is not None:
            self._nodes[previous] = replace(
                self._nodes[previous],
                next_timestep=event.timestep,
            )
        else:
            self.head_timestep = event.timestep
        self.tail_timestep = event.timestep
        return node

    def get(self, timestep: int) -> TimestepNode:
        return self._nodes[timestep]

    def get_event(self, event_id: str) -> TimestepNode:
        return self._nodes[self._event_index[event_id]]

    def invalidate(self, event_id: str, reason: str) -> None:
        timestep = self._event_index[event_id]
        node = self._nodes[timestep]
        self._nodes[timestep] = replace(
            node,
            active=False,
            invalidation_reason=reason,
        )

    def iter_nodes(self, include_inactive: bool = False) -> Iterator[TimestepNode]:
        timestep = self.head_timestep
        while timestep is not None:
            node = self._nodes[timestep]
            if include_inactive or node.active:
                yield node
            timestep = node.next_timestep

    def history_before(self, timestep: int) -> tuple[HistoricalOutcome, ...]:
        history = []
        for node in self.iter_nodes():
            if node.timestep >= timestep:
                break
            history.append(
                HistoricalOutcome(
                    event_id=node.event.event_id,
                    timestep=node.event.timestep,
                    context_action_embedding=node.event.recurrence_vector,
                    status=node.transition.appraisal.outcome_status,
                    failure_confidence=node.transition.appraisal.failure_confidence,
                    evidence_quality=node.transition.appraisal.evidence_quality,
                )
            )
        return tuple(history)
