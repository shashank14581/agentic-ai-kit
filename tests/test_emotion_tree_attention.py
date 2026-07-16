from __future__ import annotations

import pytest

from agentic_ai.memory.emotion import (
    AppraisalEngine,
    Event,
    EventTree,
    EventTreeNode,
    IdentityState,
    MemoryCandidate,
    OutcomeEvidence,
    RelevanceFirstRetriever,
    TreeAttentionEngine,
)


def deterministic_tree() -> EventTree:
    tree = EventTree(
        EventTreeNode(
            "root",
            "event-root",
            None,
            (0.8, 0.2, 0.0, 0.0),
            "event",
        )
    )
    tree.add(EventTreeNode("context", "context", "root", (0.0, 1.0, 0.0, 0.0), "context"))
    tree.add(EventTreeNode("action", "action", "root", (0.6, 0.4, 0.0, 0.0), "action"))
    tree.add(
        EventTreeNode(
            "failure-evidence",
            "failure-evidence",
            "root",
            (1.0, 0.0, 0.0, 0.0),
            "environment",
        )
    )
    tree.add(
        EventTreeNode(
            "snapshot",
            "snapshot",
            "failure-evidence",
            (0.9, 0.1, 0.0, 0.0),
            "transition",
        )
    )
    return tree


def ordinary_transition(event_id: str):
    event = Event(event_id, 1, "context", "action", (1.0, 0.0, 0.0, 0.0), 0.0)
    return AppraisalEngine().evaluate(
        IdentityState(supplied_vector=(1.0, 0.0, 0.0, 0.0)),
        event,
        OutcomeEvidence(),
    )


def test_tree_attention_matches_registered_deterministic_fixture():
    result = TreeAttentionEngine().attend(
        (1.0, 0.0, 0.0, 0.0),
        deterministic_tree(),
    )
    weights = {item.node_id: item.weight for item in result.node_attention}

    assert result.highest_attention_node == "failure-evidence"
    assert result.weight_sum == pytest.approx(1.0, abs=1.0e-9)
    assert weights == pytest.approx(
        {
            "root": 0.2113,
            "context": 0.1417,
            "action": 0.1912,
            "failure-evidence": 0.2336,
            "snapshot": 0.2222,
        },
        abs=1.0e-4,
    )
    snapshot = next(item for item in result.node_attention if item.node_id == "snapshot")
    assert snapshot.path_to_root == ("root", "failure-evidence", "snapshot")
    assert all(item.provenance for item in result.node_attention)


def test_tree_attention_runs_only_after_relevance_and_deduplication():
    valid_tree = deterministic_tree()
    wrong_dimension_tree = EventTree(
        EventTreeNode("wrong-root", "event-root", None, (1.0, 0.0), "wrong")
    )
    transition = ordinary_transition("memory")
    candidates = (
        MemoryCandidate(
            "eligible",
            "trajectory-a",
            (1.0, 0.0, 0.0, 0.0),
            transition,
            tree=valid_tree,
        ),
        MemoryCandidate(
            "duplicate",
            "trajectory-a",
            (0.99, 0.1, 0.0, 0.0),
            transition,
            tree=wrong_dimension_tree,
        ),
        MemoryCandidate(
            "irrelevant",
            "trajectory-b",
            (0.0, 1.0, 0.0, 0.0),
            transition,
            tree=wrong_dimension_tree,
        ),
    )

    trace = RelevanceFirstRetriever().retrieve(
        (1.0, 0.0, 0.0, 0.0),
        candidates,
    )

    assert trace.eligible_memory_ids == ("eligible", "duplicate")
    assert trace.deduplicated_memory_ids == ("eligible",)
    assert len(trace.selections) == 1
    assert trace.selections[0].tree_attention is not None
    assert ("irrelevant", "below-relevance-threshold") in trace.rejected


def test_tree_content_score_precedes_bounded_salience_adjustment():
    transition = ordinary_transition("memory")
    candidate = MemoryCandidate(
        "memory",
        "trajectory",
        (1.0, 0.0, 0.0, 0.0),
        transition,
        tree=deterministic_tree(),
        age=1000,
    )
    selection = RelevanceFirstRetriever().retrieve(
        (1.0, 0.0, 0.0, 0.0),
        (candidate,),
    ).selections[0]

    assert selection.tree_attention is not None
    assert selection.content_similarity == pytest.approx(
        0.5 * selection.semantic_similarity + 0.5 * selection.tree_similarity
    )
    assert abs(selection.bounded_adjustment) <= 0.05
    assert selection.final_score == pytest.approx(
        selection.content_similarity + selection.bounded_adjustment
    )
