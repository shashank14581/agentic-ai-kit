"""End-to-end emotion-aware episodic memory demo.

This example is deterministic and requires no LLM or API key.
"""

from __future__ import annotations

from agentic_ai.memory.emotion import (
    EmotionMemorySystem,
    EvidenceDirection,
    EvidenceItem,
    Event,
    IdentityState,
    LinearQLearner,
    OutcomeEvidence,
    RLStateEncoder,
)


def confirmed_failure(event_id: str) -> OutcomeEvidence:
    return OutcomeEvidence(
        items=(
            EvidenceItem(
                evidence_id=f"{event_id}:failure",
                direction=EvidenceDirection.FAILURE,
                magnitude=1.0,
                reliability=1.0,
                provenance="environment",
            ),
        )
    )


def run_demo() -> dict[str, object]:
    memory = EmotionMemorySystem(
        initial_identity=IdentityState(
            supplied_vector=(-0.29, 0.9570266, 0.0, 0.0)
        )
    )

    for timestep in (1, 2, 3):
        event_id = f"planning-failure-{timestep}"
        memory.ingest(
            Event(
                event_id=event_id,
                timestep=timestep,
                context_id="planning",
                action_id="repeat-invalid-plan",
                embedding=(1.0, 0.0, 0.0, 0.0),
                expected_valence=0.0,
                trajectory_id="planning-trajectory",
            ),
            confirmed_failure(event_id),
        )

    labels_before = dict(memory.snapshot().labels)

    retrieval = memory.retrieve(
        query=(1.0, 0.0, 0.0, 0.0),
        current_timestep=4,
    )
    selected = retrieval.selections[0]

    rl_state = RLStateEncoder().encode(
        system=memory,
        environment_state=(0.25, -0.10),
        query=(1.0, 0.0, 0.0, 0.0),
        current_timestep=4,
        context_id="planning",
        action_id="repeat-invalid-plan",
        expected_valence=-0.40,
    )
    learner = LinearQLearner(
        input_dimension=len(rl_state.vector),
        number_of_actions=2,
    )
    td_update = learner.update(
        state=rl_state,
        action=0,
        environment_reward=-1.0,
        next_state=rl_state,
        terminal=True,
    )

    memory.delete(
        "planning-failure-2",
        reason="The second failure was invalidated by corrected evidence.",
    )
    labels_after = dict(memory.snapshot().labels)

    return {
        "labels_before": labels_before,
        "labels_after": labels_after,
        "retrieved_memory": selected.memory_id,
        "highest_attention_node": (
            selected.tree_attention.highest_attention_node
            if selected.tree_attention is not None
            else None
        ),
        "policy_memory_q": rl_state.policy_memory_q,
        "td_target": td_update.target,
        "active_event_ids": memory.snapshot().active_event_ids,
    }


def main() -> None:
    result = run_demo()

    print("Labels before correction:")
    for event_id, label in result["labels_before"].items():
        print(f"  {event_id}: {label}")

    print("\nTree-attention retrieval:")
    print(f"  retrieved memory: {result['retrieved_memory']}")
    print(f"  highest-attention node: {result['highest_attention_node']}")

    print("\nRL state:")
    print(f"  signed policy evidence: {result['policy_memory_q']:.4f}")
    print(f"  environment-reward TD target: {result['td_target']:.4f}")

    print("\nLabels after deleting the second failure and replaying:")
    for event_id, label in result["labels_after"].items():
        print(f"  {event_id}: {label}")


if __name__ == "__main__":
    main()
