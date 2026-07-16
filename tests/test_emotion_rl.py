from __future__ import annotations

from dataclasses import fields

import pytest

from agentic_ai.memory.emotion import (
    EmotionMemorySystem,
    EncodedRLState,
    EvidenceDirection,
    EvidenceItem,
    Event,
    IdentityState,
    LinearQLearner,
    OutcomeEvidence,
    RLStateEncoder,
)


def evidence(direction: EvidenceDirection) -> OutcomeEvidence:
    return OutcomeEvidence(
        items=(EvidenceItem("outcome", direction, 1.0, 1.0, "environment"),)
    )


def memory_system(direction: EvidenceDirection) -> EmotionMemorySystem:
    system = EmotionMemorySystem(
        IdentityState(supplied_vector=(1.0, 0.0, 0.0, 0.0))
    )
    system.ingest(
        Event(
            event_id="episode",
            timestep=1,
            context_id="tools",
            action_id="selected-action",
            embedding=(1.0, 0.0, 0.0, 0.0),
            expected_valence=0.0,
            trajectory_id="tool-trajectory",
        ),
        evidence(direction),
    )
    return system


def test_rl_state_contains_identity_memory_expectation_and_policy():
    system = memory_system(EvidenceDirection.SUCCESS)
    state = RLStateEncoder().encode(
        system=system,
        environment_state=(0.2, -0.1),
        query=(1.0, 0.0, 0.0, 0.0),
        current_timestep=2,
        context_id="tools",
        action_id="selected-action",
        expected_valence=0.4,
    )

    assert len(state.vector) == 2 + 4 + 1 + 4 + 3
    assert state.environment_state == (0.2, -0.1)
    assert len(state.identity_state) == 4
    assert len(state.memory_state) == 4
    assert state.expected_valence == 0.4
    assert state.policy_repeat > 0.0
    assert state.policy_memory_q > 0.0
    assert state.retrieval_trace.selections


def test_failure_memory_creates_negative_policy_feature():
    system = memory_system(EvidenceDirection.FAILURE)
    state = RLStateEncoder().encode(
        system,
        environment_state=(0.0, 1.0),
        query=(1.0, 0.0, 0.0, 0.0),
        current_timestep=2,
        context_id="tools",
        action_id="selected-action",
        expected_valence=0.0,
    )

    assert state.policy_avoidance > state.policy_repeat
    assert state.policy_memory_q < 0.0


def test_linear_q_update_uses_environment_reward():
    learner = LinearQLearner(input_dimension=2, number_of_actions=2)
    update = learner.update(
        state=(1.0, 2.0),
        action=1,
        environment_reward=1.5,
        next_state=(0.5, 0.5),
        terminal=True,
    )

    assert update.environment_reward == 1.5
    assert update.prediction == 0.0
    assert update.target == 1.5
    assert update.td_error == 1.5
    assert learner.weights[1].tolist() == pytest.approx([0.075, 0.15])
    assert learner.bias[1] == pytest.approx(0.075)
    assert learner.q_values((1.0, 2.0))[1] > 0.0


def test_rl_contract_contains_no_emotion_label_or_label_reward():
    state_fields = {field.name for field in fields(EncodedRLState)}
    assert "operational_label" not in state_fields
    assert "emotion_label" not in state_fields

    system = memory_system(EvidenceDirection.SUCCESS)
    state = RLStateEncoder().encode(
        system,
        environment_state=(1.0,),
        query=(1.0, 0.0, 0.0, 0.0),
        current_timestep=2,
        context_id="tools",
        action_id="selected-action",
        expected_valence=0.0,
    )
    learner = LinearQLearner(len(state.vector), 1)
    update = learner.update(state, 0, -2.0, state, terminal=True)
    assert update.target == -2.0
    assert update.environment_reward == -2.0
