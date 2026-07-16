"""Memory-augmented reinforcement learning without label-derived rewards."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from agentic_ai.memory.emotion.config import EmotionConfig
from agentic_ai.memory.emotion.models import Vector
from agentic_ai.memory.emotion.retention import RetentionMode
from agentic_ai.memory.emotion.retrieval import RetrievalTrace
from agentic_ai.memory.emotion.system import EmotionMemorySystem


@dataclass(frozen=True)
class EncodedRLState:
    """Traceable policy input; intentionally contains no outcome label."""

    vector: Vector
    environment_state: Vector
    identity_state: Vector
    expected_valence: float
    memory_state: Vector
    policy_repeat: float
    policy_avoidance: float
    policy_memory_q: float
    retrieval_trace: RetrievalTrace


@dataclass(frozen=True)
class TDUpdate:
    action: int
    environment_reward: float
    prediction: float
    target: float
    td_error: float
    terminal: bool


class RLStateEncoder:
    """Build [environment, identity, expectation, memory, policy] state."""

    def __init__(self, config: EmotionConfig | None = None) -> None:
        self.config = config or EmotionConfig()

    def encode(
        self,
        system: EmotionMemorySystem,
        environment_state: Vector,
        query: Vector,
        current_timestep: int,
        context_id: str,
        action_id: str,
        expected_valence: float,
        retention_mode: RetentionMode | str = RetentionMode.EMOTION,
    ) -> EncodedRLState:
        environment = self._finite_vector("environment_state", environment_state)
        query = self._finite_vector("query", query)
        if not -1.0 <= expected_valence <= 1.0:
            raise ValueError("expected_valence must be in [-1, 1].")

        identity = system.identity_ledger.vector()
        retrieval_trace = system.retrieve(
            query,
            current_timestep,
            retention_mode,
        )
        memory = self._memory_summary(retrieval_trace, len(query))
        policy = system.policy_state(context_id, action_id)

        components = [environment]
        if self.config.reinforcement_learning.identity_as_policy_input:
            components.append(identity)
        components.append((float(expected_valence),))
        if self.config.reinforcement_learning.memory_as_policy_input:
            components.append(memory)
        components.append((policy.repeat, policy.avoidance, policy.memory_q))
        vector = tuple(value for component in components for value in component)

        return EncodedRLState(
            vector=vector,
            environment_state=environment,
            identity_state=identity,
            expected_valence=float(expected_valence),
            memory_state=memory,
            policy_repeat=policy.repeat,
            policy_avoidance=policy.avoidance,
            policy_memory_q=policy.memory_q,
            retrieval_trace=retrieval_trace,
        )

    @staticmethod
    def _finite_vector(name: str, values: Vector) -> Vector:
        vector = tuple(float(value) for value in values)
        if not vector:
            raise ValueError(f"{name} must not be empty.")
        if not all(math.isfinite(value) for value in vector):
            raise ValueError(f"{name} must contain only finite values.")
        return vector

    @staticmethod
    def _memory_summary(trace: RetrievalTrace, dimension: int) -> Vector:
        if not trace.selections:
            return tuple(0.0 for _ in range(dimension))
        embeddings = [selection.memory_embedding for selection in trace.selections]
        if any(len(embedding) != dimension for embedding in embeddings):
            raise ValueError("Retrieved memory dimensions must match the query.")

        scores = np.asarray(
            [selection.final_score for selection in trace.selections],
            dtype=np.float64,
        )
        scores -= float(np.max(scores))
        weights = np.exp(scores)
        weights /= float(np.sum(weights))
        summary = sum(
            weight * np.asarray(embedding, dtype=np.float64)
            for weight, embedding in zip(weights, embeddings, strict=True)
        )
        return tuple(float(value) for value in summary)


class LinearQLearner:
    """A small deterministic linear Q-learning backend for discrete actions."""

    def __init__(
        self,
        input_dimension: int,
        number_of_actions: int,
        config: EmotionConfig | None = None,
    ) -> None:
        if input_dimension <= 0:
            raise ValueError("input_dimension must be positive.")
        if number_of_actions <= 0:
            raise ValueError("number_of_actions must be positive.")
        self.config = config or EmotionConfig()
        self.input_dimension = input_dimension
        self.number_of_actions = number_of_actions
        self.weights = np.zeros(
            (number_of_actions, input_dimension),
            dtype=np.float64,
        )
        self.bias = np.zeros(number_of_actions, dtype=np.float64)
        self._random = np.random.default_rng(
            self.config.reinforcement_learning.seed
        )

    def q_values(self, state: EncodedRLState | Vector) -> Vector:
        vector = self._state_vector(state)
        values = self.weights @ vector + self.bias
        return tuple(float(value) for value in values)

    def select_action(
        self,
        state: EncodedRLState | Vector,
        explore: bool = True,
    ) -> int:
        epsilon = self.config.reinforcement_learning.epsilon
        if explore and self._random.random() < epsilon:
            return int(self._random.integers(self.number_of_actions))
        return int(np.argmax(self.q_values(state)))

    def update(
        self,
        state: EncodedRLState | Vector,
        action: int,
        environment_reward: float,
        next_state: EncodedRLState | Vector,
        terminal: bool = False,
    ) -> TDUpdate:
        if not 0 <= action < self.number_of_actions:
            raise ValueError("action is outside the configured action space.")
        if not math.isfinite(environment_reward):
            raise ValueError("environment_reward must be finite.")

        vector = self._state_vector(state)
        next_vector = self._state_vector(next_state)
        prediction = float(self.weights[action] @ vector + self.bias[action])
        next_value = 0.0 if terminal else max(self.q_values(tuple(next_vector)))
        rl_config = self.config.reinforcement_learning
        target = float(
            environment_reward + rl_config.discount_factor * next_value
        )
        td_error = target - prediction
        self.weights[action] += rl_config.learning_rate * td_error * vector
        self.bias[action] += rl_config.learning_rate * td_error

        return TDUpdate(
            action=action,
            environment_reward=float(environment_reward),
            prediction=prediction,
            target=target,
            td_error=td_error,
            terminal=terminal,
        )

    def _state_vector(self, state: EncodedRLState | Vector) -> np.ndarray:
        values = state.vector if isinstance(state, EncodedRLState) else state
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (self.input_dimension,):
            raise ValueError(
                f"State must have shape {(self.input_dimension,)}, got {vector.shape}."
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("State must contain only finite values.")
        return vector
