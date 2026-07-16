"""Retention and fidelity policies for emotion-aware episodic memory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from agentic_ai.memory.emotion.config import EmotionConfig
from agentic_ai.memory.emotion.models import OutcomeLabel, TransitionResult


class RetentionMode(str, Enum):
    NONE = "none"
    TIME = "time"
    IDENTITY = "identity"
    REWARD = "reward"
    EMOTION = "emotion"


@dataclass(frozen=True)
class RetentionResult:
    mode: RetentionMode
    age: float
    identity_retention: float
    retention_floor: float
    retention: float
    active: bool


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return float(min(high, max(low, value)))


class RetentionEngine:
    """Compute survival/fidelity strength without affecting relevance."""

    def __init__(self, config: EmotionConfig | None = None) -> None:
        self.config = config or EmotionConfig()

    def evaluate(
        self,
        transition: TransitionResult,
        age: float,
        mode: RetentionMode | str = RetentionMode.EMOTION,
        active: bool = True,
    ) -> RetentionResult:
        if age < 0.0:
            raise ValueError("Memory age must be non-negative.")
        mode = RetentionMode(mode)
        if not active:
            return RetentionResult(mode, age, 0.0, 0.0, 0.0, False)

        config = self.config.retention
        alignment = transition.appraisal.identity_alignment
        identity_retention = math.exp(
            -config.identity_decay_lambda * (1.0 - alignment) * age
        )
        time_retention = math.exp(-config.time_decay_lambda * age)

        if mode is RetentionMode.NONE:
            floor = 0.0
            retention = 1.0
        elif mode is RetentionMode.TIME:
            floor = 0.0
            retention = time_retention
        elif mode is RetentionMode.IDENTITY:
            floor = 0.0
            retention = identity_retention
        elif mode is RetentionMode.REWARD:
            floor = self._reward_floor(transition.state.operational_label)
            retention = floor + (1.0 - floor) * time_retention
        else:
            floor = transition.state.retention_floor
            retention = floor + (1.0 - floor) * identity_retention

        return RetentionResult(
            mode=mode,
            age=age,
            identity_retention=_clip(identity_retention),
            retention_floor=_clip(floor, 0.0, config.maximum_floor),
            retention=_clip(retention),
            active=True,
        )

    def _reward_floor(self, label: OutcomeLabel) -> float:
        config = self.config.retention
        floors = {
            OutcomeLabel.ORDINARY: config.reward_ordinary_floor,
            OutcomeLabel.SUCCESS: config.reward_success_floor,
            OutcomeLabel.FAILURE: config.reward_failure_floor,
            OutcomeLabel.WOUND: config.reward_wound_floor,
            OutcomeLabel.TRAUMA: config.reward_trauma_floor,
        }
        return floors[label]
