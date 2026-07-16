"""Deterministic query-conditioned attention over one eligible event tree."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from agentic_ai.memory.emotion.config import EmotionConfig
from agentic_ai.memory.emotion.models import Vector
from agentic_ai.memory.emotion.topology import EventTree


@dataclass(frozen=True)
class NodeAttention:
    node_id: str
    node_type: str
    logit: float
    weight: float
    provenance: str
    path_to_root: tuple[str, ...]


@dataclass(frozen=True)
class TreeAttentionResult:
    tree_root_id: str
    tree_embedding: Vector
    node_attention: tuple[NodeAttention, ...]

    @property
    def highest_attention_node(self) -> str:
        return max(self.node_attention, key=lambda item: item.weight).node_id

    @property
    def weight_sum(self) -> float:
        return sum(item.weight for item in self.node_attention)


class TreeAttentionEngine:
    """Scaled dot-product attention restricted to a single event tree."""

    def __init__(
        self,
        config: EmotionConfig | None = None,
        query_projection: np.ndarray | None = None,
        key_projection: np.ndarray | None = None,
        value_projection: np.ndarray | None = None,
        type_bias: Mapping[str, float] | None = None,
    ) -> None:
        self.config = config or EmotionConfig()
        self.query_projection = query_projection
        self.key_projection = key_projection
        self.value_projection = value_projection
        self.type_bias = dict(type_bias or {})

    def attend(self, query: Vector, tree: EventTree) -> TreeAttentionResult:
        embedded_nodes = tuple(node for node in tree.nodes if node.embedding is not None)
        if not embedded_nodes:
            raise ValueError("Tree attention requires at least one embedded node.")

        dimension = len(query)
        if dimension == 0:
            raise ValueError("Query embedding must not be empty.")
        for node in embedded_nodes:
            if len(node.embedding or ()) != dimension:
                raise ValueError("Query and tree-node dimensions must match.")

        wq = self._projection(self.query_projection, dimension, "query")
        wk = self._projection(self.key_projection, dimension, "key")
        wv = self._projection(self.value_projection, dimension, "value")
        query_array = np.asarray(query, dtype=np.float64)
        projected_query = wq @ query_array

        logits = []
        values = []
        for node in embedded_nodes:
            node_array = np.asarray(node.embedding, dtype=np.float64)
            projected_key = wk @ node_array
            projected_value = wv @ node_array
            depth_bias = (
                self.config.tree_attention.depth_bias_per_level * tree.depth(node.node_id)
            )
            logit = (
                float(np.dot(projected_query, projected_key)) / math.sqrt(dimension)
                + self.type_bias.get(node.node_type, 0.0)
                + depth_bias
            )
            logits.append(logit)
            values.append(projected_value)

        weights = self._softmax(np.asarray(logits, dtype=np.float64))
        tree_embedding = sum(
            weight * value for weight, value in zip(weights, values, strict=True)
        )
        node_attention = tuple(
            NodeAttention(
                node_id=node.node_id,
                node_type=node.node_type,
                logit=float(logit),
                weight=float(weight),
                provenance=node.provenance,
                path_to_root=tree.path_to_root(node.node_id),
            )
            for node, logit, weight in zip(
                embedded_nodes,
                logits,
                weights,
                strict=True,
            )
        )

        result = TreeAttentionResult(
            tree_root_id=tree.root_id,
            tree_embedding=tuple(float(value) for value in tree_embedding),
            node_attention=node_attention,
        )
        if (
            self.config.tree_attention.attention_weights_must_sum_to_one
            and not math.isclose(result.weight_sum, 1.0, abs_tol=1.0e-9)
        ):
            raise RuntimeError("Tree-attention weights do not sum to one.")
        return result

    @staticmethod
    def _projection(
        projection: np.ndarray | None,
        dimension: int,
        name: str,
    ) -> np.ndarray:
        if projection is None:
            return np.eye(dimension, dtype=np.float64)
        array = np.asarray(projection, dtype=np.float64)
        if array.shape != (dimension, dimension):
            raise ValueError(
                f"{name} projection must have shape {(dimension, dimension)}."
            )
        return array

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - float(np.max(logits))
        exponentials = np.exp(shifted)
        return exponentials / float(np.sum(exponentials))
