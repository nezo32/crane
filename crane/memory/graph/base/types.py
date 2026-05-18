from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class MemoryNode:
    """A single memory unit stored in the graph."""

    id: str
    vector: torch.Tensor
    payload: dict[str, Any] = field(default_factory=dict)
    activation: float = 0.0


@dataclass(frozen=True)
class MemoryEdge:
    """A weighted association between two memory units."""

    source_id: str
    target_id: str
    weight: float
    last_used: int = 0


@dataclass
class RetrievalResult:
    """
    Output of `MemoryGraphStore.retrieve`.

    Shapes:
        vectors:      (N, D)
        similarities: (N,)
    """

    nodes: list[MemoryNode]
    edges: list[MemoryEdge]
    vectors: torch.Tensor
    similarities: torch.Tensor
    node_ids: list[str]


__all__ = [
    "MemoryNode",
    "MemoryEdge",
    "RetrievalResult",
]
