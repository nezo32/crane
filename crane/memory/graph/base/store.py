from abc import ABC, abstractmethod
from typing import Any

import torch

from .types import RetrievalResult


class MemoryGraphStore(ABC):
    """
    Vector-graph memory: each memory unit has a dense vector and is connected
    to other units by weighted association edges. Retrieval combines vector
    similarity with graph traversal so a query activates both a node and its
    neighborhood (linked memory).
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the dimensionality of stored vectors."""
        raise NotImplementedError

    @abstractmethod
    def retrieve(
        self,
        query: torch.Tensor,
        k: int,
        hops: int = 1,
    ) -> RetrievalResult:
        """
        Vector search for top-`k` nodes then expand up to `hops` hops along
        `ASSOCIATED_WITH` edges.

        Input:
            query: (D,) tensor
        """
        raise NotImplementedError

    @abstractmethod
    def write(
        self,
        vectors: torch.Tensor,
        payload: list[dict[str, Any]],
        links: list[tuple[str, str, float]],
    ) -> list[str]:
        """
        Insert new memory nodes with weighted links to existing nodes.

        Input:
            vectors: (B, D) — one new node per row
            payload: list of length B with arbitrary metadata
            links:   list of (existing_node_id, new_node_marker, weight). The
                     `new_node_marker` is the placeholder string `"<new:i>"`
                     that resolves to the i-th newly created node.

        Returns:
            ids of the newly created nodes in the same order as `vectors`.
        """
        raise NotImplementedError

    @abstractmethod
    def reinforce(
        self,
        edge_updates: list[tuple[str, str, float]],
    ) -> None:
        """
        Hebbian-style update: add `delta` to the weight of each edge between
        the given pair of node ids (creating it if absent). Co-activation
        strengthens association. `delta` may be negative for prediction-error
        driven down-weighting.
        """
        raise NotImplementedError

    @abstractmethod
    def bump_activation(
        self,
        node_ids: list[str],
        delta: float,
    ) -> None:
        """
        Add `delta` to each given node's activation. Nodes that are activated
        more often resist pruning; this is the node-level analog of edge
        reinforcement.
        """
        raise NotImplementedError

    @abstractmethod
    def decay(self, factor: float) -> None:
        """
        Multiply both edge weights and node activations by `factor` (<= 1) so
        unused links and unused nodes fade together — the moment-to-moment
        forgetting curve.
        """
        raise NotImplementedError

    @abstractmethod
    def prune(
        self,
        *,
        min_edge_weight: float,
        min_node_activation: float,
        max_nodes: int | None = None,
    ) -> dict[str, int]:
        """
        Hard pruning ("synaptic pruning") — physically delete:
          1. edges with weight strictly less than `min_edge_weight`,
          2. nodes whose activation is strictly less than `min_node_activation`
             AND have no remaining edges with weight >= `min_edge_weight`,
          3. if `max_nodes` is set, all but the top-`max_nodes` by activation.

        Returns a dict with counts: {"edges_deleted": int, "nodes_deleted": int}.
        """
        raise NotImplementedError


__all__ = [
    "MemoryGraphStore",
]
