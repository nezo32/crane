import torch

from ..base.significance import MemorySignificanceFilter
from ..base.types import RetrievalResult


class ActivationSignificance(MemorySignificanceFilter):
    """
    Default significance scorer.

    score(node) = w_sim * cosine(query, node.vector)
                + w_edge * mean_incident_edge_weight(node)
                + w_act  * node.activation

    The mean incident edge weight captures how strongly each candidate is
    embedded in the local graph neighborhood that was activated by retrieval.
    """

    def __init__(
        self,
        similarity_weight: float = 1.0,
        edge_weight: float = 0.5,
        activation_weight: float = 0.25,
    ) -> None:
        self._w_sim = float(similarity_weight)
        self._w_edge = float(edge_weight)
        self._w_act = float(activation_weight)

    def score(
        self,
        query: torch.Tensor,
        result: RetrievalResult,
    ) -> torch.Tensor:
        n = len(result.nodes)
        if n == 0:
            return torch.zeros((0,), dtype=torch.float32)

        q = torch.nn.functional.normalize(query.to(torch.float32), dim=-1)
        v = torch.nn.functional.normalize(result.vectors.to(torch.float32), dim=-1)
        sim = v @ q

        edge_sum = torch.zeros(n, dtype=torch.float32)
        edge_count = torch.zeros(n, dtype=torch.float32)
        id_to_idx = {nid: i for i, nid in enumerate(result.node_ids)}

        for edge in result.edges:
            for nid in (edge.source_id, edge.target_id):
                idx = id_to_idx.get(nid)
                if idx is not None:
                    edge_sum[idx] += float(edge.weight)
                    edge_count[idx] += 1.0

        mean_edge = torch.where(
            edge_count > 0, edge_sum / edge_count.clamp(min=1.0), torch.zeros_like(edge_sum)
        )

        activations = torch.tensor(
            [node.activation for node in result.nodes], dtype=torch.float32
        )

        return (
            self._w_sim * sim
            + self._w_edge * mean_edge
            + self._w_act * activations
        )


__all__ = [
    "ActivationSignificance",
]
