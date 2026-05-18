from abc import ABC, abstractmethod

import torch
from torch import nn


class MemoryFusion(nn.Module, ABC):
    """
    Analysis stage that combines the input token embeddings with the
    significant memory units retrieved from the graph. Output is an enriched
    sequence that downstream reasoning operates on.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Input:
            tokens:         (B, T, D)
            memory:         (B, M, D)
            memory_weights: (B, M) — significance scores per memory unit
                            (used as additive log-weight bias on attention)

        Output:
            (B, T, D)
        """
        raise NotImplementedError


__all__ = [
    "MemoryFusion",
]
