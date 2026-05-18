from abc import ABC, abstractmethod

import torch
from torch import nn


class SalientExtractor(nn.Module, ABC):
    """
    Compresses a sequence of token embeddings into a fixed-size set of
    "salient slots" — the important / feature-extracted summary that becomes
    the query into memory and the seed of analysis.
    """

    @property
    @abstractmethod
    def num_slots(self) -> int:
        """Number of salient slots produced per input."""
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of each slot vector."""
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: (batch, seq_len, embedding_dim)

        Output:
            (batch, num_slots, embedding_dim)
        """
        raise NotImplementedError


__all__ = [
    "SalientExtractor",
]
