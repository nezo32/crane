from abc import ABC, abstractmethod

import torch
from torch import nn


class Reasoner(nn.Module, ABC):
    """
    Reasons over a fused sequence of input + memory and produces a single
    "reasoning state" vector that will be turned into a thought by ThoughtHead.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: (B, T, D) fused sequence

        Output:
            (B, D) — pooled reasoning state
        """
        raise NotImplementedError


__all__ = [
    "Reasoner",
]
