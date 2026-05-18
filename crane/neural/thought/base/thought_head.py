from abc import ABC, abstractmethod

import torch
from torch import nn


class ThoughtHead(nn.Module, ABC):
    """
    Reads the reasoning state and emits:
        - thought_vector: a continuous, non-linguistic "hypothesis"
        - confidence:     a scalar per batch element used by the cognitive
                          engine to decide whether to stop iterating
    """

    @property
    @abstractmethod
    def thought_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            state: (B, D)

        Output:
            thought_vector: (B, D_thought)
            confidence:     (B,)  in [0, 1]
        """
        raise NotImplementedError


__all__ = [
    "ThoughtHead",
]
