from abc import ABC, abstractmethod

import torch

from .types import RetrievalResult


class MemorySignificanceFilter(ABC):
    """
    Scores how significant each retrieved memory unit is for the current input.

    The score combines vector similarity with graph context (edge weights,
    recency, activation). The engine then picks the top-M nodes by this score.
    """

    @abstractmethod
    def score(
        self,
        query: torch.Tensor,
        result: RetrievalResult,
    ) -> torch.Tensor:
        """
        Input:
            query:  (D,)
            result: retrieval result containing N candidates

        Returns:
            scores: (N,) — higher is more significant
        """
        raise NotImplementedError


__all__ = [
    "MemorySignificanceFilter",
]
