from abc import ABC, abstractmethod

import torch
from torch import nn


class ThoughtLanguageDecoder(nn.Module, ABC):
    """
    Converts a continuous thought vector back into discrete tokens — the
    "language interface" to cognition.

    Implementations should:
      - support a teacher-forcing forward pass returning logits, and
      - support autoregressive `generate` returning token ids.
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        thought: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Input:
            thought:        (B, D)
            target_tokens:  (B, T) — teacher-forcing inputs (usually BOS-shifted)

        Output:
            logits: (B, T, vocab_size)
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        thought: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> torch.Tensor:
        """
        Autoregressively decode tokens conditioned on `thought`.

        Input:
            thought: (B, D)

        Output:
            (B, L) token ids — variable L <= max_len, padded with `eos_id`.
        """
        raise NotImplementedError


__all__ = [
    "ThoughtLanguageDecoder",
]
