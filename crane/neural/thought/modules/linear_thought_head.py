import torch
from torch import nn

from ..base.thought_head import ThoughtHead


class LinearThoughtHead(ThoughtHead):
    """
    Two linear heads on top of the reasoning state:
        - a tanh-bounded thought projection
        - a sigmoid confidence scalar
    """

    def __init__(
        self,
        state_dim: int,
        thought_dim: int | None = None,
    ) -> None:
        super().__init__()
        self._thought_dim = thought_dim if thought_dim is not None else state_dim
        self.proj = nn.Linear(state_dim, self._thought_dim)
        self.conf = nn.Linear(state_dim, 1)

    @property
    def thought_dim(self) -> int:
        return self._thought_dim

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state.dim() != 2:
            raise ValueError(f"expected (B, D), got {tuple(state.shape)}")

        thought = torch.tanh(self.proj(state))
        confidence = torch.sigmoid(self.conf(state)).squeeze(-1)
        return thought, confidence


__all__ = [
    "LinearThoughtHead",
]
