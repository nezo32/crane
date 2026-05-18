import torch
from torch import nn

from ..base.reasoner import Reasoner


class TransformerReasoner(Reasoner):
    """
    Stack of standard Transformer encoder blocks (multi-head self-attention
    + position-wise FFN) prepended with a learnable `[CLS]`-style state token.
    The state token's final embedding is returned as the reasoning state.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_layers: int = 4,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self._embedding_dim = embedding_dim
        ff = feedforward_dim if feedforward_dim is not None else 4 * embedding_dim

        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.state_token = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
        self.norm = nn.LayerNorm(embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self._embedding_dim:
            raise ValueError(
                f"expected (B, T, {self._embedding_dim}), got {tuple(x.shape)}"
            )

        batch = x.size(0)
        state = self.state_token.expand(batch, -1, -1)
        h = torch.cat([state, x], dim=1)
        h = self.encoder(h)
        return self.norm(h[:, 0])


__all__ = [
    "TransformerReasoner",
]
