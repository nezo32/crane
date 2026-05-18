import torch
from torch import nn

from ..base.extractor import SalientExtractor


class AttentionPoolSalientExtractor(SalientExtractor):
    """
    `num_slots` learnable query vectors cross-attend over the input token
    embeddings. Each slot ends up holding an attention-pooled summary of one
    facet of the input — these slots are the salient feature representation.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_slots: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self._embedding_dim = embedding_dim
        self._num_slots = num_slots

        self.slots = nn.Parameter(torch.randn(num_slots, embedding_dim) * 0.02)
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.norm_out = nn.LayerNorm(embedding_dim)

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.size(-1) != self._embedding_dim:
            raise ValueError(
                f"expected (B, T, {self._embedding_dim}), got {tuple(x.shape)}"
            )

        batch = x.size(0)
        q = self.slots.unsqueeze(0).expand(batch, -1, -1)

        attended, _ = self.attn(q, x, x, need_weights=False)
        h = self.norm(q + attended)
        return self.norm_out(h + self.ffn(h))


__all__ = [
    "AttentionPoolSalientExtractor",
]
