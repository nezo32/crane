import torch
from torch import nn

from ..base.fusion import MemoryFusion


class CrossAttentionMemoryFusion(MemoryFusion):
    """
    Input tokens query over the retrieved memory units. Per-memory significance
    weights are added as a log-bias into attention scores so highly significant
    memory units dominate the resulting fused tokens. A sigmoid gate decides
    how much the memory-derived signal updates each token.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self._embedding_dim = embedding_dim
        self._num_heads = num_heads

        self.cross = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Linear(2 * embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim),
        )
        self.norm_out = nn.LayerNorm(embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(
        self,
        tokens: torch.Tensor,
        memory: torch.Tensor,
        memory_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if tokens.dim() != 3 or tokens.size(-1) != self._embedding_dim:
            raise ValueError(
                f"tokens must be (B, T, {self._embedding_dim}), got {tuple(tokens.shape)}"
            )
        if memory.dim() != 3 or memory.size(-1) != self._embedding_dim:
            raise ValueError(
                f"memory must be (B, M, {self._embedding_dim}), got {tuple(memory.shape)}"
            )
        if tokens.size(0) != memory.size(0):
            raise ValueError("tokens and memory must share batch dim")

        if memory.size(1) == 0:
            return tokens

        attn_mask = self._build_bias(memory_weights, tokens.size(1), tokens.device)
        attended, _ = self.cross(
            query=tokens,
            key=memory,
            value=memory,
            attn_mask=attn_mask,
            need_weights=False,
        )

        gate = torch.sigmoid(self.gate(torch.cat([tokens, attended], dim=-1)))
        h = self.norm(tokens + gate * attended)
        return self.norm_out(h + self.ffn(h))

    def _build_bias(
        self,
        memory_weights: torch.Tensor | None,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if memory_weights is None:
            return None
        if memory_weights.dim() != 2:
            raise ValueError("memory_weights must be (B, M)")

        batch, m = memory_weights.shape
        bias = torch.log(memory_weights.clamp(min=1e-8)).to(device)
        bias = bias.unsqueeze(1).expand(batch, seq_len, m)
        bias = bias.unsqueeze(1).expand(batch, self._num_heads, seq_len, m)
        return bias.reshape(batch * self._num_heads, seq_len, m)


__all__ = [
    "CrossAttentionMemoryFusion",
]
