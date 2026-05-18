import torch
from torch import nn

from ..base.decoder import ThoughtLanguageDecoder


class TransformerThoughtDecoder(ThoughtLanguageDecoder):
    """
    Small Transformer decoder. The thought vector is projected and used as the
    `memory` for cross-attention; tokens are produced autoregressively.
    """

    def __init__(
        self,
        thought_dim: int,
        vocab_size: int,
        embedding_dim: int,
        max_len: int,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self._vocab_size = vocab_size
        self._embedding_dim = embedding_dim
        self._max_len = max_len
        ff = feedforward_dim if feedforward_dim is not None else 4 * embedding_dim

        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb = nn.Embedding(max_len, embedding_dim)
        self.thought_proj = nn.Linear(thought_dim, embedding_dim)

        layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def forward(
        self,
        thought: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> torch.Tensor:
        if thought.dim() != 2:
            raise ValueError(f"thought must be (B, D), got {tuple(thought.shape)}")
        if target_tokens.dim() != 2:
            raise ValueError(f"target_tokens must be (B, T), got {tuple(target_tokens.shape)}")

        return self._step(thought, target_tokens)

    @torch.no_grad()
    def generate(
        self,
        thought: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_len: int,
    ) -> torch.Tensor:
        if thought.dim() != 2:
            raise ValueError(f"thought must be (B, D), got {tuple(thought.shape)}")

        batch = thought.size(0)
        device = thought.device
        limit = min(max_len, self._max_len)

        out_ids = torch.full(
            (batch, 1), bos_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch, dtype=torch.bool, device=device)

        for _ in range(limit - 1):
            logits = self._step(thought, out_ids)
            next_tok = logits[:, -1].argmax(dim=-1)
            next_tok = torch.where(
                finished,
                torch.full_like(next_tok, eos_id),
                next_tok,
            )
            out_ids = torch.cat([out_ids, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == eos_id)
            if bool(finished.all()):
                break

        return out_ids

    def _step(
        self,
        thought: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq = tokens.shape
        if seq > self._max_len:
            raise ValueError(
                f"sequence length {seq} exceeds max_len {self._max_len}"
            )

        positions = torch.arange(seq, device=tokens.device).unsqueeze(0).expand(batch, -1)
        tgt = self.token_emb(tokens) + self.pos_emb(positions)

        memory = self.thought_proj(thought).unsqueeze(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq, device=tokens.device
        )

        h = self.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask)
        h = self.norm(h)
        return self.head(h)


__all__ = [
    "TransformerThoughtDecoder",
]
