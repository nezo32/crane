import torch
from torch import nn

from ..base.embedder import Embedder


class LookupEmbedder(Embedder, nn.Module):
    """
    Plain `nn.Embedding` table — enough to run the cognitive loop end-to-end
    with random weights. Real implementations can subclass `Embedder` directly.
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        nn.Module.__init__(self)
        self._embedding_dim = embedding_dim
        self.table = nn.Embedding(vocab_size, embedding_dim)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed_tokens(self, token_ids: torch.LongTensor) -> torch.Tensor:
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be (B, T), got {tuple(token_ids.shape)}")
        return self.table(token_ids)


__all__ = [
    "LookupEmbedder",
]
