from abc import ABC, abstractmethod
import torch

class Embedder(ABC):
    @abstractmethod
    def embed_tokens(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        Convert token IDs into embeddings.
        
        Input:
            (batch, seq_len)

        Output:
            (batch, seq_len, embedding_dim)
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        raise NotImplementedError

__all__ = [
    "Embedder",
]