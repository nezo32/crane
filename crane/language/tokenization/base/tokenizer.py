from abc import ABC, abstractmethod
from typing import Sequence

class Tokenizer(ABC):
    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: Sequence[int]) -> str:
        """Convert token IDs back into text."""
        raise NotImplementedError

    @abstractmethod
    def vocab_size(self) -> int:
        """Return tokenizer vocabulary size."""
        raise NotImplementedError
    
__all__ = [
    "Tokenizer",
]