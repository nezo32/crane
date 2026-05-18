from pathlib import Path
from typing import Iterable, Sequence

from tokenizers import Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

from ..base.tokenizer import Tokenizer


class BPETokenizer(Tokenizer):
    """
    Byte-level BPE tokenizer backed by HuggingFace `tokenizers`.

    Primary entry point is `BPETokenizer.from_pretrained("gpt2")`, which
    downloads (and caches) the canonical GPT-2 BPE vocab (~50k tokens) from
    the HF Hub. `train_from_iterator` and `load` remain available as
    alternatives for fitting a custom vocab or loading a local file.

    `encode(text)` returns `[bos, ...content..., eos]`. `encode_raw(text)`
    returns only the content ids — useful for slicing a long stream into
    fixed-width training windows.
    """

    TRAIN_SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(
        self,
        hf_tokenizer: HFTokenizer,
        *,
        bos_id: int,
        eos_id: int,
        pad_id: int,
    ) -> None:
        self._tok = hf_tokenizer
        self._bos_id = int(bos_id)
        self._eos_id = int(eos_id)
        self._pad_id = int(pad_id)
        self._specials = {self._pad_id, self._bos_id, self._eos_id}

    @classmethod
    def from_pretrained(cls, name: str = "gpt2") -> "BPETokenizer":
        """
        Load a ready-trained BPE tokenizer from the HuggingFace Hub.

        Downloads on first call, cached in `~/.cache/huggingface/` afterwards.
        Special-token ids are resolved automatically:

          - GPT-2 family: `<|endoftext|>` plays bos/eos/pad
          - BERT family:  `[CLS]` -> bos, `[SEP]` -> eos, `[PAD]` -> pad
          - RoBERTa:      `<s>` -> bos, `</s>` -> eos, `<pad>` -> pad
        """
        tok = HFTokenizer.from_pretrained(name)
        bos, eos, pad = cls._resolve_specials(tok)
        return cls(tok, bos_id=bos, eos_id=eos, pad_id=pad)

    @classmethod
    def train_from_iterator(
        cls,
        texts: Iterable[str],
        vocab_size: int = 8000,
        min_frequency: int = 2,
    ) -> "BPETokenizer":
        """Train a fresh byte-level BPE on `texts`."""
        tok = HFTokenizer(BPE(unk_token="<unk>"))
        tok.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=cls.TRAIN_SPECIAL_TOKENS,
            initial_alphabet=ByteLevelPreTokenizer.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        return cls(
            tok,
            bos_id=cls._require_id(tok, "<bos>"),
            eos_id=cls._require_id(tok, "<eos>"),
            pad_id=cls._require_id(tok, "<pad>"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        tok = HFTokenizer.from_file(str(path))
        bos, eos, pad = cls._resolve_specials(tok)
        return cls(tok, bos_id=bos, eos_id=eos, pad_id=pad)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._tok.save(str(path))

    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    def encode(self, text: str) -> list[int]:
        ids = self._tok.encode(text).ids
        return [self._bos_id, *ids, self._eos_id]

    def encode_raw(self, text: str) -> list[int]:
        """Encode without prepending BOS or appending EOS."""
        return self._tok.encode(text).ids

    def decode(self, token_ids: Sequence[int]) -> str:
        filtered = [int(t) for t in token_ids if int(t) not in self._specials]
        return self._tok.decode(filtered)

    @staticmethod
    def _resolve_specials(tok: HFTokenizer) -> tuple[int, int, int]:
        """Best-effort resolution of bos/eos/pad across common tokenizer families."""

        def maybe(name: str) -> int | None:
            tid = tok.token_to_id(name)
            return int(tid) if tid is not None else None

        candidates_bos = ["<bos>", "<|endoftext|>", "<s>", "[CLS]"]
        candidates_eos = ["<eos>", "<|endoftext|>", "</s>", "[SEP]"]
        candidates_pad = ["<pad>", "<|endoftext|>", "<pad>", "[PAD]"]

        bos = next((maybe(c) for c in candidates_bos if maybe(c) is not None), None)
        eos = next((maybe(c) for c in candidates_eos if maybe(c) is not None), None)
        pad = next((maybe(c) for c in candidates_pad if maybe(c) is not None), None)

        if bos is None or eos is None or pad is None:
            raise ValueError(
                "could not resolve bos/eos/pad ids from the loaded tokenizer; "
                "supply them explicitly via the constructor"
            )
        return bos, eos, pad

    @staticmethod
    def _require_id(tok: HFTokenizer, token: str) -> int:
        tid = tok.token_to_id(token)
        if tid is None:
            raise ValueError(f"tokenizer is missing required special token {token!r}")
        return int(tid)


__all__ = [
    "BPETokenizer",
]
