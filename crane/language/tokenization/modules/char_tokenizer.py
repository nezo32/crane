from typing import Sequence

from ..base.tokenizer import Tokenizer


class CharTokenizer(Tokenizer):
    """
    Minimal byte-level character tokenizer used by the demo pipeline.

    Reserved ids:
        0 = <pad>
        1 = <bos>
        2 = <eos>
        3 = <unk>
    Bytes 0..255 are mapped to ids 4..259.
    """

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    _OFFSET = 4

    def __init__(self) -> None:
        self._vocab = self._OFFSET + 256

    def vocab_size(self) -> int:
        return self._vocab

    @property
    def pad_id(self) -> int:
        return self.PAD_ID

    @property
    def bos_id(self) -> int:
        return self.BOS_ID

    @property
    def eos_id(self) -> int:
        return self.EOS_ID

    def encode(self, text: str) -> list[int]:
        data = text.encode("utf-8")
        return [self.BOS_ID, *(b + self._OFFSET for b in data), self.EOS_ID]

    def decode(self, token_ids: Sequence[int]) -> str:
        out = bytearray()
        for tid in token_ids:
            if tid in (self.PAD_ID, self.BOS_ID, self.EOS_ID):
                continue
            if tid == self.UNK_ID:
                out.append(ord("?"))
                continue
            byte = tid - self._OFFSET
            if 0 <= byte < 256:
                out.append(byte)
        return out.decode("utf-8", errors="replace")


__all__ = [
    "CharTokenizer",
]
