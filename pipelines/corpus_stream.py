"""
Disk-backed token stream for large corpora.

Avoids loading the full corpus text and a Python ``list[int]`` (which can
use several GB for WikiText-103). Tokens are stored as int32 and read via
``numpy.memmap`` (~4 bytes per token on disk, negligible RAM).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from crane.language.tokenization.modules.bpe_tokenizer import BPETokenizer

DEFAULT_CACHE_PATH = Path("data/corpus.tokens.int32")
META_SUFFIX = ".meta.json"


def cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + META_SUFFIX)


def _corpus_fingerprint(corpus_path: Path) -> dict[str, object]:
    stat = corpus_path.stat()
    return {
        "corpus_path": str(corpus_path.resolve()),
        "corpus_size": int(stat.st_size),
        "corpus_mtime_ns": int(stat.st_mtime_ns),
    }


def cache_is_valid(
    corpus_path: Path,
    cache_path: Path,
    *,
    tokenizer_name: str,
) -> bool:
    meta_path = cache_meta_path(cache_path)
    if not cache_path.is_file() or not meta_path.is_file():
        return False
    if not corpus_path.is_file():
        return False

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    fp = _corpus_fingerprint(corpus_path)
    return (
        meta.get("tokenizer_name") == tokenizer_name
        and meta.get("corpus_path") == fp["corpus_path"]
        and meta.get("corpus_size") == fp["corpus_size"]
        and meta.get("corpus_mtime_ns") == fp["corpus_mtime_ns"]
        and int(meta.get("num_tokens", 0)) > 0
        and int(meta.get("num_tokens", 0)) * 4 == cache_path.stat().st_size
    )


def build_token_cache(
    corpus_path: Path,
    tokenizer: BPETokenizer,
    cache_path: Path,
    *,
    tokenizer_name: str,
    log_every_lines: int = 50_000,
) -> int:
    """
    Tokenize ``corpus_path`` line-by-line into a flat int32 file.

    Line-wise encoding avoids holding the full text in RAM and keeps BPE
  boundaries sane for newline-delimited wiki dumps.
    """
    corpus_path = corpus_path.resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    num_tokens = 0
    num_lines = 0
    print(f"building token cache -> {cache_path}")

    with corpus_path.open(encoding="utf-8", errors="replace") as src, tmp.open(
        "wb"
    ) as out:
        for line in src:
            ids = tokenizer.encode_raw(line)
            if not ids:
                continue
            arr = np.asarray(ids, dtype=np.int32)
            arr.tofile(out)
            num_tokens += int(arr.size)
            num_lines += 1
            if log_every_lines and num_lines % log_every_lines == 0:
                print(f"  {num_lines:,} lines, {num_tokens:,} tokens...")

    if num_tokens == 0:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"no tokens produced from {corpus_path}")

    tmp.replace(cache_path)
    meta = {
        **_corpus_fingerprint(corpus_path),
        "tokenizer_name": tokenizer_name,
        "num_tokens": num_tokens,
        "dtype": "int32",
    }
    cache_meta_path(cache_path).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(
        f"token cache ready: {num_tokens:,} tokens "
        f"({cache_path.stat().st_size / (1024**2):.1f} MiB on disk)"
    )
    return num_tokens


def open_token_stream(
    corpus_path: Path,
    tokenizer: BPETokenizer,
    cache_path: Path,
    *,
    tokenizer_name: str,
    rebuild: bool = False,
) -> np.memmap:
    """Return a read-only memmap over cached token ids."""
    if rebuild or not cache_is_valid(corpus_path, cache_path, tokenizer_name=tokenizer_name):
        build_token_cache(
            corpus_path,
            tokenizer,
            cache_path,
            tokenizer_name=tokenizer_name,
        )

    meta = json.loads(cache_meta_path(cache_path).read_text(encoding="utf-8"))
    n = int(meta["num_tokens"])
    return np.memmap(cache_path, dtype=np.int32, mode="r", shape=(n,))


def split_train_val(
    stream: np.memmap,
    val_fraction: float,
) -> tuple[np.memmap, np.memmap]:
    """Hold out the last ``val_fraction`` of tokens for validation."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    n = int(stream.shape[0])
    val_n = max(1, int(n * val_fraction))
    train_n = n - val_n
    if train_n < 128:
        raise ValueError(
            f"train split too small ({train_n} tokens); "
            f"reduce CRANE_VAL_FRACTION or use a larger corpus"
        )
    return stream[:train_n], stream[train_n:]


def sample_window(
    stream: np.memmap,
    window: int,
    bos_id: int,
) -> tuple[list[int], list[int]]:
    max_start = len(stream) - 2 * window - 1
    if max_start <= 0:
        raise ValueError(
            f"stream too short ({len(stream)} tokens) for window={window}"
        )
    start = int(np.random.randint(0, max_start + 1))
    prefix = stream[start : start + window].tolist()
    target = [bos_id, *stream[start + window : start + 2 * window].tolist()]
    return prefix, target


def decode_snippet(
    stream: np.memmap,
    tokenizer: BPETokenizer,
    *,
    center: int | None = None,
    token_span: int = 96,
) -> str:
    """Decode a short span from the token stream for logging samples."""
    n = len(stream)
    if n == 0:
        return ""
    if center is None:
        center = int(np.random.randint(0, n))
    start = max(0, center - token_span // 2)
    end = min(n, start + token_span)
    return tokenizer.decode(stream[start:end].tolist())


def default_cache_path() -> Path:
    return Path(os.environ.get("CRANE_TOKEN_CACHE_PATH", str(DEFAULT_CACHE_PATH)))
