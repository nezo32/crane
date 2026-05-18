"""
Download a small text corpus into `data/corpus.txt` for training.

By default fetches the well-known TinyShakespeare dataset (~1.1 MB plain text)
which is a standard benchmark for tiny language models.

Override via env vars:
    CRANE_CORPUS_URL   - source URL (default: TinyShakespeare on GitHub)
    CRANE_CORPUS_PATH  - destination path (default: data/corpus.txt)
"""

import os
import sys
import urllib.request
from pathlib import Path

from crane.config import load_dotenv_if_present

DEFAULT_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)
DEFAULT_PATH = Path("data/corpus.txt")


def fetch(url: str, dest: Path, *, force: bool = False) -> None:
    if dest.exists() and not force:
        size = dest.stat().st_size
        print(f"corpus already present at {dest} ({size:,} bytes); skip.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")

    request = urllib.request.Request(
        url, headers={"User-Agent": "crane-fetch-corpus/0.1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()

    dest.write_bytes(data)
    print(f"saved {len(data):,} bytes to {dest}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv_if_present()
    argv = argv if argv is not None else sys.argv[1:]
    url = os.environ.get("CRANE_CORPUS_URL", DEFAULT_URL)
    dest = Path(os.environ.get("CRANE_CORPUS_PATH", str(DEFAULT_PATH)))
    force = "--force" in argv

    try:
        fetch(url, dest, force=force)
    except Exception as exc:
        print(f"failed to fetch corpus: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
