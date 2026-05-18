"""
Download a text corpus into `data/corpus.txt` for training.

Presets (pass as first argument, or set CRANE_CORPUS_PRESET):

    tinyshakespeare   ~1.1 MB plain text (default)
    wikitext2         ~10 MB training split (WikiText-2 raw)
    wikitext103       ~100 MB training split (WikiText-103 raw)

Override via env vars:
    CRANE_CORPUS_PRESET  - preset name (default: tinyshakespeare)
    CRANE_CORPUS_URL     - direct URL (overrides preset)
    CRANE_CORPUS_PATH    - destination path (default: data/corpus.txt)

Examples:
    uv run python pipelines/fetch_corpus.py
    uv run python pipelines/fetch_corpus.py wikitext103 --force
    CRANE_CORPUS_PRESET=wikitext2 uv run python pipelines/fetch_corpus.py --force
"""

from __future__ import annotations

import os
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from crane.config import load_dotenv_if_present

DEFAULT_PATH = Path("data/corpus.txt")

# Legacy Meta/S3 URLs now 301/403; HF mirror hosts the same zip archives.
HF_WIKITEXT_103_ZIP = (
    "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/"
    "resolve/main/wikitext-103-raw-v1.zip"
)
# WikiText-2 zip only on archive.org Wayback (original S3 bucket retired).
WAYBACK_WIKITEXT_2_ZIP = (
    "https://web.archive.org/web/2023/"
    "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"
)


@dataclass(frozen=True)
class CorpusPreset:
    name: str
    description: str
    url: str
    zip_member: str | None = None


PRESETS: dict[str, CorpusPreset] = {
    "tinyshakespeare": CorpusPreset(
        name="tinyshakespeare",
        description="Tiny Shakespeare (~1.1 MB)",
        url=(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
            "data/tinyshakespeare/input.txt"
        ),
    ),
    "wikitext2": CorpusPreset(
        name="wikitext2",
        description="WikiText-2 raw training split (~10 MB)",
        url=WAYBACK_WIKITEXT_2_ZIP,
        zip_member="wiki.train.raw",
    ),
    "wikitext103": CorpusPreset(
        name="wikitext103",
        description="WikiText-103 raw training split (~100 MB)",
        url=HF_WIKITEXT_103_ZIP,
        zip_member="wikitext-103-raw/wiki.train.raw",
    ),
}


def _preset(name: str) -> CorpusPreset:
    key = name.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "shakespeare": "tinyshakespeare",
        "tiny": "tinyshakespeare",
        "wt2": "wikitext2",
        "wt103": "wikitext103",
        "wiki2": "wikitext2",
        "wiki103": "wikitext103",
    }
    resolved = aliases.get(key, key)
    if resolved not in PRESETS:
        choices = ", ".join(sorted(PRESETS))
        raise ValueError(f"unknown preset {name!r}; choose one of: {choices}")
    return PRESETS[resolved]


def download_bytes(url: str, *, timeout: int = 1800) -> bytes:
    print(f"downloading {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "crane-fetch-corpus/0.2"}
    )
    opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())
    with opener.open(request, timeout=timeout) as response:
        total = response.headers.get("Content-Length")
        total = int(total) if total else None
        chunks: list[bytes] = []
        done = 0
        block = 1 << 20
        while True:
            chunk = response.read(block)
            if not chunk:
                break
            chunks.append(chunk)
            done += len(chunk)
            if total:
                pct = 100.0 * done / total
                print(f"  {done:,} / {total:,} bytes ({pct:.1f}%)", end="\r")
            elif done % (10 * block) == 0:
                print(f"  {done:,} bytes...", end="\r")
        print()
        return b"".join(chunks)


def resolve_zip_member(names: list[str], member: str) -> str:
    if member in names:
        return member
    suffix = member.split("/")[-1]
    matches = [n for n in names if n.endswith(suffix) and not n.endswith("/")]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        preferred = [n for n in matches if "train" in n]
        if len(preferred) == 1:
            return preferred[0]
    raise ValueError(
        f"{member!r} not in archive; members: {names[:8]}{'...' if len(names) > 8 else ''}"
    )


def extract_zip_member(data: bytes, member: str) -> bytes:
    with zipfile.ZipFile(BytesIO(data)) as zf:
        path = resolve_zip_member(zf.namelist(), member)
        if path != member:
            print(f"  using zip member {path!r}")
        return zf.read(path)


def fetch(
    url: str,
    dest: Path,
    *,
    zip_member: str | None = None,
    force: bool = False,
) -> None:
    if dest.exists() and not force:
        size = dest.stat().st_size
        print(f"corpus already present at {dest} ({size:,} bytes); skip.")
        print("  use --force to re-download")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    raw = download_bytes(url)

    if zip_member:
        print(f"extracting {zip_member!r} from zip")
        text_bytes = extract_zip_member(raw, zip_member)
    else:
        text_bytes = raw

    dest.write_bytes(text_bytes)
    lines = text_bytes.count(b"\n") + 1
    print(f"saved {len(text_bytes):,} bytes ({lines:,} lines) to {dest}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv_if_present()
    argv = argv if argv is not None else sys.argv[1:]
    force = "--force" in argv
    args = [a for a in argv if a != "--force"]

    preset_name = os.environ.get("CRANE_CORPUS_PRESET", "tinyshakespeare")
    if args and not args[0].startswith("-"):
        preset_name = args[0]

    dest = Path(os.environ.get("CRANE_CORPUS_PATH", str(DEFAULT_PATH)))
    url_override = os.environ.get("CRANE_CORPUS_URL")

    try:
        if url_override:
            print(f"using CRANE_CORPUS_URL (preset {preset_name!r} ignored)")
            fetch(url_override, dest, force=force)
        else:
            preset = _preset(preset_name)
            print(f"preset: {preset.name} — {preset.description}")
            fetch(preset.url, dest, zip_member=preset.zip_member, force=force)
    except Exception as exc:
        print(f"failed to fetch corpus: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
