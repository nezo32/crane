"""
Optional ``.env`` loading for local development.

``load_dotenv_if_present`` only reads a file when it exists and uses
``override=False`` so values already set in the process environment (Docker,
Kubernetes, CI, ``export``) always win over keys in ``.env``.
"""

from pathlib import Path

from dotenv import find_dotenv, load_dotenv


def load_dotenv_if_present(*, dotenv_path: str | Path | None = None) -> bool:
    """
    Load environment variables from a ``.env`` file if one is found.

    - If ``dotenv_path`` is set, only that path is tried (must exist).
    - Otherwise ``find_dotenv`` walks upward from the current working directory.
    - Never overrides existing ``os.environ`` entries (``override=False``).

    Returns True if a file was found and passed to ``load_dotenv``, False otherwise.
    """
    if dotenv_path is not None:
        path = Path(dotenv_path).resolve()
        if not path.is_file():
            return False
        load_dotenv(path, override=False)
        return True

    found = find_dotenv(usecwd=True)
    if not found:
        return False
    load_dotenv(found, override=False)
    return True


__all__ = ["load_dotenv_if_present"]
