from pathlib import Path

ROOT = ROOT = Path(__file__).resolve().parents[2] / "crane"

IGNORE_DIRS = {
    "__pycache__",
    ".venv",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
}

START = "# <auto-exports>"
END = "# </auto-exports>"


def is_public(path: Path) -> bool:
    return not path.name.startswith("_")


def should_skip_dir(path: Path) -> bool:
    return path.name in IGNORE_DIRS or path.name.startswith(".") or path is ROOT


def collect_exports(package_dir: Path) -> list[str]:
    lines: list[str] = []

    for item in sorted(package_dir.iterdir(), key=lambda p: p.name):
        if item.name == "__init__.py":
            continue

        if item.is_dir():
            if should_skip_dir(item) or not is_public(item):
                continue

            # создаём __init__.py даже если его ещё нет
            lines.append(f"from .{item.name} import *")

        elif item.is_file():
            if item.suffix != ".py" or not is_public(item):
                continue

            lines.append(f"from .{item.stem} import *")

    return lines


def update_init(package_dir: Path) -> None:
    init_file = package_dir / "__init__.py"
    exports = collect_exports(package_dir)

    block = "\n".join([START, *exports, END, ""])

    old = init_file.read_text(encoding="utf-8") if init_file.exists() else ""

    if START in old and END in old:
        before = old.split(START)[0].rstrip()
        after = old.split(END, 1)[1].lstrip()

        parts = []
        if before:
            parts.append(before)
        parts.append(block.rstrip())
        if after:
            parts.append(after)

        new = "\n\n".join(parts) + "\n"
    else:
        if old.strip():
            new = old.rstrip() + "\n\n" + block
        else:
            new = block

    init_file.write_text(new, encoding="utf-8")


def main() -> None:
    dirs = [ROOT, *[p for p in ROOT.rglob("*") if p.is_dir()]]

    # снизу вверх, чтобы вложенные пакеты были готовы раньше верхних
    for directory in sorted(dirs, key=lambda p: len(p.parts), reverse=True):
        if should_skip_dir(directory):
            continue

        update_init(directory)

    print("Generated __init__.py exports")


if __name__ == "__main__":
    main()