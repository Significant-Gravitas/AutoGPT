#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path, PurePosixPath

DOC_IMAGES = PurePosixPath("docs/platform/.gitbook/assets")


def validate_checkout(root: Path) -> tuple[int, int]:
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root, encoding="utf-8"
    ).split("\0")[:-1]
    missing = []
    omitted_images = 0
    for name in tracked:
        path = root / name
        if path.exists() or path.is_symlink():
            continue
        relative = PurePosixPath(name)
        if relative.is_relative_to(DOC_IMAGES) and relative.suffix in {".png", ".gif"}:
            omitted_images += 1
        else:
            missing.append(name)
    if missing:
        raise ValueError("Missing tracked source files:\n" + "\n".join(sorted(missing)))
    return len(tracked), omitted_images


def main() -> int:
    root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], encoding="utf-8"
        ).strip()
    )
    try:
        count, omitted = validate_checkout(root)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    print(f"Verified {count} tracked paths; omitted {omitted} documentation images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
