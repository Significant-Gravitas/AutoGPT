"""Clean up verified leftover services after the shell observes runtime exit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from autogpt_desktop.maintenance import DataLock
from autogpt_desktop.process import ChildRegistry


def reap(directory: Path) -> None:
    with DataLock(directory) as held:
        registry = ChildRegistry(held.root / "run" / "children.json")
        registry.reap_leftovers()
        if os.path.lexists(registry.path):
            raise RuntimeError(
                "The service registry could not be read; it is retained."
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    reap(args.data_dir)


if __name__ == "__main__":
    main()
