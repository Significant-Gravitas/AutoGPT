"""Checks on the packaging files that no other test runs."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _runtime_modules() -> set[str]:
    """The import names of pyproject.toml's runtime dependencies."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {re.split(r"[\s;<>=!~\[]", spec, maxsplit=1)[0] for spec in project["dependencies"]}
    return {name.lower().replace("-", "_") for name in names}


def test_the_homebrew_test_imports_every_runtime_dependency() -> None:
    """The formula's test ran only `--version`, which needs pydantic alone,
    so a release whose resource blocks missed another dependency
    (pydantic-settings, say) passed it and installed a shim that failed at
    `start`."""
    formula = (ROOT / "packaging" / "homebrew" / "autogpt-local-executor.rb").read_text(
        encoding="utf-8"
    )
    test_block = formula.split("test do", 1)[1]
    imported = {
        name.strip()
        for line in re.findall(r'"import ([^"]+)"', test_block)
        for name in line.split(",")
    }

    assert _runtime_modules() <= imported
