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


def test_hello_and_the_audit_log_report_the_version_releases_are_tagged_with() -> None:
    """__version__, which HELLO and every audit record send, was a literal
    of its own ("0.0.1-experimental") that no release step checks, while
    `--version` printed the installed distribution's: pyproject.toml's,
    which the release workflow checks against the tag."""
    from importlib.metadata import version

    import autogpt_local_executor
    from autogpt_local_executor.audit import _detect_shim_version

    assert autogpt_local_executor.__version__ == version("autogpt-local-executor")
    assert _detect_shim_version() == autogpt_local_executor.__version__


def test_a_source_tree_that_isnt_installed_reports_pyproject_tomls_version(
    monkeypatch,
) -> None:
    from importlib.metadata import PackageNotFoundError

    from autogpt_local_executor import _version

    def not_installed(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr(_version, "version", not_installed)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert _version.distribution_version() == project["version"]
