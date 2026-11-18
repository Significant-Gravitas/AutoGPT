"""Checks on the packaging files that no other test runs."""

from __future__ import annotations

import re
import tomllib
import zipfile
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


def test_the_wheel_pipx_builds_holds_the_autostart_templates(tmp_path: Path) -> None:
    """The platform's setup command is `pipx install git+...`, which builds
    this wheel. pyproject.toml force-included autogpt_local_executor/templates,
    which the package already ships, and hatchling refuses a file added twice,
    so that build failed. CI installs the package editable, which never builds
    the wheel."""
    from hatchling.builders.wheel import WheelBuilder

    wheel = next(WheelBuilder(str(ROOT)).build(directory=str(tmp_path), versions=["standard"]))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    templates = {
        f"autogpt_local_executor/templates/{template.name}"
        for template in (ROOT / "autogpt_local_executor" / "templates").glob("*.template")
    }

    assert templates
    assert templates <= names


# Folders under the package's root that hold no file of its own: a virtual
# environment, caches, build output.
_NOT_THE_PACKAGES = {"__pycache__", "build", "dist", "node_modules"}


def _package_files() -> list[Path]:
    """Every file of the package, its docs, its packaging and its tests:
    all that is under the root, apart from version control, caches, build
    output and virtual environments."""
    files = []
    pending = [ROOT]
    while pending:
        folder = pending.pop()
        for entry in sorted(folder.iterdir()):
            if entry.is_symlink():
                continue
            if entry.is_dir():
                hidden = entry.name.startswith(".") and entry.name != ".github"
                if (
                    hidden
                    or entry.name in _NOT_THE_PACKAGES
                    or entry.name.endswith(".egg-info")
                    or (entry / "pyvenv.cfg").exists()
                ):
                    continue
                pending.append(entry)
            elif entry.is_file():
                files.append(entry)
    return files


def test_nothing_names_a_host_the_project_does_not_run() -> None:
    """The default platform, the docs' endpoints and the security contact
    named a domain that isn't the project's: a sign-in with no
    --platform-url went to whoever holds it. The hosted platform is
    backend.agpt.co and platform.agpt.co; nothing here may name the other
    domain again, in any letter case."""
    not_ours = b"autogpt" + b".net"
    files = _package_files()
    named = [
        f"{path.relative_to(ROOT).as_posix()}:{number}"
        for path in files
        for number, line in enumerate(path.read_bytes().splitlines(), start=1)
        if not_ours in line.lower()
    ]

    # The walk reaches the code, the docs and the packaging.
    reached = {path.relative_to(ROOT).as_posix() for path in files}
    assert {
        "README.md",
        "pyproject.toml",
        "autogpt_local_executor/config.py",
        "autogpt_local_executor/templates/launchd.plist.template",
        "docs/PROTOCOL.md",
        "packaging/homebrew/autogpt-local-executor.rb",
        "tests/test_packaging.py",
    } <= reached
    assert not named, "These name a domain the project does not run:\n" + "\n".join(named)
