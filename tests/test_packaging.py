"""Checks on the packaging files that no other test runs."""

from __future__ import annotations

import json
import re
import tomllib
import warnings
import zipfile
from pathlib import Path

import pytest

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


# The one line that may spell that domain, back to front as a launchd label
# is written: the LaunchAgent's earlier label, kept so that `install` and
# `uninstall` can find a service left under it and remove it.
_FORMER_LABEL_FILE = "autogpt_local_executor/install.py"
_FORMER_LABEL_LINE = b"FORMER_LAUNCHD_LABEL = " + b'"net.' + b'autogpt.shim"'


# The domain as whole labels, either way round. A label is letters, digits
# and hyphens, so a longer name that only holds these letters (a host under
# `.network`, a label that ends in `net`) is some other domain's.
_OTHER_DOMAIN = re.compile(rb"(?<![a-z0-9-])autogpt\.net(?![a-z0-9-])")
_OTHER_DOMAIN_REVERSED = re.compile(rb"(?<![a-z0-9-])net\.autogpt(?![a-z0-9-])")


def _names_the_other_domain(path: Path, line: bytes) -> bool:
    """Whether `line` of `path` names the domain that isn't the project's,
    either way round, in any letter case."""
    lowered = line.lower()
    if _OTHER_DOMAIN.search(lowered):
        return True
    if not _OTHER_DOMAIN_REVERSED.search(lowered):
        return False
    return (path.relative_to(ROOT).as_posix(), line.strip()) != (
        _FORMER_LABEL_FILE,
        _FORMER_LABEL_LINE,
    )


def test_the_check_for_the_other_domain_sees_it_either_way_round() -> None:
    """A launchd label is a domain's name back to front, which the check
    didn't look for: the LaunchAgent's label named that domain until it
    became co.agpt.shim."""
    docs = ROOT / "docs" / "CROSS_PLATFORM.md"
    install = ROOT / _FORMER_LABEL_FILE
    label = b"net." + b"autogpt.shim"

    assert _names_the_other_domain(docs, b"https://backend." + b"autogpt" + b".net/api")
    assert _names_the_other_domain(docs, b"`launchctl kickstart -k gui/501/" + label + b"`")
    assert _names_the_other_domain(docs, b"~/Library/LaunchAgents/NET." + b"AutoGPT.shim.plist")
    assert not _names_the_other_domain(docs, b"`launchctl kickstart -k gui/501/co.agpt.shim`")
    # Whole labels only: a name that merely holds those letters is another's.
    assert not _names_the_other_domain(docs, b"https://pla" + b"net." + b"autogpt.example/")
    assert not _names_the_other_domain(docs, b"the " + b"autogpt" + b".network folder")
    assert not _names_the_other_domain(docs, b"see my" + b"autogpt" + b".net for more")
    assert _names_the_other_domain(docs, b"wss://ws." + b"autogpt" + b".net:443/x")
    # Only the line that keeps the earlier label, and only as it stands.
    assert not _names_the_other_domain(install, b"    " + _FORMER_LABEL_LINE)
    assert _names_the_other_domain(install, b'LAUNCHD_LABEL = "' + label + b'"')
    assert _names_the_other_domain(docs, _FORMER_LABEL_LINE)
    assert _FORMER_LABEL_LINE.decode() in install.read_text(encoding="utf-8").splitlines()


def test_nothing_names_a_host_the_project_does_not_run() -> None:
    """The default platform, the docs' endpoints and the security contact
    named a domain that isn't the project's: a sign-in with no
    --platform-url went to whoever holds it. The hosted platform is
    backend.agpt.co and platform.agpt.co; nothing here may name the other
    domain again, in any letter case, or back to front as a launchd label
    names one."""
    not_ours = b"autogpt" + b".net"
    files = _package_files()
    named = [
        f"{path.relative_to(ROOT).as_posix()}:{number}"
        for path in files
        for number, line in enumerate(path.read_bytes().splitlines(), start=1)
        if _names_the_other_domain(path, line)
    ]
    assert not_ours not in _FORMER_LABEL_LINE

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


def test_every_source_file_compiles_without_a_warning() -> None:
    """A backslash that starts no escape (a Windows path, or `\\"`, in a
    docstring that isn't raw) is a warning now and an error in a later
    Python, and meanwhile the text isn't what was written: `\\"` reads as
    a bare quote. The lint rules in use don't look for it."""
    warned = []
    for folder in ("autogpt_local_executor", "scripts", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                compile(path.read_bytes(), str(path), "exec")
            warned += [
                f"{path.relative_to(ROOT).as_posix()}:{warning.lineno}: {warning.message}"
                for warning in caught
            ]

    assert not warned, "\n".join(warned)


# ── The Scoop manifest ───────────────────────────────────────────────────────

SCOOP = ROOT / "packaging" / "scoop" / "autogpt-local-executor.json"


def _scoop_manifest() -> dict:
    return json.loads(SCOOP.read_text(encoding="utf-8"))


def test_the_scoop_manifest_installs_the_file_scoop_downloaded() -> None:
    """Its installer script ran `pip install --target ... "$url"`. Scoop
    sets no `$url` for an installer script, so pip was handed nothing and
    failed ("Invalid requirement"). With the address written in, pip would
    have fetched the wheel a second time, outside the hash Scoop checked.

    Scoop leaves the file it downloaded and checked in `$dir`, under
    `$fname`: that file is what gets installed."""
    script = _scoop_manifest()["installer"]["script"]
    text = "\n".join(script)

    assert "$url" not in text
    assert '"$dir\\$fname"' in text
    # Every variable the script uses is one Scoop gives an installer script.
    used = set(re.findall(r"\$\{?([A-Za-z_]\w*)", text))
    assert used <= {"dir", "fname", "LASTEXITCODE"}, used


def test_the_scoop_manifest_installs_into_an_environment_its_command_runs_from() -> None:
    """`pip --target` put the package in a folder the launcher it made
    doesn't look in: `autogpt-shim` started and failed with
    ModuleNotFoundError. A virtual environment's launcher runs its own
    Python, which finds what was installed beside it."""
    manifest = _scoop_manifest()
    text = "\n".join(manifest["installer"]["script"])

    assert "--target" not in text
    assert 'python -m venv "$dir\\venv"' in text
    assert '& "$dir\\venv\\Scripts\\python.exe" -m pip install' in text
    assert manifest["bin"] == [["venv\\Scripts\\autogpt-shim.exe", "autogpt-shim"]]


# The line after a step that runs a program: stop, with an error that ends
# the script, when the program's exit status wasn't 0.
_STOPS_WHEN_IT_FAILED = re.compile(r"if \(\$LASTEXITCODE -ne 0\) \{ throw '[^']+' \}")


def test_the_scoop_installer_stops_at_a_step_that_failed() -> None:
    """PowerShell goes on after a program that failed. Scoop would then
    report an install with no `autogpt-shim` in it."""
    script = _scoop_manifest()["installer"]["script"]
    steps = [index for index, line in enumerate(script) if "python" in line]

    assert len(steps) == 2
    for index in steps:
        assert _STOPS_WHEN_IT_FAILED.fullmatch(script[index + 1].strip()), script[index + 1]


@pytest.mark.parametrize(
    "line",
    [
        # Each names the exit status, which is all the check used to ask.
        "Write-Host $LASTEXITCODE",
        "if ($LASTEXITCODE -ne 0) { Write-Warning 'Could not install the wheel.' }",
        "if ($LASTEXITCODE -eq 0) { throw 'Could not install the wheel.' }",
        "# if ($LASTEXITCODE -ne 0) { throw 'Could not install the wheel.' }",
    ],
)
def test_a_line_that_only_names_the_exit_status_is_no_stop(line: str) -> None:
    assert "$LASTEXITCODE" in line
    assert not _STOPS_WHEN_IT_FAILED.fullmatch(line)


def test_the_scoop_manifest_is_still_a_stub_and_says_so() -> None:
    """Nothing can run it until a wheel is published: its address and hash
    are placeholders, and the packaging notes and the release runbook say
    it has never been installed from a real bucket."""
    manifest = _scoop_manifest()
    assert manifest["url"].startswith("PLACEHOLDER-")
    assert manifest["hash"].startswith("PLACEHOLDER-")
    readme = (ROOT / "packaging" / "README.md").read_text(encoding="utf-8")
    runbook = (ROOT / "docs" / "RELEASE_RUNBOOK.md").read_text(encoding="utf-8")
    for document in (readme, runbook):
        # Whatever the line breaks.
        assert "has never been installed from a real bucket" in " ".join(document.split())
    assert "scoop install" in runbook
    assert "Before the manifest goes to the bucket" in " ".join(runbook.split())
