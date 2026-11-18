"""Adding computer use keeps the shim that is installed.

The hint that preflight, `doctor` and the driver's errors show for a shim
without cua-driver ran `pipx install --force` on the repository with no
revision. A person who had installed the revision their deployment pins,
and followed it, had their shim replaced by whatever was on the default
branch, which may speak another protocol than their deployment.

The hint adds the extra's packages to the install that is there now. pip
leaves an installed package as it is, whatever it was installed from, so
the shim keeps its revision without the hint having to name one.
"""

from __future__ import annotations

import base64
import hashlib
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from autogpt_local_executor import doctor
from autogpt_local_executor.computer_use import driver
from autogpt_local_executor.computer_use.driver import (
    COMPUTER_USE_INSTALL,
    CuaDriverBridge,
    DriverUnavailable,
)
from autogpt_local_executor.config import ShimConfig

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "github.com/Significant-Gravitas/autogpt-local-executor"


def test_the_hint_adds_the_extra_to_the_install_that_is_there() -> None:
    assert COMPUTER_USE_INSTALL == (
        'pipx runpip autogpt-local-executor install "autogpt-local-executor[computer-use]"'
    )
    # Nothing in it installs the shim again, from anywhere.
    assert "--force" not in COMPUTER_USE_INSTALL
    assert "pipx install" not in COMPUTER_USE_INSTALL
    assert "git+" not in COMPUTER_USE_INSTALL
    assert REPOSITORY not in COMPUTER_USE_INSTALL
    # As cmd.exe, PowerShell and POSIX shells all take it.
    assert not set(COMPUTER_USE_INSTALL) & set("'$`!%^&|<>")


async def test_a_missing_driver_is_answered_with_the_hint(monkeypatch) -> None:
    monkeypatch.setattr(driver, "find_driver_binary", lambda _explicit=None: None)
    bridge = CuaDriverBridge(tools=frozenset({"click"}), platform="linux")
    try:
        with pytest.raises(DriverUnavailable) as missing:
            await bridge.snapshot()
    finally:
        await bridge.close()

    assert COMPUTER_USE_INSTALL in missing.value.message
    assert "einstall" not in missing.value.message
    assert "leaves the local executor" in missing.value.message


def test_doctor_shows_the_hint(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setattr(doctor, "find_driver_binary", lambda _path=None: None)
    config = ShimConfig(
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "_audit.log",
        platform_url="http://localhost:9999",
        machine_id="test-machine",
        enable_computer_use=True,
    )

    doctor.run_doctor(config)

    out = capsys.readouterr().out
    assert COMPUTER_USE_INSTALL in out
    assert "pipx install --force" not in out


# ── What the hint rests on ───────────────────────────────────────────────────


def _wheel(folder: Path, name: str, *, requires: str = "", extra: str = "") -> Path:
    """A wheel made by hand: no build tool, and nothing fetched."""
    info = f"{name}-1.0.dist-info"
    metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n"
    if extra:
        metadata += f'Provides-Extra: {extra}\nRequires-Dist: {requires}; extra == "{extra}"\n'
    files = {
        f"{name}/__init__.py": "INSTALLED = 'the first one'\n",
        f"{info}/METADATA": metadata,
        f"{info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: by-hand\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    record = []
    for path, text in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=")
        record.append(f"{path},sha256={digest.decode()},{len(text.encode())}")
    files[f"{info}/RECORD"] = "\n".join([*record, f"{info}/RECORD,,"]) + "\n"
    folder.mkdir(parents=True, exist_ok=True)
    wheel = folder / f"{name}-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, text in files.items():
            archive.writestr(path, text)
    return wheel


def _run(*command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=600,
        check=False,
    )


def test_pip_adds_an_extra_to_an_installed_package_and_leaves_the_package_alone(
    tmp_path: Path,
) -> None:
    """What the hint asks pip to do, with real pip and nothing fetched: the
    shim is installed (from a file here; pip records a Git revision or an
    index the same way), and then its extra is asked for with only the
    extra's package to be had. pip can't fetch the shim again, and doesn't
    try: it adds what the extra needs."""
    environment = tmp_path / "environment"
    made = _run(sys.executable, "-m", "venv", str(environment))
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if made.returncode != 0 or not python.exists():
        pytest.skip(f"no virtual environment with pip can be made here: {made.stderr[-200:]}")
    shim = _wheel(tmp_path / "only-the-shim", "theshim", requires="thedriver==1.0", extra="use")
    _wheel(tmp_path / "only-the-driver", "thedriver")
    pip = (str(python), "-m", "pip", "--disable-pip-version-check")
    installed = _run(*pip, "install", "--no-index", str(shim))
    assert installed.returncode == 0, installed.stderr
    module = next(environment.rglob("theshim/__init__.py"))
    module.write_text("INSTALLED = 'the first one, untouched'\n", encoding="utf-8")

    added = _run(
        *pip,
        "install",
        "--no-index",
        "--find-links",
        str(tmp_path / "only-the-driver"),
        "theshim[use]",
    )

    assert added.returncode == 0, added.stdout + added.stderr
    listed = _run(*pip, "list", "--format=freeze").stdout
    assert "theshim==1.0" in listed
    assert "thedriver==1.0" in listed
    # The installed package wasn't written again.
    assert module.read_text(encoding="utf-8") == "INSTALLED = 'the first one, untouched'\n"


# ── The docs ─────────────────────────────────────────────────────────────────

_HEADING = re.compile(r"^#{1,6} +(.*)$")


def _unpinned_installs(text: str) -> list[tuple[str, str]]:
    """Each line that installs the shim from its repository without naming
    a revision, with the heading it is under."""
    found: list[tuple[str, str]] = []
    heading = ""
    for line in text.splitlines():
        if title := _HEADING.match(line):
            heading = title.group(1)
        if "pipx install" not in line or REPOSITORY not in line:
            continue
        after = line.split(REPOSITORY, 1)[1]
        if not re.match(r"(\.git)?@[\w<]", after):
            found.append((heading, line.strip()))
    return found


@pytest.mark.parametrize(
    "document",
    [
        "README.md",
        "packaging/README.md",
        *sorted(f"docs/{p.name}" for p in (ROOT / "docs").glob("*.md")),
    ],
)
def test_the_docs_install_a_named_revision_except_for_development(document: str) -> None:
    """The README's first command was the unpinned one. A person follows
    the README, and their deployment's platform may speak another protocol
    than the default branch's head."""
    for heading, line in _unpinned_installs((ROOT / document).read_text(encoding="utf-8")):
        assert "development" in heading.lower(), (document, heading, line)


def test_the_check_tells_a_pinned_install_from_an_unpinned_one() -> None:
    text = (
        "## Install\n"
        f"pipx install git+https://{REPOSITORY}.git@0123abc\n"
        f'pipx install "git+https://{REPOSITORY}.git@<revision>"\n'
        f"pipx install git+https://{REPOSITORY}.git\n"
        "### For development\n"
        f"pipx install git+https://{REPOSITORY}.git\n"
    )

    assert _unpinned_installs(text) == [
        ("Install", f"pipx install git+https://{REPOSITORY}.git"),
        ("For development", f"pipx install git+https://{REPOSITORY}.git"),
    ]


def test_the_readme_sends_a_person_to_their_deployments_setup_card() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    install = readme[readme.index("## Install") : readme.index("### Which platform")]

    assert "setup card" in install
    assert f"{REPOSITORY}.git@<revision>" in install
    assert "### For development" in install


def test_computer_use_md_adds_the_extra_the_way_the_shim_says() -> None:
    document = (ROOT / "docs" / "COMPUTER_USE.md").read_text(encoding="utf-8")

    assert COMPUTER_USE_INSTALL in document
    assert "pipx install --force" not in document
