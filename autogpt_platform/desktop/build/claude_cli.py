"""The Claude Code CLI in the bundle is the one the locked SDK ships.

The backend's Docker image gets the CLI from one place: inside the
claude-agent-sdk wheel that backend/poetry.lock pins. The desktop bundle
installs the same locked SDK on every platform, and so carries the same CLI,
wherever that wheel exists.

Anthropic does not publish every SDK version for every platform (0.2.160 to
0.2.163 have no Windows wheel; 0.2.158 and 0.2.159 do). Where the locked
version has no wheel, the SDK installs from its source distribution, which
is the same Python code without the program. The build then puts the program
where the wheel would have: the official build of exactly the version that
SDK names (`_cli_version.py`), from Anthropic's release bucket, pinned by
digest in artifacts.py, byte for byte as published.

A bump of the lock that changes the CLI version stops the build until
artifacts.py names the new one: a bundle never carries a CLI other than the
one its SDK was built with. It stops on every platform, also where the wheel
brought the right program and nothing has to be downloaded, so that the pin
cannot go stale on the platforms that do not use it.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from artifacts import Artifact
from fetch import download

SDK = "claude-agent-sdk"
EXE = ".exe" if sys.platform == "win32" else ""
VERSION_TIMEOUT_SECONDS = 120


def ensure(bundled: Path, artifact: Artifact | None, pinned: str, cache: Path) -> Path:
    """Make sure the CLI is in `bundled` (claude_agent_sdk/_bundled) and is
    the version the installed SDK names. `artifact` is this platform's pinned
    download of version `pinned`, None where the wheel is relied on."""
    cli = bundled / f"claude{EXE}"
    declared = declared_version(bundled.parent)
    check_pin(pinned, declared)
    if cli.is_file() and reported_version(cli, cache) == declared:
        return cli
    had = (
        f"one that reports {reported_version(cli, cache)}" if cli.is_file() else "none"
    )
    if artifact is None:
        raise RuntimeError(
            f"the installed {SDK} names Claude Code {declared} and brought {had}. Its wheel "
            "for this platform is missing or changed; pin the official build of that version "
            "for this platform in build/artifacts.py ('claude-cli')."
        )
    bundled.mkdir(parents=True, exist_ok=True)
    shutil.copy2(download(artifact, cache, name=f"claude-{pinned}{EXE}"), cli)
    cli.chmod(0o755)
    return cli


def check_pin(pinned: str, declared: str) -> None:
    """artifacts.CLAUDE_CLI_VERSION names the CLI the locked SDK is built
    with, whether or not this platform downloads it."""
    if pinned != declared:
        raise RuntimeError(
            f"the locked {SDK} was built with Claude Code {declared}, and build/artifacts.py "
            f"pins {pinned}. Set CLAUDE_CLI_VERSION to {declared} and each 'claude-cli' sha256 "
            "to the one in that release's manifest.json "
            f"(.../claude-code-releases/{declared}/manifest.json)."
        )


def check(bundled: Path, lock: Path, cache: Path, pinned: str) -> None:
    """Refuse a bundle whose SDK is not the locked one, whose CLI is not the
    one that SDK names, or for which artifacts.py pins another (`pinned`)."""
    sdk = bundled.parent
    locked, installed = locked_sdk_version(lock), installed_sdk_version(sdk)
    if installed != locked:
        raise RuntimeError(
            f"the bundle has {SDK} {installed}; backend/poetry.lock pins {locked}. "
            "Run the deps step."
        )
    cli = bundled / f"claude{EXE}"
    if not cli.is_file():
        raise RuntimeError(
            f"the bundle has no Claude Code CLI at {cli}; run the deps step"
        )
    declared, reported = declared_version(sdk), reported_version(cli, cache)
    check_pin(pinned, declared)
    if reported != declared:
        raise RuntimeError(
            f"{cli} reports version {reported}; the bundled {SDK} {installed} was built with "
            f"{declared}. Run the deps step."
        )


def declared_version(sdk: Path) -> str:
    """The CLI version the installed SDK was built with."""
    source = sdk / "_cli_version.py"
    try:
        match = re.search(
            r'^__cli_version__\s*=\s*"([^"]+)"',
            source.read_text(encoding="utf-8"),
            re.M,
        )
    except OSError:
        match = None
    if not match:
        raise RuntimeError(
            f"{source} no longer says which Claude Code CLI the SDK is built with "
            "(__cli_version__). Find where it says so now and update build/claude_cli.py and "
            "runtime/autogpt_desktop/claude_code.py."
        )
    return match.group(1)


def installed_sdk_version(sdk: Path) -> str:
    name = SDK.replace("-", "_")
    found = [path.name for path in sdk.parent.glob(f"{name}-*.dist-info")]
    if len(found) != 1:
        raise RuntimeError(
            f"expected one {name}-*.dist-info beside {sdk}, found {found}"
        )
    return found[0].removeprefix(f"{name}-").removesuffix(".dist-info")


def locked_sdk_version(lock: Path) -> str:
    packages = tomllib.loads(lock.read_text(encoding="utf-8"))["package"]
    versions = [package["version"] for package in packages if package["name"] == SDK]
    if len(versions) != 1:
        raise RuntimeError(f"{lock} pins {SDK} {len(versions)} times, not once")
    return versions[0]


def reported_version(cli: Path, cache: Path) -> str | None:
    """What `claude --version` says, with a configuration directory of its
    own: a build must neither read nor write the builder's Claude Code setup."""
    config = cache / "claude-config"
    config.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CLAUDE_CONFIG_DIR": str(config), "DISABLE_AUTOUPDATER": "1"}
    try:
        result = subprocess.run(
            [str(cli), "--version"],
            env=env,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=VERSION_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"\d+\.\d+\.\d+", result.stdout)
    return match.group(0) if match and result.returncode == 0 else None
