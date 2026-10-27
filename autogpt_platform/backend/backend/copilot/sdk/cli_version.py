"""The oldest Claude Code CLI the copilot hands chat text to."""

import functools
import logging
import os
import re
import subprocess
from pathlib import Path

from backend.copilot.sdk.launch_env import inherited_env

logger = logging.getLogger(__name__)

# First CLI release that honours ``client_composed`` on a user message, which
# ``ClaudeAgentOptions.verbatim_prompts`` sets. Older releases ignore the field
# without an error, so the version is checked here instead.
MIN_CLI_VERSION = (2, 1, 248)

_VERSION_PROBE_TIMEOUT_SECONDS = 10
# The whole first line of ``claude -v``: a release version, optionally followed
# by the product name. Anything else is refused.
_VERSION_LINE_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)(?: \(Claude Code\))?")
# Versions read so far, by the identity of the file they were read from: a
# binary replaced in place is read again. A failed read is never stored.
_VERSIONS: dict[tuple[str, int, int, int, int], tuple[int, ...]] = {}


def require_verbatim_prompts_cli(cli_path: str | Path | None) -> str:
    """Return the resolved path of the CLI to start, or raise.

    *cli_path* is ``ClaudeAgentOptions.cli_path``; when it is unset, the CLI
    the SDK finds on its own is checked. Callers start the returned path, so
    the binary that runs is the one whose version was read. Each refusal is
    logged once, with the binary's path; the raised message names only the
    file.
    """
    path = os.path.realpath(cli_path if cli_path else _cli_the_sdk_finds())
    try:
        version = _cli_version(path)
    except RuntimeError as e:
        logger.error("Refusing to start the Claude Code CLI at %s: %s", path, e)
        raise
    if version >= MIN_CLI_VERSION:
        return path
    logger.error(
        "Refusing to start the Claude Code CLI at %s: version %s is older than %s",
        path,
        _dotted(version),
        _dotted(MIN_CLI_VERSION),
    )
    raise RuntimeError(
        f"The Claude Code CLI ({os.path.basename(path)} {_dotted(version)}) is "
        f"older than {_dotted(MIN_CLI_VERSION)}, the first release that hands "
        "chat text to the model exactly as written. Install a newer one."
    )


def parse_cli_version(output: str) -> tuple[int, ...] | None:
    """The version in the output of ``claude -v``, or None when its first line
    is not exactly ``X.Y.Z``, optionally followed by `` (Claude Code)``."""
    first_line = output.strip().split("\n", 1)[0].rstrip()
    match = _VERSION_LINE_RE.fullmatch(first_line)
    return tuple(int(part) for part in match.groups()) if match else None


@functools.cache
def _cli_the_sdk_finds() -> str:
    # Imported here so that loading the block list, which imports this module
    # through the orchestrator block, does not load the SDK.
    from claude_agent_sdk import ClaudeAgentOptions
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    transport = SubprocessCLITransport(prompt="", options=ClaudeAgentOptions())
    return transport._find_cli()


def _cli_version(path: str) -> tuple[int, ...]:
    identity = _identity(path)
    if identity in _VERSIONS:
        return _VERSIONS[identity]
    version = _probe_version(path)
    if _identity(path) != identity:
        raise RuntimeError(
            f"The Claude Code CLI ({os.path.basename(path)}) changed while its "
            "version was read"
        )
    _VERSIONS[identity] = version
    return version


def _identity(path: str) -> tuple[str, int, int, int, int]:
    try:
        stat = os.stat(path)
    except OSError as e:
        raise RuntimeError(_cannot_run(path)) from e
    return (path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _probe_version(path: str) -> tuple[int, ...]:
    try:
        result = subprocess.run(
            [path, "-v"],
            stdin=subprocess.DEVNULL,
            env=inherited_env(),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=_VERSION_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(_cannot_run(path)) from e
    version = parse_cli_version(result.stdout or "")
    if result.returncode != 0 or version is None:
        raise RuntimeError(
            "Could not read the version of the Claude Code CLI "
            f"({os.path.basename(path)})"
        )
    return version


def _cannot_run(path: str) -> str:
    return (
        f"Could not run the Claude Code CLI ({os.path.basename(path)}) "
        "to read its version"
    )


def _dotted(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)
