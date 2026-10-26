"""Warm the Claude CLI binary once per executor process, off the turn path.

On a fresh pod the first spawn of the ~250 MB bundled binary is slow (about
7 s, against 0.6 s on a warm pod) because the binary is still being read from
the container image. ``claude -v`` alone doesn't read all of it, so the
warm-up reads it through once, in a background thread started with the
executor. No turn waits for it. ``claude -v`` runs with the parent variables
every launch keeps (``inherited_env``), not the backend's whole environment.
"""

import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable
from typing import cast

from backend.copilot.sdk.launch_env import inherited_env

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 8 * 1024 * 1024
_VERSION_TIMEOUT_SECONDS = 30


def start_cli_warmup() -> threading.Thread:
    """Warm the CLI in a daemon thread, so startup doesn't wait for it."""
    thread = threading.Thread(target=warm_cli, name="copilot-cli-warmup", daemon=True)
    thread.start()
    return thread


def warm_cli(cli_path: str | None = None) -> None:
    """Read the CLI binary through once, then run ``claude -v``. Never raises."""
    try:
        path = cli_path or _resolve_cli_path()
        if not path:
            return
        # A read that never returns logs nothing more, so this line is what
        # tells a stuck warm-up from one that never started.
        logger.info("CLI warm-up started")
        started = time.monotonic()
        size = _read_through(path)
        result = subprocess.run(
            [path, "-v"],
            capture_output=True,
            timeout=_VERSION_TIMEOUT_SECONDS,
            env=inherited_env(),
        )
        done = (
            f"read {size / 1e6:.0f} MB and ran -v (rc={result.returncode}) "
            f"in {time.monotonic() - started:.1f}s: {os.path.basename(path)}"
        )
        if result.returncode == 0:
            logger.info(f"CLI warm-up {done}")
        else:
            logger.warning(f"CLI warm-up failed: {done}")
    except Exception as e:
        # The type only: an OSError's text carries the binary's full path.
        logger.warning(f"CLI warm-up skipped: {type(e).__name__}")


def _resolve_cli_path() -> str | None:
    """The binary turns will run: same precedence as ``ChatConfig``."""
    configured = os.getenv("CHAT_CLAUDE_AGENT_CLI_PATH") or os.getenv(
        "CLAUDE_AGENT_CLI_PATH"
    )
    if configured:
        return configured
    # Private SDK module, imported here so an SDK release that moves it only
    # skips the warm-up instead of failing the executor's import.
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    # _find_bundled_cli never touches self (a __file__-based lookup).
    finder = cast(
        Callable[[object], str | None], SubprocessCLITransport._find_bundled_cli
    )
    return finder(None)


def _read_through(path: str) -> int:
    """Read the whole file, discarding it; return how many bytes it had."""
    size = 0
    with open(path, "rb") as binary:
        while chunk := binary.read(_CHUNK_BYTES):
            size += len(chunk)
    return size
