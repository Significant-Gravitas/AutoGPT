"""Warm the Claude CLI binary once per executor process, off the turn path.

On a fresh pod the first CLI spawn took about 7 s on a preview, against
0.6 s on a warm one: the ~250 MB binary is still being read from the
container image. Reading it through once when the executor starts, in a
background thread, moves that cost off the first turn.

This used to be a ``claude -v`` run in each worker thread's initializer. That
made every thread's first turn wait for it, and ``-v`` doesn't read the
whole binary, so the first real spawn stayed slow.
"""

import logging
import os
import subprocess
import threading
import time
from collections.abc import Callable
from typing import cast

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 8 * 1024 * 1024


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
        started = time.monotonic()
        size = _read_through(path)
        result = subprocess.run([path, "-v"], capture_output=True, timeout=30)
        logger.info(
            f"CLI warm-up read {size / 1e6:.0f} MB and ran -v "
            f"(rc={result.returncode}) in {time.monotonic() - started:.1f}s: {path}"
        )
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
