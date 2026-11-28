"""The runtime's half of the shell contract (see desktop/src/runtime.js).

stdout carries one JSON object per line and nothing else, so every child
service logs to a file and the runtime's own diagnostics go to stderr.
"""

from __future__ import annotations

import contextlib
import json
import logging
import sys
import threading

_lock = threading.Lock()

logger = logging.getLogger("autogpt_desktop")


def progress(step: str, message: str) -> None:
    logger.info(message)
    _emit({"event": "progress", "step": step, "message": message})


def stopping(message: str, grace_seconds: int) -> None:
    """Ask the shell for more time than it allows a stop by default."""
    logger.info(message)
    _emit(
        {
            "event": "progress",
            "step": "stopping",
            "message": message,
            "grace_seconds": grace_seconds,
        }
    )


def cancellation_wait(seconds: float, outcome: str) -> None:
    _emit(
        {
            "event": "shutdown",
            "phase": "cancellation",
            "seconds": seconds,
            "outcome": outcome,
        }
    )


def activity(graph_runs: int, copilot_turns: int) -> None:
    """How many agent runs and AutoPilot turns are in progress, whenever
    that changes. The shell asks before it quits with any (src/quit.js)."""
    _emit(
        {"event": "activity", "graph_runs": graph_runs, "copilot_turns": copilot_turns}
    )


def ready(url: str) -> None:
    logger.info(f"AutoGPT is ready at {url}")
    _emit({"event": "ready", "url": url})


def timing(phase: str, seconds: float) -> None:
    """How long one phase of the start took (timing.py). The shell shows
    nothing for it; it is for the log and for build/smoke_test.py."""
    logger.info(f"{phase}: {seconds:.1f} s")
    _emit({"event": "timing", "phase": phase, "seconds": round(seconds, 2)})


def started(message: str, seconds: float, phases: dict[str, float]) -> None:
    """The whole start, from the runtime beginning it to the app answering."""
    logger.info(message)
    _emit(
        {
            "event": "timing",
            "phase": "total",
            "seconds": round(seconds, 2),
            "phases": {name: round(spent, 2) for name, spent in phases.items()},
        }
    )


def claude_code(
    state: str, message: str, cli: str | None, version: str | None, bundled: bool
) -> None:
    """Whether AutoPilot runs on the user's Claude Code sign-in
    (claude_code.py). The shell words its status line from `state`; nothing
    here names the account."""
    logger.info(message)
    _emit(
        {
            "event": "claude_code",
            "state": state,
            "message": message,
            "cli": cli,
            "version": version,
            "bundled": bundled,
        }
    )


def error(message: str, *, fatal: bool) -> None:
    logger.error(message)
    _emit({"event": "error", "message": message, "fatal": fatal})


def _emit(payload: dict[str, object]) -> None:
    line = json.dumps(payload, separators=(",", ":"))
    # A shell that died took the pipe with it. That must not keep the runtime
    # from shutting its services down, which is what it does next.
    with _lock, contextlib.suppress(OSError, ValueError):
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def configure_logging() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s [desktop] %(levelname)s %(message)s",
    )
