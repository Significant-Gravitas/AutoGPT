"""Process-scoped registry of running sub-AutoPilot tasks.

Sub-AutoPilot runs (via ``run_sub_session`` tool) are spawned as
``asyncio.Task`` instances that must outlive the originating HTTP request so
the user can close the tab, come back later, and retrieve the result. The
registry is therefore a **module-level dict**, not a per-context ContextVar —
entries survive stream teardown and any single client disconnect.

Each entry is keyed by a caller-opaque ``sub_session_id`` (``sub-<uuid>``).
Lookups verify the user_id so one user cannot inspect or cancel another's
tasks.

Finished entries (done/cancelled/errored) are retained briefly so the next
``get_sub_session_result`` call can still observe the terminal state, then
garbage-collected by :func:`prune_finished` which callers should invoke
opportunistically.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# Max wait for a single get_sub_session_result call. Short enough that the
# stream's 10-min idle timeout still makes sense; long enough to amortise the
# agent's polling cadence. Matches run_agent's wait_for_result / agent_output's
# wait_if_running caps for consistency.
MAX_SUB_SESSION_WAIT_SECONDS = 5 * 60  # 5 minutes

# Retention for terminal entries after they finish. Gives the requesting agent
# a window to retrieve the result even if it's late to poll.
_TERMINAL_TTL_SECONDS = 30 * 60  # 30 minutes

# Process-wide registry. Survives stream teardown.
# sub_session_id → {
#     "task": asyncio.Task[...],
#     "user_id": str,
#     "parent_session_id": str,
#     "prompt_preview": str,
#     "started_at": float,
#     "finished_at": float | None,
# }
_sub_sessions: dict[str, dict[str, Any]] = {}


def register_sub_session(
    task: asyncio.Task,
    user_id: str,
    parent_session_id: str,
    prompt: str,
    inner_session_id: str,
) -> str:
    """Register *task* and return a fresh ``sub_session_id``.

    ``inner_session_id`` is the sub-AutoPilot's own ChatSession id — stored
    so ``get_sub_session_result`` can peek at the running conversation to
    report progress while the task is still in flight. The caller remains
    responsible for scheduling the task on the running event loop; this
    function only stores the reference so the task survives request
    teardown and is retrievable later.
    """
    sub_session_id = f"sub-{uuid.uuid4().hex[:12]}"
    _sub_sessions[sub_session_id] = {
        "task": task,
        "user_id": user_id,
        "parent_session_id": parent_session_id,
        "inner_session_id": inner_session_id,
        "prompt_preview": prompt[:200],
        "started_at": time.monotonic(),
        "finished_at": None,
    }
    # Record finished_at when the task completes so prune_finished can
    # evict it after _TERMINAL_TTL_SECONDS.
    task.add_done_callback(lambda _t, sid=sub_session_id: _mark_finished(sid))
    logger.info(
        "Registered sub-session %s for user %s (parent session=%s)",
        sub_session_id,
        user_id,
        parent_session_id,
    )
    return sub_session_id


def get_sub_session(sub_session_id: str, user_id: str) -> dict[str, Any] | None:
    """Return the entry for *sub_session_id* if it belongs to *user_id*.

    Returns ``None`` when the entry doesn't exist or belongs to someone else
    (caller treats both as "not found" to avoid leaking existence).
    """
    entry = _sub_sessions.get(sub_session_id)
    if entry is None:
        return None
    if entry["user_id"] != user_id:
        logger.warning(
            "User %s attempted to access sub-session %s owned by %s",
            user_id,
            sub_session_id,
            entry["user_id"],
        )
        return None
    return entry


def unregister_sub_session(sub_session_id: str) -> None:
    """Drop the entry for *sub_session_id*. Idempotent."""
    _sub_sessions.pop(sub_session_id, None)


def cancel_sub_session(sub_session_id: str, user_id: str) -> bool:
    """Cancel the task and drop the entry. Returns True if cancelled."""
    entry = get_sub_session(sub_session_id, user_id)
    if entry is None:
        return False
    task: asyncio.Task = entry["task"]
    if not task.done():
        task.cancel()
    unregister_sub_session(sub_session_id)
    logger.info("Cancelled sub-session %s by user %s", sub_session_id, user_id)
    return True


def prune_finished(now: float | None = None) -> int:
    """Evict finished entries older than ``_TERMINAL_TTL_SECONDS``.

    Returns the number evicted. Called opportunistically from the tool
    handlers; not driven by a timer.
    """
    if now is None:
        now = time.monotonic()
    stale = [
        sid
        for sid, e in _sub_sessions.items()
        if e["finished_at"] is not None
        and now - e["finished_at"] >= _TERMINAL_TTL_SECONDS
    ]
    for sid in stale:
        _sub_sessions.pop(sid, None)
    if stale:
        logger.info("Pruned %d terminal sub-session entries", len(stale))
    return len(stale)


def _mark_finished(sub_session_id: str) -> None:
    """Record the time a task completed so prune_finished can evict it later."""
    entry = _sub_sessions.get(sub_session_id)
    if entry is not None and entry["finished_at"] is None:
        entry["finished_at"] = time.monotonic()


# Test-only helper — production code should not need to reset the registry.
def _reset_for_test() -> None:
    _sub_sessions.clear()
