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
from dataclasses import dataclass, field

from backend.copilot.constants import MAX_TOOL_WAIT_SECONDS

logger = logging.getLogger(__name__)

# Max wait for a single get_sub_session_result call. Shared with every other
# long-running tool (run_agent, view_agent_output, run_block) so the stream
# idle timeout's 2× headroom holds uniformly.
MAX_SUB_SESSION_WAIT_SECONDS = MAX_TOOL_WAIT_SECONDS

# Retention for terminal entries after they finish. Gives the requesting agent
# a window to retrieve the result even if it's late to poll.
_TERMINAL_TTL_SECONDS = 30 * 60  # 30 minutes

# Absolute ceiling on how long a running sub-session may stay alive in the
# registry. A sub that the agent stopped polling (tab closed, session
# abandoned) still runs until it completes or trips this cap. Tuned well
# above any legitimate AutoPilot run so it only catches genuinely wedged
# tasks. 6h is arbitrary but > 99th percentile of real sub-AutoPilots.
_RUNNING_MAX_AGE_SECONDS = 6 * 60 * 60  # 6 hours


@dataclass
class SubSessionEntry:
    """A single registry entry — all per-sub state in one typed record."""

    task: asyncio.Task
    user_id: str
    parent_session_id: str
    inner_session_id: str
    prompt_preview: str
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None


# Process-wide registry. Survives stream teardown.
_sub_sessions: dict[str, SubSessionEntry] = {}


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
    _sub_sessions[sub_session_id] = SubSessionEntry(
        task=task,
        user_id=user_id,
        parent_session_id=parent_session_id,
        inner_session_id=inner_session_id,
        prompt_preview=prompt[:200],
    )
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


def get_sub_session(sub_session_id: str, user_id: str) -> SubSessionEntry | None:
    """Return the entry for *sub_session_id* if it belongs to *user_id*.

    Returns ``None`` when the entry doesn't exist or belongs to someone else
    (caller treats both as "not found" to avoid leaking existence).
    """
    entry = _sub_sessions.get(sub_session_id)
    if entry is None:
        return None
    if entry.user_id != user_id:
        logger.warning(
            "User %s attempted to access sub-session %s owned by %s",
            user_id,
            sub_session_id,
            entry.user_id,
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
    if not entry.task.done():
        entry.task.cancel()
    unregister_sub_session(sub_session_id)
    logger.info("Cancelled sub-session %s by user %s", sub_session_id, user_id)
    return True


def prune_finished(now: float | None = None) -> int:
    """Evict stale entries from the registry.

    Two kinds of stale:
      1. Terminal entries older than ``_TERMINAL_TTL_SECONDS`` — pruned
         silently (result was retrievable for the retention window).
      2. Running entries older than ``_RUNNING_MAX_AGE_SECONDS`` — cancelled
         AND evicted. Catches subs the agent stopped polling (tab closed
         mid-session) so they don't pile up in the process registry forever.

    Returns the total number of entries evicted. Called opportunistically
    from the tool handlers; not driven by a timer.
    """
    if now is None:
        now = time.monotonic()

    terminal_stale: list[str] = []
    running_stale: list[str] = []
    for sid, entry in _sub_sessions.items():
        if entry.finished_at is not None:
            if now - entry.finished_at >= _TERMINAL_TTL_SECONDS:
                terminal_stale.append(sid)
        elif now - entry.started_at >= _RUNNING_MAX_AGE_SECONDS:
            running_stale.append(sid)

    for sid in running_stale:
        entry = _sub_sessions.get(sid)
        if entry is None:
            continue
        if not entry.task.done():
            entry.task.cancel()
        _sub_sessions.pop(sid, None)
        logger.warning(
            "Cancelled abandoned sub-session %s (age=%.0fs)",
            sid,
            now - entry.started_at,
        )

    for sid in terminal_stale:
        _sub_sessions.pop(sid, None)

    total = len(terminal_stale) + len(running_stale)
    if total:
        logger.info(
            "Pruned sub-session registry: %d terminal, %d abandoned",
            len(terminal_stale),
            len(running_stale),
        )
    return total


def _mark_finished(sub_session_id: str) -> None:
    """Record the time a task completed so prune_finished can evict it later."""
    entry = _sub_sessions.get(sub_session_id)
    if entry is not None and entry.finished_at is None:
        entry.finished_at = time.monotonic()


async def notify_shutdown_and_cancel_all(reason: str) -> int:
    """Graceful-shutdown hook: cancel every running sub-session and write a
    visible error marker into each sub's ChatSession.

    Called from the copilot_executor cleanup path when the worker is going
    down. Without this, a user who reopens a sub's conversation would see
    a stalled turn with no explanation — the sub's task was running in the
    dying process, so its final result was never written.

    The error marker uses the retryable prefix so the frontend renders a
    "Try Again" button — users typically just want to resume the sub-
    AutoPilot with the same prompt.

    Returns the number of running sub-sessions that were notified. Safe to
    call when the registry is empty (returns 0).
    """
    # Delayed imports: model imports sdk.service, which imports tools, which
    # imports this module — a module-load cycle. Importing here breaks it.
    from backend.copilot.constants import COPILOT_RETRYABLE_ERROR_PREFIX
    from backend.copilot.model import ChatMessage, get_chat_session, upsert_chat_session

    # Sub-AutoPilot tasks are pinned to the worker-loop that spawned them.
    # If notify_shutdown is called per-worker (copilot_executor cleanup),
    # filter to entries owned by the current loop so we don't try to cancel
    # tasks from a different worker's loop (which would race that worker's
    # own cleanup).
    try:
        current_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    running: list[tuple[str, SubSessionEntry]] = []
    for sid, entry in _sub_sessions.items():
        if entry.finished_at is not None or entry.task.done():
            continue
        if current_loop is not None and entry.task.get_loop() is not current_loop:
            continue
        running.append((sid, entry))
    if not running:
        return 0

    logger.warning(
        "notify_shutdown: terminating %d running sub-session(s): %s",
        len(running),
        reason,
    )

    display_msg = (
        "Sub-AutoPilot was terminated because its host process shut down "
        f"({reason}). The last user message is preserved — click retry to "
        "resume."
    )

    notified = 0
    for sid, entry in running:
        if not entry.task.done():
            entry.task.cancel()
        try:
            inner = await get_chat_session(entry.inner_session_id)
            if inner is None:
                continue
            inner.messages.append(
                ChatMessage(
                    role="assistant",
                    content=f"{COPILOT_RETRYABLE_ERROR_PREFIX} {display_msg}",
                )
            )
            await upsert_chat_session(inner)
            notified += 1
        except Exception as exc:  # best-effort on shutdown
            logger.error(
                "notify_shutdown: failed to mark sub %s (inner=%s): %s",
                sid,
                entry.inner_session_id,
                exc,
            )
        finally:
            _sub_sessions.pop(sid, None)
    return notified


# Test-only helper — production code should not need to reset the registry.
def _reset_for_test() -> None:
    _sub_sessions.clear()
