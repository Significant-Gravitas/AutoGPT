"""Cross-process helper: wait for a copilot session's turn to finish.

Used by the sub-AutoPilot tools (``run_sub_session``,
``get_sub_session_result``) now that sub-AutoPilots run as regular
copilot_executor queue jobs — one of the executor workers picks up the
enqueued ``CoPilotExecutionEntry`` and streams events through the shared
``stream_registry``. The parent tool waits by subscribing to the same
stream and returning as soon as a terminal event (``StreamFinish`` /
``StreamError``) arrives or the cap fires.

Mirrors ``tools/execution_utils.wait_for_execution`` which does the
equivalent for graph executions via the executor event bus.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from backend.copilot import stream_registry
from backend.copilot.response_model import StreamError, StreamFinish

logger = logging.getLogger(__name__)


SessionOutcome = Literal["completed", "failed", "running"]


async def wait_for_session_completion(
    *,
    session_id: str,
    user_id: str | None,
    timeout: float,
) -> SessionOutcome:
    """Block up to *timeout* seconds for a copilot turn to finish.

    Subscribes to ``stream_registry`` for the session and drains events
    until a terminal one arrives:

    * ``StreamFinish`` → ``"completed"`` (the worker wrote its final
      assistant message; callers can now read the session's last message
      for the content).
    * ``StreamError`` → ``"failed"``.

    Anything else (text deltas, tool events, heartbeats) is ignored.

    On timeout, or when the session meta isn't visible yet (the worker
    may not have claimed the job), returns ``"running"`` so the caller
    can poll again with a fresh cap. The enqueued job is not touched —
    it keeps running on whichever worker picked it up.
    """
    queue = await stream_registry.subscribe_to_session(
        session_id=session_id,
        user_id=user_id,
    )
    if queue is None:
        # Session meta not in Redis yet, or the caller doesn't own it.
        # ``subscribe_to_session`` already retried with backoff for the
        # "not yet created" case before returning None.
        return "running"

    # ``subscribe_to_session`` spawned a background XREAD listener keyed by
    # this queue — the finally unsubscribes on every exit path (cap fired,
    # terminal event, caller cancellation) so we don't leak listener tasks
    # / Redis connections across repeated polls (sentry r3105348640).
    try:
        deadline = asyncio.get_event_loop().time() + max(timeout, 0)
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return "running"
            event = await asyncio.wait_for(queue.get(), timeout=remaining)
            if isinstance(event, StreamFinish):
                return "completed"
            if isinstance(event, StreamError):
                return "failed"
    except asyncio.TimeoutError:
        return "running"
    finally:
        await stream_registry.unsubscribe_from_session(
            session_id=session_id,
            subscriber_queue=queue,
        )
