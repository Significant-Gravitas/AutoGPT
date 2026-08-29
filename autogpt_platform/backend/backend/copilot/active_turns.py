"""Per-user concurrent AutoPilot turn tracking, backed entirely by Postgres.

Each :class:`prisma.models.ChatSession` carries a ``chatStatus`` text
enum: ``"idle"`` (no turn in flight, the 99% case), ``"queued"``
(waiting for a running slot to free), ``"running"`` (a turn is being
processed).  The cap and queue queries are both ``count`` / ``find_many``
on ``ChatSession`` by ``chatStatus``.

Public API
----------

* :func:`acquire_turn_slot` — async context manager. Counts the user's
  ``"running"`` sessions, raises :class:`ConcurrentTurnLimitError` at
  the cap, otherwise flips the session to ``"running"`` and yields a
  handle whose release transfers to ``mark_session_completed`` via
  :meth:`TurnSlot.keep`.
* :func:`release_turn_slot` — flips the session back to ``"idle"``.
  Called from ``mark_session_completed`` when the turn ends.
* :func:`count_running_turns` / :func:`get_running_session_ids` —
  used by the queue layer (in-flight = running + queued) and the
  dispatcher's busy-session check.

Cap admission is a *non-locked* count-then-update. Two concurrent
submits from the same user can both pass the count and both update,
leaving the user briefly one or two over the cap. This is the same
trade-off the graph-execution credit rate-limit accepts on its
``INCRBY`` path: the cap is a safeguard, not a budget.

DB access goes through :func:`backend.data.db_accessors.chat_db` so
the dispatcher works from both the HTTP server (Prisma directly) and
the copilot_executor process (RPC via DatabaseManager).
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from backend.copilot.model import (
    CHAT_STATUS_IDLE,
    CHAT_STATUS_QUEUED,
    CHAT_STATUS_RUNNING,
)
from backend.data.db_accessors import chat_db
from backend.util.settings import Settings

# Upper bound on a single AutoPilot turn's wall-clock duration.  Re-exported
# for callers (e.g. ``backend.blocks.autopilot``) that need a sensible
# upper-wait timeout.  Stale running sessions older than this are an
# operational concern surfaced via metrics + manual recovery, not
# enforced at read time.
MAX_TURN_LIFETIME_SECONDS = 6 * 60 * 60


def get_running_turn_limit() -> int:
    """Configured soft cap on concurrently *running* turns per user."""
    return Settings().config.max_running_copilot_turns_per_user


def get_inflight_turn_limit() -> int:
    """Configured hard cap on in-flight (running + queued) turns per user."""
    return Settings().config.max_inflight_copilot_turns_per_user


def inflight_turn_limit_message(limit: int | None = None) -> str:
    """User-facing 429 detail when the in-flight cap is hit."""
    resolved = get_inflight_turn_limit() if limit is None else limit
    return (
        f"You've reached the limit of {resolved} active tasks (running + queued). "
        "Please wait for one of your current tasks to finish before starting a new one."
    )


def running_turn_limit_message(limit: int | None = None) -> str:
    """Default :class:`ConcurrentTurnLimitError` detail when the
    *running* cap is hit on a path that does not queue (e.g.
    ``AutoPilotBlock``, ``run_sub_session``).  The HTTP route catches
    the error before it surfaces and replaces the message with the
    inflight one."""
    resolved = get_running_turn_limit() if limit is None else limit
    return (
        f"You have {resolved} AutoPilot tasks already running. "
        "Please wait for one of them to finish before starting a new one."
    )


def queued_turn_message() -> str:
    """User-facing message rendered when a turn is queued instead of
    starting immediately because the running cap is full."""
    return (
        "Your task has been queued and will start automatically when one of "
        "your current tasks finishes."
    )


class ConcurrentTurnLimitError(Exception):
    """User has reached the configured running AutoPilot turn cap."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or running_turn_limit_message())


async def count_running_turns(user_id: str) -> int:
    """User's current running-turn count."""
    return await chat_db().count_chat_sessions_by_status(
        user_id=user_id, status=CHAT_STATUS_RUNNING
    )


async def get_running_session_ids(user_id: str) -> set[str]:
    """Set of the user's session IDs currently running a turn."""
    rows = await chat_db().list_chat_sessions_by_status(
        user_id=user_id, status=CHAT_STATUS_RUNNING
    )
    return {r.session_id for r in rows}


async def release_turn_slot(user_id: str, session_id: str) -> None:
    """Flip the session back to ``"idle"``.  Idempotent — the CAS on
    ``chatStatus='running'`` is a no-op when the status has already
    changed (e.g. a parallel cancel)."""
    if not user_id:
        return
    await chat_db().update_chat_session_status(
        session_id=session_id,
        expect_status=CHAT_STATUS_RUNNING,
        status=CHAT_STATUS_IDLE,
        user_id=user_id,
    )


class TurnSlot:
    """Handle yielded by :func:`acquire_turn_slot`.

    Call :meth:`keep` once a turn has been successfully scheduled to
    transfer release ownership to ``mark_session_completed``.  Without
    ``keep``, the context manager auto-releases on exit — but only when
    *this* caller admitted the slot.
    """

    __slots__ = ("user_id", "session_id", "admitted", "_kept")

    def __init__(self, user_id: str, session_id: str) -> None:
        self.user_id = user_id
        self.session_id = session_id
        self.admitted = False
        self._kept = False

    def keep(self) -> None:
        """Transfer slot ownership out of this context."""
        self._kept = True


@asynccontextmanager
async def acquire_turn_slot(
    user_id: str | None,
    session_id: str,
    capacity: int | None = None,
) -> AsyncIterator[TurnSlot]:
    """Reserve a turn slot for the duration of the ``async with`` block.

    Three branches on entry:

    * **Admitted** — user below the cap; ``chatStatus`` flips to
      ``"running"``.  Exit auto-releases unless :meth:`TurnSlot.keep`
      was called.
    * **Refreshed** — same ``session_id`` is already ``"running"``
      (network retry, duplicate request); status stays as-is and this
      caller does NOT own the release.
    * **Rejected** — at the cap; raises :class:`ConcurrentTurnLimitError`.

    Anonymous sessions (``user_id`` falsy) bypass the cap entirely.
    """
    handle = TurnSlot(user_id or "", session_id)
    if not user_id:
        yield handle
        return

    resolved_capacity = capacity if capacity is not None else get_running_turn_limit()
    db = chat_db()

    # Try fresh admit: promote idle → running in one CAS-gated update.
    if await db.update_chat_session_status(
        session_id=session_id,
        expect_status=CHAT_STATUS_IDLE,
        status=CHAT_STATUS_RUNNING,
        user_id=user_id,
    ):
        # Fresh admit: enforce the cap by counting AFTER the flip.
        # Reading after-write is OK because over-admit just briefly
        # exceeds the cap — the user gets one extra slot at most under
        # burst, same trade-off as the prior count-then-update path.
        if await count_running_turns(user_id) > resolved_capacity:
            # Roll back our flip; the caller falls through to the queue.
            await release_turn_slot(user_id, session_id)
            raise ConcurrentTurnLimitError(
                running_turn_limit_message(resolved_capacity)
            )
        handle.admitted = True
    else:
        # CAS failed: session was not idle.  Disambiguate by reading
        # the current status — running (legitimate SSE-retry refresh)
        # vs queued (this user already has a pending task for this
        # session; the route must fall through to the queue path
        # instead of double-dispatching).
        current = await db.get_chat_session_status(session_id)
        if current == CHAT_STATUS_QUEUED:
            raise ConcurrentTurnLimitError(
                running_turn_limit_message(resolved_capacity)
            )
        # Any other state (running, or unexpectedly idle from a
        # parallel transition) is treated as refresh: no admit, no
        # release ownership, no error.  Caller's dispatch path is
        # idempotent on duplicate message_ids via the ChatMessage PK.

    try:
        yield handle
    finally:
        if handle.admitted and not handle._kept:
            await release_turn_slot(user_id, session_id)
