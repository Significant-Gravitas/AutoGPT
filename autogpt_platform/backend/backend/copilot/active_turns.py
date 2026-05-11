"""Per-user concurrent AutoPilot turn tracking.

Caps how many copilot chat turns a single user can have running
concurrently so a single API key cannot spawn hundreds of simultaneous
turns and exhaust shared infrastructure.

This module is the **domain wrapper** over the generic
:func:`backend.data.redis_helpers.try_acquire_concurrency_slot` primitive
— it supplies the per-user pool keying, the cap-from-Settings lookup,
the user-facing error message, and the
:func:`acquire_turn_slot` context manager that drives the slot's
admit / release / refresh lifecycle.

Public API
----------

* :func:`acquire_turn_slot` — async context manager every entry point
  (HTTP route, ``run_sub_session`` tool, ``AutoPilotBlock``) wraps around
  the create-session + enqueue dance. Raises
  :class:`ConcurrentTurnLimitError` on rejection.
* :func:`release_turn_slot` — invoked by ``mark_session_completed``
  when a turn ends, freeing the slot for the next admission.
* :func:`get_concurrent_turn_limit` /
  :func:`concurrent_turn_limit_message` — operator-tunable cap and the
  matching user-facing 429 detail.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from redis.exceptions import RedisClusterException, RedisError

from backend.data.redis_client import get_redis_async
from backend.data.redis_helpers import SlotAdmission, try_acquire_concurrency_slot
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


# Upper bound on a single AutoPilot turn's wall-clock duration. Beyond
# this we treat the turn as abandoned: the slot is reclaimed by the
# stale-cutoff sweep (so a crashed turn doesn't hold a slot forever) and
# the :class:`AutoPilotBlock` execution wait gives up. Far exceeds typical
# chat turn duration (seconds-minutes) so legitimate long-running tool
# calls (E2B sandbox, deep web crawls, etc.) aren't penalised. The normal
# release path is ``mark_session_completed``; this is the safety net.
MAX_TURN_LIFETIME_SECONDS = 6 * 60 * 60

_USER_ACTIVE_TURNS_KEY_PREFIX = "copilot:user_active_turns:"


def get_concurrent_turn_limit() -> int:
    """Resolve the configured per-user concurrent-turn cap at call time.

    Reading at call time (rather than module load) lets operators retune
    the cap by editing the env-backed Settings without redeploying the
    code that imports this module.
    """
    return Settings().config.max_concurrent_copilot_turns_per_user


def concurrent_turn_limit_message(limit: int | None = None) -> str:
    """User-facing 429 detail string. Pass ``limit`` if you already
    resolved it; otherwise we read the configured value."""
    resolved = get_concurrent_turn_limit() if limit is None else limit
    return (
        f"You've reached the limit of {resolved} active tasks. Please wait "
        f"for one of your current tasks to finish before starting a new one."
    )


class ConcurrentTurnLimitError(Exception):
    """User has reached the configured concurrent in-flight AutoPilot
    turn cap. Maps to HTTP 429 in the API layer.
    """

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or concurrent_turn_limit_message())


def _user_pool_key(user_id: str) -> str:
    # Hash-tag braces ensure all keys for a single user co-locate on the
    # same Redis Cluster slot — required for any future Lua that touches
    # multiple per-user keys atomically.
    return f"{_USER_ACTIVE_TURNS_KEY_PREFIX}{{{user_id}}}"


async def _try_admit_user_turn(user_id: str, session_id: str) -> SlotAdmission:
    """Atomic admit/refresh against the user's active-turn pool.

    Fails open (returns ``ADMITTED``) on Redis errors so a brown-out
    doesn't 429 every user — the cap is a safeguard, not a budget.
    """
    try:
        redis = await get_redis_async()
        now = time.time()
        return await try_acquire_concurrency_slot(
            redis,
            pool_key=_user_pool_key(user_id),
            slot_id=session_id,
            capacity=get_concurrent_turn_limit(),
            score=now,
            stale_before_score=now - MAX_TURN_LIFETIME_SECONDS,
            ttl_seconds=MAX_TURN_LIFETIME_SECONDS,
        )
    except (RedisError, RedisClusterException, ConnectionError, OSError) as exc:
        logger.warning(
            "concurrent-turn cap: Redis unavailable for user=%s; failing open: %s",
            user_id,
            exc,
        )
        return SlotAdmission.ADMITTED


async def release_turn_slot(user_id: str, session_id: str) -> None:
    """Free ``user_id``'s slot for ``session_id``. Idempotent.

    Best-effort — a Redis error only delays release until the next
    stale-cutoff sweep.
    """
    try:
        redis = await get_redis_async()
        await redis.zrem(_user_pool_key(user_id), session_id)
    except (RedisError, RedisClusterException, ConnectionError, OSError) as exc:
        logger.warning(
            "release_turn_slot: Redis unavailable for user=%s session=%s: %s",
            user_id,
            session_id,
            exc,
        )


class TurnSlot:
    """Handle yielded by :func:`acquire_turn_slot`.

    Call :meth:`keep` once a turn has been successfully scheduled to
    transfer ownership to ``mark_session_completed`` (the release path).
    Without ``keep``, the context manager auto-releases on exit — but
    only when *this* caller admitted the slot. A re-entrant refresh
    leaves the slot alone, since some earlier caller still owns it.
    """

    __slots__ = ("user_id", "session_id", "admitted", "_kept")

    def __init__(self, user_id: str, session_id: str) -> None:
        self.user_id = user_id
        self.session_id = session_id
        self.admitted = False
        self._kept = False

    def keep(self) -> None:
        """Transfer slot ownership out of this context. Caller is now
        responsible for ensuring ``mark_session_completed`` releases the
        slot (or accepts the stale-cutoff fallback)."""
        self._kept = True


@asynccontextmanager
async def acquire_turn_slot(
    user_id: str | None,
    session_id: str,
) -> AsyncIterator[TurnSlot]:
    """Reserve a turn slot for the duration of the ``async with`` block.

    Three branches on entry:

    * **Admitted** — fresh slot acquired; ``keep()`` transfers ownership
      to ``mark_session_completed``, otherwise the slot is released on
      exit.
    * **Refreshed** — same-``session_id`` re-entry (network retry,
      duplicate request); the existing slot's score is bumped but this
      caller does NOT own its release. Exiting without ``keep`` is a
      no-op.
    * **Rejected** — pool is at the configured cap; raises
      :class:`ConcurrentTurnLimitError` (caller maps to HTTP 429).

    Anonymous sessions (``user_id`` falsy) bypass the gate entirely and
    yield a no-op handle.
    """
    handle = TurnSlot(user_id or "", session_id)
    if user_id:
        outcome = await _try_admit_user_turn(user_id, session_id)
        if outcome is SlotAdmission.REJECTED:
            raise ConcurrentTurnLimitError()
        if outcome is SlotAdmission.ADMITTED:
            handle.admitted = True

    try:
        yield handle
    finally:
        if handle.admitted and not handle._kept:
            await release_turn_slot(handle.user_id, handle.session_id)
