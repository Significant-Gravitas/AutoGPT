"""Lazy auto-registration of the per-user morning-briefing schedule.

Single-job counterpart to ``backend/copilot/dream/scheduling.py``. That
module's registry-of-crons shape earns its weight over two-plus jobs; the
briefing system has exactly one (``morning_briefing_{user_id}``, daily
user-local), so this module inlines the same registration logic without
the registry indirection.

**Timezone drift handling.** APScheduler binds the cron trigger to the
timezone at job-creation time — a later ``User.timezone`` change silently
leaves the cron firing at the old local time. Two mechanisms cover that:
``update_user_timezone`` clears the Redis marker so the very next call
re-registers, and the marker's TTL bounds how long a drift arriving by any
other route can persist. The scheduler side compares the stored trigger
timezone and only rewrites the job when it actually differs — re-adding an
existing cron would reset a pending ``next_run_time`` and drop that day's
briefing.

**Hot-path cost.** This helper runs on every chat turn. The LaunchDarkly
gate runs first (in-process cached), then the Redis marker short-circuits
the rest. Only the first call after the marker expires pays for the
timezone lookup (a DB read) and the scheduler RPC.

Failures are logged at WARN and swallowed — ``ensure_morning_briefing_scheduled``
is fired via ``asyncio.create_task`` from the hot chat-stream request path
and must never raise.
"""

from __future__ import annotations

import logging

from backend.data.db_accessors import user_db
from backend.data.model import USER_TIMEZONE_NOT_SET
from backend.data.redis_client import get_redis_async
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)

# How long a registration marker suppresses re-checking. Mirrors
# ``dream/scheduling.py``: a timezone change goes through
# ``update_user_timezone``, which clears the marker outright, so the TTL only
# has to bound out-of-band drift (an externally deleted cron, a tz written
# straight to the DB) rather than carry drift detection on its own.
REGISTRATION_TTL_SECONDS = 7 * 24 * 3600

BRIEFING_REGISTRATION_PREFIX = "morning_briefing_registered"


async def _resolve_user_timezone(user_id: str) -> str | None:
    """Look up the user's IANA timezone from Postgres.

    Returns ``"UTC"`` only when the answer is authoritative (user missing
    or timezone genuinely unset) and ``None`` when the lookup itself
    failed — a transient DB blip is "unknown", not "UTC", and must never
    silently re-register the user's local-time cron onto UTC.

    Routes through the ``user_db()`` accessor, NOT ``User.prisma()``: this
    runs in the scheduler process, which never connects a local Prisma
    client. A direct Prisma call raises ``ClientNotConnectedError`` on
    every invocation there. The accessor falls back to the
    DatabaseManager RPC in Prisma-less processes (same pattern as
    ``dream/scheduling.py``).
    """
    try:
        try:
            user = await user_db().get_user_by_id(user_id)
        except ValueError:
            # Authoritative: the user row doesn't exist.
            return "UTC"
        tz = (user.timezone or "").strip()
        if not tz or tz == USER_TIMEZONE_NOT_SET:
            return "UTC"
        return tz
    except Exception:
        logger.warning(
            "Could not resolve timezone for user %s; leaving existing "
            "morning-briefing schedule untouched this cycle",
            user_id[:12],
            exc_info=True,
        )
        return None


async def _read_registration_marker(user_id: str) -> str | None:
    """Read the freshness marker for this user's briefing cron.

    The key is written with ``REGISTRATION_TTL_SECONDS``, so its mere
    presence means "registered recently enough, don't re-check". Its value
    is the timezone the cron was last registered with, kept for debugging
    and log context; drift itself is decided scheduler-side, by comparing
    the live job's trigger timezone (see
    ``Scheduler.add_morning_briefing_schedule``).

    Returns:
      * The stored timezone string when the key exists.
      * ``None`` when the key is missing OR Redis is unavailable. The
        caller treats both as "needs registration" — the scheduler no-ops
        when the existing job already matches, so a redundant call is cheap
        and, crucially, can't reset a pending ``next_run_time``.
    """
    try:
        redis = await get_redis_async()
        key = f"{BRIEFING_REGISTRATION_PREFIX}:{user_id}"
        stored = await redis.get(key)
        if stored is None:
            return None
        if isinstance(stored, bytes):
            return stored.decode("utf-8", errors="replace")
        return str(stored)
    except Exception:
        logger.debug(
            "Redis read failed for %s:%s; treating as not-registered",
            BRIEFING_REGISTRATION_PREFIX,
            user_id[:12],
            exc_info=True,
        )
        return None


async def _write_registration_marker(user_id: str, current_tz: str) -> None:
    """Persist the freshness marker, valued with the timezone just used.

    Best-effort — a Redis write failure means the next call will see the
    key as missing and force a redundant re-register, which the scheduler
    turns into a no-op when nothing changed.
    """
    try:
        redis = await get_redis_async()
        key = f"{BRIEFING_REGISTRATION_PREFIX}:{user_id}"
        await redis.set(key, current_tz, ex=REGISTRATION_TTL_SECONDS)
    except Exception:
        logger.debug(
            "Redis write failed for %s:%s; lazy path will re-detect later",
            BRIEFING_REGISTRATION_PREFIX,
            user_id[:12],
            exc_info=True,
        )


async def clear_briefing_registration_marker(user_id: str) -> None:
    """Delete the Redis registration marker for the briefing cron.

    Called from ``update_user_timezone`` so a profile change immediately
    re-opens lazy registration instead of leaving the marker to block it
    for the remainder of its TTL. Single-key DEL so it routes on
    Redis Cluster.

    Best-effort — on Redis failure the marker simply expires via TTL.
    """
    try:
        redis = await get_redis_async()
        await redis.delete(f"{BRIEFING_REGISTRATION_PREFIX}:{user_id}")
    except Exception:
        logger.warning(
            "Redis delete failed for %s:%s; marker will expire via TTL",
            BRIEFING_REGISTRATION_PREFIX,
            user_id[:12],
            exc_info=True,
        )


async def ensure_morning_briefing_scheduled(user_id: str) -> None:
    """Idempotently register the morning-briefing cron for a user.

    Fire-and-forget callable from two trigger points:

    * **Lazy path** — ``stream_chat_post`` fires this after resolving the
      authenticated user on every turn. The Redis marker short-circuits
      the common case, so only the first turn after the marker expires
      re-resolves the timezone and re-registers.
    * **Eager path** — ``update_user_timezone`` clears the marker first
      (see :func:`clear_briefing_registration_marker`) then calls this,
      so a profile change re-registers within a single call instead of
      waiting for the dedup key's TTL to expire.

    Never raises — swallows and logs every failure so a bad call can
    never break the caller's hot path.
    """
    try:
        if not user_id:
            return
        if not await is_feature_enabled(Flag.HIRE_EXPERTS, user_id, default=False):
            return

        # Marker next: one Redis GET is the rest of the cost of the common
        # case. Resolving the timezone is a DB read, and this runs on
        # every chat turn, so it must stay behind this gate.
        if await _read_registration_marker(user_id) is not None:
            return

        tz = await _resolve_user_timezone(user_id)
        if tz is None:
            # Lookup failed — "unknown" is not "UTC". Re-registering would
            # silently rebind the user's local-time cron to UTC; leave the
            # existing cron and stored tz untouched until a later call
            # resolves the real timezone.
            return

        # Lazy client handle, mirroring dream/scheduling.py: a flag-off
        # user never even constructs the scheduler client, and the lazy
        # import avoids a circular import during process bootstrap.
        from backend.util.clients import get_scheduler_client

        await get_scheduler_client().add_morning_briefing_schedule(
            user_id=user_id, user_timezone=tz
        )
        await _write_registration_marker(user_id, tz)
        logger.info(
            "Morning briefing: registered cron for user %s (tz=%s)",
            user_id[:12],
            tz,
        )
    except Exception:
        logger.warning(
            "Morning briefing: failed to register cron for user %s",
            user_id[:12],
            exc_info=True,
        )
