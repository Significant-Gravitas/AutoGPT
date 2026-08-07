"""Lazy auto-registration of the per-user morning-briefing schedule.

Single-job counterpart to ``backend/copilot/dream/scheduling.py``. That
module's registry-of-crons shape earns its weight over two-plus jobs; the
briefing system has exactly one (``morning_briefing_{user_id}``, daily
user-local), so this module inlines the same registration logic without
the registry indirection.

**Timezone drift handling.** APScheduler binds the cron trigger to the
timezone at job-creation time — a later ``User.timezone`` change silently
leaves the cron firing at the old local time. The Redis dedup key stores
the timezone the cron was registered with; every call compares the stored
value to the user's current timezone and re-registers on mismatch via the
scheduler's ``replace_existing=True``.

Failures are logged at WARN and swallowed — ``ensure_morning_briefing_scheduled``
is fired via ``asyncio.create_task`` from the hot chat-stream request path
and must never raise.
"""

from __future__ import annotations

import logging

from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)

# Matches dream's TTL rationale: long enough that a lazy re-check every few
# days is enough to catch drift, short enough that an out-of-band cron
# deletion self-heals within a bounded window.
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
        from backend.data.db_accessors import user_db
        from backend.data.model import USER_TIMEZONE_NOT_SET

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


async def _read_registration_tz(user_id: str) -> str | None:
    """Read the timezone the briefing cron was last registered with.

    Returns:
      * The stored timezone string when the key exists.
      * ``None`` when the key is missing OR Redis is unavailable. The
        caller treats both as "needs registration" — scheduler-side
        ``replace_existing=True`` makes a redundant call a cheap no-op.
    """
    try:
        from backend.data.redis_client import get_redis_async

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


async def _write_registration_tz(user_id: str, current_tz: str) -> None:
    """Persist the timezone we just registered the cron with.

    Best-effort — a Redis write failure means the next call will see the
    key as missing and force a redundant re-register (cheap via
    ``replace_existing=True``).
    """
    try:
        from backend.data.redis_client import get_redis_async

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
    for the remainder of its 7-day TTL. Single-key DEL so it routes on
    Redis Cluster.

    Best-effort — on Redis failure the marker simply expires via TTL.
    """
    try:
        from backend.data.redis_client import get_redis_async

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
      authenticated user on every turn. Drift-detects timezone changes
      via the Redis stored value and re-registers when the user's
      current timezone differs from the stored one.
    * **Eager path** — ``update_user_timezone`` clears the marker first
      (see :func:`clear_briefing_registration_marker`) then calls this,
      so a profile change re-registers within a single call instead of
      waiting for the dedup key's 7-day TTL to expire.

    Never raises — swallows and logs every failure so a bad call can
    never break the caller's hot path.
    """
    try:
        if not user_id:
            return
        if not await is_feature_enabled(Flag.MORNING_BRIEFING, user_id, default=False):
            return

        tz = await _resolve_user_timezone(user_id)
        if tz is None:
            # Lookup failed — "unknown" is not "UTC". Re-registering would
            # silently rebind the user's local-time cron to UTC; leave the
            # existing cron and stored tz untouched until a later call
            # resolves the real timezone.
            return

        if await _read_registration_tz(user_id) == tz:
            # Same tz, still within TTL → no work.
            return

        # Lazy client handle, mirroring dream/scheduling.py: a flag-off
        # user never even constructs the scheduler client, and the lazy
        # import avoids a circular import during process bootstrap.
        from backend.util.clients import get_scheduler_client

        await get_scheduler_client().add_morning_briefing_schedule(
            user_id=user_id, user_timezone=tz
        )
        await _write_registration_tz(user_id, tz)
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
