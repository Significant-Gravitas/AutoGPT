"""Lazy auto-registration of per-user community rebuild schedules.

Called from the Graphiti ingestion path so a user gets a weekly
community rebuild cron registered exactly when they start using memory.
No wasted schedules for users who never write a memory; no startup
sweep that scales O(users) on every backend boot.

Two layers of idempotency:

1. **Redis SETNX with 7-day TTL** — fast path that avoids the
   scheduler RPC on every memory write. One SETNX roundtrip per write
   (cheap); once set, subsequent writes for the same user are a no-op.
2. **`replace_existing=True` on the scheduler add_job call** — durable
   backstop. If the Redis key expires (or Redis is briefly unavailable)
   we re-register; the scheduler treats it as a no-op update.

Failures are logged at WARN and swallowed — the caller's ingestion
path must not break because community-rebuild registration failed.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

REGISTRATION_KEY_TEMPLATE = "community_rebuild_registered:{user_id}"
# Matches the cron cadence (weekly) so we re-check at least once per
# cron-tick interval. Setting longer creates a window where Redis says
# "registered" but the actual schedule has been deleted out-of-band;
# setting shorter wastes scheduler RPC calls.
REGISTRATION_TTL_SECONDS = 7 * 24 * 3600  # 7 days


async def _resolve_user_timezone(user_id: str) -> str:
    """Look up the user's IANA timezone from Postgres, falling back to UTC.

    Matches the resolution other scheduled jobs use (e.g. the graph-
    execution schedule). When the user has no timezone configured
    (sentinel ``"not-set"`` or missing row), defaults to UTC so the
    cron still has a deterministic firing point.

    Uses the module-level ``prisma`` instance already managed by the
    app's lifecycle (same pattern as ``backend/data/user.py``).
    """
    try:
        from prisma import Prisma  # noqa: F401 — ensures registry
        from prisma.models import User

        from backend.data.model import USER_TIMEZONE_NOT_SET

        user = await User.prisma().find_unique(where={"id": user_id})
        if user is None:
            return "UTC"
        tz = (user.timezone or "").strip()
        if not tz or tz == USER_TIMEZONE_NOT_SET:
            return "UTC"
        return tz
    except Exception:
        logger.debug(
            "Could not resolve timezone for user %s; defaulting to UTC",
            user_id[:12],
            exc_info=True,
        )
        return "UTC"


async def _is_communities_enabled(user_id: str) -> bool:
    """Defense-in-depth: skip registration if the LD flag is off for the user.

    Mirrors the gate the scheduler @expose itself will apply when the
    audit-branch flag-gate fixup merges; until then, this helper is the
    only guard. Either way, double-checking is cheap.
    """
    try:
        from .config import is_communities_enabled_for_user

        return await is_communities_enabled_for_user(user_id)
    except Exception:
        logger.debug(
            "Could not evaluate is_communities_enabled_for_user for %s; "
            "defaulting to OFF",
            user_id[:12],
            exc_info=True,
        )
        return False


async def ensure_community_rebuild_scheduled(user_id: str) -> dict[str, Any] | None:
    """Idempotently register a weekly community rebuild for ``user_id``.

    Designed to be called fire-and-forget from the ingestion path on
    every memory write. Returns:

    - ``None`` when the fast-path SETNX says "already registered"
    - A scheduler-result dict when we did register this call
    - A small ``{"skipped": True, "reason": ...}`` dict when flag is off
      or registration failed

    Never raises. Catches every exception path so the caller's ingestion
    flow is never affected by community-rebuild plumbing problems.
    """
    if not user_id:
        return None

    # Flag gate first — cheap LD call, avoids the rest.
    if not await _is_communities_enabled(user_id):
        return {"skipped": True, "reason": "graphiti_communities_disabled"}

    # Fast path: Redis SETNX. If the key already exists, we've already
    # registered (or attempted to) within the TTL window — bail out.
    try:
        from backend.data.redis_client import get_redis_async

        redis = await get_redis_async()
        key = REGISTRATION_KEY_TEMPLATE.format(user_id=user_id)
        ok = await redis.set(key, "1", nx=True, ex=REGISTRATION_TTL_SECONDS)
        if not ok:
            return None
    except Exception:
        # If Redis is unavailable we still register via the scheduler
        # (which has its own replace_existing=True idempotency); we
        # just lose the SETNX dedup so we make an extra RPC call.
        logger.debug(
            "Redis SETNX failed for %s; proceeding to scheduler call",
            user_id[:12],
            exc_info=True,
        )

    # Slow path: actually register the cron.
    try:
        from backend.util.clients import get_scheduler_client

        tz = await _resolve_user_timezone(user_id)
        client = get_scheduler_client()
        result = await client.add_community_rebuild_schedule(
            user_id=user_id, user_timezone=tz
        )
        logger.info(
            "Registered community rebuild for user %s (tz=%s)",
            user_id[:12],
            tz,
        )
        return result
    except Exception:
        logger.warning(
            "Failed to register community rebuild for user %s",
            user_id[:12],
            exc_info=True,
        )
        return {"skipped": True, "reason": "registration_failed"}
