"""Redis hit tracker for tentative MemoryFact edges.

Owns the ``mem:hits:{user_id}:{edge_uuid}`` key shape and the
INCR/read paths the ratification pass consults. Kept in its own
module so ``ratification.py`` stays focused on the pass logic and
fits the file-length budget.

Wiring note: warm-context retrieval (``graphiti/context.py``) fires
``ratification.try_ratify_on_hit`` for every retrieved edge, which
both bumps the counter here (via ``record_memory_hit``) and promotes
tentative edges inline. The nightly ratification sweep therefore
rarely promotes — it primarily owns grace-period supersession of
tentatives that never earned a hit.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Grace period a tentative edge has to earn a warm-context hit before
# being superseded. Spec §5 mandates 30 days so even low-cadence users
# get several sessions worth of opportunities to retrieve the memory.
RATIFICATION_GRACE_PERIOD = timedelta(days=30)

# Redis key prefix for warm-context hit tracking. One key per
# (user, edge) so we can INCR without contention and let TTL clean up
# automatically. Format: ``mem:hits:{user_id}:{edge_uuid}``.
HIT_TRACKER_KEY_PREFIX = "mem:hits"


async def record_memory_hit(user_id: str, edge_uuid: str) -> None:
    """INCR the Redis hit counter for one tentative edge.

    Called from any code path that surfaces a tentative edge to the
    user (warm-context retrieval, recall tool results, etc.). TTL is
    refreshed to the full grace period on every hit so a tentative
    edge that keeps getting used stays in the counter until ratified.

    Failures are swallowed: a missing hit doesn't break chat, and the
    next call site can record it instead. If Redis is down for a long
    stretch, the worst case is unhit tentatives get superseded — the
    spec already treats that as the safe direction.
    """
    try:
        from backend.data.redis_client import get_redis_async

        redis = await get_redis_async()
        key = hit_key(user_id, edge_uuid)
        ttl_seconds = int(RATIFICATION_GRACE_PERIOD.total_seconds())
        # SET with NX + EX seeds the key with TTL, then INCR (which
        # never touches TTL), then EXPIRE so every hit refreshes the
        # window to the full grace period. All three are single-key
        # commands, so this stays cluster-safe.
        await redis.set(key, 0, nx=True, ex=ttl_seconds)
        await redis.incr(key)
        await redis.expire(key, ttl_seconds)
    except Exception:
        logger.debug(
            "record_memory_hit failed for user %s edge %s",
            user_id[:12],
            edge_uuid,
            exc_info=True,
        )


async def get_hit_count(user_id: str, edge_uuid: str) -> int:
    """Read the Redis hit counter for one edge. Missing key → 0."""
    try:
        from backend.data.redis_client import get_redis_async

        redis = await get_redis_async()
        raw = await redis.get(hit_key(user_id, edge_uuid))
        if raw is None:
            return 0
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        return int(raw)
    except Exception:
        # Redis hiccup: treat as zero hits. The edge stays tentative
        # until the grace period elapses; we never falsely promote.
        logger.debug(
            "hit-count read failed for user %s edge %s",
            user_id[:12],
            edge_uuid,
            exc_info=True,
        )
        return 0


def hit_key(user_id: str, edge_uuid: str) -> str:
    return f"{HIT_TRACKER_KEY_PREFIX}:{user_id}:{edge_uuid}"


def parse_created_at(value: Any) -> datetime | None:
    """Best-effort coerce FalkorDB's ``created_at`` into an aware datetime.

    FalkorDB returns either a native datetime (Python driver) or a
    string for some property paths. The ratification clock is in UTC
    everywhere; naive values are assumed UTC rather than rejected.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None
