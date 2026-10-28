"""Each account's authenticated API requests per UTC day, counted in Redis for
the daily report of what accounts without an active plan did.

One sorted set per API surface per day, scored by request count, so a request
costs one round trip and no database write.
"""

import asyncio
import logging
import math
from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel

from backend.data.redis_client import get_redis_async
from backend.util.settings import Config

logger = logging.getLogger(__name__)
config = Config()

APISurface = Literal["internal", "external"]
_SURFACES: tuple[APISurface, ...] = ("internal", "external")

# Outlives the day it counts and the report that reads it the next morning.
_KEY_TTL_SECONDS = 3 * 24 * 60 * 60
# Every authenticated request waits on this, so a slow Redis fails it open fast.
_REDIS_TIMEOUT_SECONDS = 0.25


class APIRequestCounts(BaseModel):
    internal: int = 0
    external: int = 0

    @property
    def total(self) -> int:
        return self.internal + self.external


async def record_api_request(user_id: str, surface: APISurface) -> None:
    """Count one authenticated request. Fails open: a Redis outage loses the
    count, never the request."""
    if not config.no_plan_activity_report_enabled:
        return
    key = _day_key(surface, datetime.now(UTC).date())
    try:
        await asyncio.wait_for(_increment(key, user_id), _REDIS_TIMEOUT_SECONDS)
    except Exception as e:
        logger.warning(f"Counting an API request ({surface}) failed open: {e}")


async def find_busy_accounts(
    day: date, min_requests: int
) -> dict[str, APIRequestCounts]:
    """The accounts whose requests on `day`, both surfaces together, reached
    `min_requests`."""
    redis = await get_redis_async()
    # A total at the threshold is at least half of it on one surface.
    floor = math.ceil(min_requests / 2)
    candidates: set[str] = set()
    for surface in _SURFACES:
        members = await redis.zrangebyscore(_day_key(surface, day), floor, "+inf")
        candidates.update(str(member) for member in members)
    counts = await get_request_counts(day, sorted(candidates))
    return {
        user_id: count
        for user_id, count in counts.items()
        if count.total >= min_requests
    }


async def get_request_counts(
    day: date, user_ids: list[str]
) -> dict[str, APIRequestCounts]:
    if not user_ids:
        return {}
    redis = await get_redis_async()
    internal = await redis.zmscore(_day_key("internal", day), user_ids)
    external = await redis.zmscore(_day_key("external", day), user_ids)
    return {
        user_id: APIRequestCounts(internal=int(i or 0), external=int(e or 0))
        for user_id, i, e in zip(user_ids, internal, external)
    }


async def _increment(key: str, user_id: str) -> None:
    redis = await get_redis_async()
    pipe = redis.pipeline(transaction=True)
    pipe.zincrby(key, 1, user_id)
    pipe.expire(key, _KEY_TTL_SECONDS)
    await pipe.execute()


def _day_key(surface: APISurface, day: date) -> str:
    return f"api_requests:{surface}:{day.isoformat()}"
