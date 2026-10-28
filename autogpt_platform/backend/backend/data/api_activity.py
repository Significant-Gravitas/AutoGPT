"""Each account's authenticated API requests per UTC day, counted in Redis for
the daily report of what accounts without an active plan did.

One sorted set per API surface per day, scored by request count. A request only
adds to this process's tally; a background task writes the tally to Redis every
few seconds. So the request path never touches Redis, and a slow or unreachable
Redis costs counts, never requests.
"""

import asyncio
import logging
import math
import threading
import time
from collections import Counter
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
_FLUSH_INTERVAL_SECONDS = 10.0
_FLUSH_TIMEOUT_SECONDS = 2.0
_FAILURE_LOG_INTERVAL_SECONDS = 60


class APIRequestCounts(BaseModel):
    internal: int = 0
    external: int = 0

    @property
    def total(self) -> int:
        return self.internal + self.external


def record_api_request(user_id: str, surface: APISurface) -> None:
    """Add one authenticated request to this process's tally, without waiting."""
    if not config.no_plan_activity_report_enabled:
        return
    _tally.add(_day_key(surface, datetime.now(UTC).date()), user_id)


async def flush() -> None:
    """Write this process's tally to Redis now, as its flusher does on its own."""
    await _tally.flush()


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


async def _write(counts: dict[tuple[str, str], int]) -> None:
    redis = await get_redis_async()
    # Not MULTI/EXEC, which costs ~0.6 ms of CPU on the cluster client; a key
    # a failed write leaves without its TTL gets one from the next flush.
    pipe = redis.pipeline(transaction=False)
    for (key, user_id), amount in counts.items():
        pipe.zincrby(key, amount, user_id)
    for key in {key for key, _ in counts}:
        pipe.expire(key, _KEY_TTL_SECONDS)
    await pipe.execute()


def _day_key(surface: APISurface, day: date) -> str:
    return f"api_requests:{surface}:{day.isoformat()}"


class _Tally:
    """This process's counts not yet written, and the task that writes them."""

    def __init__(self):
        # Test clients serve requests from threads of their own.
        self._lock = threading.Lock()
        self._counts: Counter[tuple[str, str]] = Counter()
        self._flusher: asyncio.Task[None] | None = None

    def add(self, key: str, user_id: str) -> None:
        with self._lock:
            self._counts[(key, user_id)] += 1
        loop = asyncio.get_running_loop()
        flusher = self._flusher
        if flusher is None or flusher.done() or flusher.get_loop() is not loop:
            self._flusher = loop.create_task(self._flush_periodically())

    async def flush(self) -> None:
        """A failed write drops the counts it carried."""
        with self._lock:
            counts, self._counts = self._counts, Counter()
        if not counts:
            return
        try:
            await asyncio.wait_for(_write(dict(counts)), _FLUSH_TIMEOUT_SECONDS)
        except Exception as e:
            _failures.record(e)

    async def _flush_periodically(self) -> None:
        while True:
            await asyncio.sleep(_FLUSH_INTERVAL_SECONDS)
            await self.flush()


class _FailureLog:
    """One warning a minute while writes fail, saying how many it held back."""

    def __init__(self):
        self.logged_at = -math.inf
        self.held_back = 0

    def record(self, error: Exception) -> None:
        now = time.monotonic()
        if now - self.logged_at < _FAILURE_LOG_INTERVAL_SECONDS:
            self.held_back += 1
            return
        logger.warning(
            f"Writing API request counts to Redis failed, dropping them "
            f"({self.held_back} more failures since the last warning): {error!r}"
        )
        self.logged_at, self.held_back = now, 0


_tally = _Tally()
_failures = _FailureLog()
