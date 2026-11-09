"""A voice call's shared state in Redis.

Every operation touches one key, so the relay and the REST API run against the
production Redis cluster unchanged:

- ``voice:live:<user id>`` names the user's live call: one live interview per
  user. It is taken with SET NX and a TTL longer than any call, so a dead
  pod's lock clears itself, and released by compare-and-delete, so a call
  never frees a lock that a later call holds.
- ``voice:call:<call id>``, a hash, is the call's record for a day: its owner,
  pod, leg and status, and how it ended.
- ``voice:concurrent``, a sorted set of live call ids scored by when each was
  last refreshed, counts live calls across the platform. A call whose pod died
  stops counting once its entry goes stale.
- ``voice:end:<call id>`` asks the pod that owns a call to end it; the pod
  looks every second.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Mapping
from typing import TypeVar

from redis.typing import EncodableT, FieldT

from backend.data.redis_client import AsyncRedisClient
from backend.data.redis_helpers import as_str, string_compare_and_set
from backend.voice_relay.config import (
    CALL_RECORD_TTL_S,
    END_REQUEST_TTL_S,
    LIVE_CALL_STALE_S,
    USER_LOCK_TTL_S,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

LIVE_CALLS_KEY = "voice:concurrent"
# How long one Redis operation of a live call may take before it is skipped.
REDIS_TIMEOUT_S = 0.5


def user_lock_key(user_id: str) -> str:
    return f"voice:live:{user_id}"


def call_record_key(call_id: str) -> str:
    return f"voice:call:{call_id}"


def end_flag_key(call_id: str) -> str:
    return f"voice:end:{call_id}"


async def take_user_lock(
    redis: AsyncRedisClient, user_id: str, call_id: str
) -> str | None:
    """Make *call_id* the user's live call.

    Returns None once it is, or the id of the live call that already holds
    the lock (blank in the rare case it kept changing hands while read).
    """
    key = user_lock_key(user_id)
    for _ in range(2):
        if await redis.set(key, call_id, nx=True, ex=USER_LOCK_TTL_S):
            return None
        holder = as_str(await redis.get(key))
        if holder is not None:
            return holder
        # The holder's lock expired between the two reads: try once more.
    return ""


async def live_call_of(redis: AsyncRedisClient, user_id: str) -> str | None:
    """The user's live call, or None."""
    return as_str(await redis.get(user_lock_key(user_id)))


async def release_user_lock(
    redis: AsyncRedisClient, user_id: str, call_id: str
) -> bool:
    """Free the user's lock if *call_id* still holds it. True when it did."""
    return await string_compare_and_set(
        redis, user_lock_key(user_id), expected=call_id, new=None
    )


async def write_call_record(
    redis: AsyncRedisClient, call_id: str, fields: Mapping[str, object]
) -> None:
    """Set *fields* on the call's record and give it a day to live.

    A None value is left out; anything that is not a string is stored as
    JSON.
    """
    mapping: dict[FieldT, EncodableT] = {
        name: value if isinstance(value, str) else json.dumps(value)
        for name, value in fields.items()
        if value is not None
    }
    if not mapping:
        return
    key = call_record_key(call_id)
    pipe = redis.pipeline(transaction=True)
    pipe.hset(key, mapping=mapping)
    pipe.expire(key, CALL_RECORD_TTL_S)
    await pipe.execute()


async def read_call_record(redis: AsyncRedisClient, call_id: str) -> dict[str, str]:
    raw = await redis.hgetall(call_record_key(call_id))
    return {
        as_str(name) or "": as_str(value) or "" for name, value in (raw or {}).items()
    }


async def register_live_call(redis: AsyncRedisClient, call_id: str, now: float) -> None:
    """Count *call_id* as live across the platform, or refresh its entry.

    Entries a dead pod left behind are dropped on the way.
    """
    pipe = redis.pipeline(transaction=True)
    pipe.zremrangebyscore(LIVE_CALLS_KEY, "-inf", now - LIVE_CALL_STALE_S)
    pipe.zadd(LIVE_CALLS_KEY, {call_id: now})
    pipe.expire(LIVE_CALLS_KEY, int(LIVE_CALL_STALE_S * 3))
    await pipe.execute()


async def unregister_live_call(redis: AsyncRedisClient, call_id: str) -> None:
    await redis.zrem(LIVE_CALLS_KEY, call_id)


async def live_call_count(redis: AsyncRedisClient, now: float) -> int:
    """Live calls across the platform, not counting entries gone stale."""
    return int(
        await redis.zcount(LIVE_CALLS_KEY, f"({now - LIVE_CALL_STALE_S}", "+inf")
    )


async def request_end(redis: AsyncRedisClient, call_id: str) -> None:
    """Ask the pod that owns *call_id* to end it."""
    await redis.set(end_flag_key(call_id), "1", ex=END_REQUEST_TTL_S)


async def end_requested(redis: AsyncRedisClient, call_id: str) -> bool:
    return bool(await redis.exists(end_flag_key(call_id)))


class CallStore:
    """One live call's shared state, as the pod that owns the call keeps it.

    A Redis outage never ends or stalls a call: each operation is bounded,
    and a failed one is logged and skipped.
    """

    def __init__(self, redis: AsyncRedisClient, *, call_id: str, user_id: str) -> None:
        self.redis = redis
        self.call_id = call_id
        self.user_id = user_id

    async def record(self, **fields: object) -> None:
        await self._bounded(
            "record", write_call_record(self.redis, self.call_id, fields)
        )

    async def refresh_live(self, now: float) -> None:
        await self._bounded(
            "live-call refresh", register_live_call(self.redis, self.call_id, now)
        )

    async def end_requested(self) -> bool:
        return bool(
            await self._bounded("end flag", end_requested(self.redis, self.call_id))
        )

    async def release(self) -> None:
        """Free the user's lock and stop counting the call as live. Idempotent."""
        await self._bounded(
            "lock release", release_user_lock(self.redis, self.user_id, self.call_id)
        )
        await self._bounded(
            "live-call release", unregister_live_call(self.redis, self.call_id)
        )

    async def _bounded(self, what: str, operation: Awaitable[T]) -> T | None:
        try:
            return await asyncio.wait_for(operation, timeout=REDIS_TIMEOUT_S)
        except Exception:
            logger.warning(f"Voice call {self.call_id}: {what} failed", exc_info=True)
            return None
