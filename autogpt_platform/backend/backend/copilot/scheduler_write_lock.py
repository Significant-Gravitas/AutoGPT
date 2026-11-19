"""Renew a Redis lock while a caller updates shared scheduler state."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from redis.asyncio.lock import Lock as AsyncRedisLock
from redis.exceptions import LockNotOwnedError

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# A read and a write to the scheduler take well under a second, but its client
# retries a failing call for much longer, so the holder renews the lock while
# it works. The expiry frees a chat whose holder died before letting go.
_LOCK_TTL_SECONDS = 10
_LOCK_RENEW_EVERY_SECONDS = 3
# How long a writer waits for the one ahead of it.
_LOCK_WAIT_SECONDS = 5
# How long a lock call also waits on a Redis that doesn't answer. Its client
# retries for far longer, and each caller has its own way on without the lock.
_REDIS_DEADLINE_SECONDS = 2

# Renewals under way, held so one isn't collected before it ends.
_renewals: set[asyncio.Task[None]] = set()


@asynccontextmanager
async def scheduler_write_lock(key: str) -> AsyncIterator[bool]:
    """Hold a scheduler write lock for the caller's fully qualified Redis key.

    Yields False when Redis cannot be reached or the lock stays held past
    the wait. The caller decides whether it can safely proceed.
    """
    lock = await _acquire(key)
    if lock is None:
        yield False
        return
    renewal = asyncio.create_task(_keep_renewed(lock))
    _renewals.add(renewal)
    renewal.add_done_callback(_renewals.discard)
    try:
        yield True
    finally:
        renewal.cancel()
        await _release(lock)


async def _acquire(key: str) -> AsyncRedisLock | None:
    try:
        async with asyncio.timeout(_LOCK_WAIT_SECONDS + _REDIS_DEADLINE_SECONDS):
            redis = await get_redis_async()
            lock = redis.lock(
                key,
                timeout=_LOCK_TTL_SECONDS,
                blocking_timeout=_LOCK_WAIT_SECONDS,
            )
            if await lock.acquire():
                return lock
        logger.warning("Scheduler write lock stayed held for %ss", _LOCK_WAIT_SECONDS)
    except Exception:
        logger.warning("Could not acquire scheduler write lock", exc_info=True)
    return None


async def _keep_renewed(lock: AsyncRedisLock) -> None:
    """Reset the lock's expiry while its holder works, so a slow scheduler
    call doesn't let it lapse under the holder."""
    while True:
        await asyncio.sleep(_LOCK_RENEW_EVERY_SECONDS)
        try:
            async with asyncio.timeout(_REDIS_DEADLINE_SECONDS):
                await lock.reacquire()
        except LockNotOwnedError:
            logger.warning("Lost scheduler write lock while holding it")
            return
        except Exception:
            # A blip: the lock outlives a few missed renewals.
            logger.warning("Could not renew scheduler write lock", exc_info=True)


async def _release(lock: AsyncRedisLock) -> None:
    try:
        async with asyncio.timeout(_REDIS_DEADLINE_SECONDS):
            await lock.release()
    except Exception:
        # Held past its expiry, so the next writer may have it now, or Redis
        # went away; the expiry frees it either way.
        logger.warning("Could not release scheduler write lock", exc_info=True)
