"""One writer at a time on a chat's turn at the budget reset, on any replica.

A chat has at most one resume and one send at the reset, each a scheduler job
under the chat's own id (``at_reset_schedule_id``). Three things read that job
and then set it: the API joining a new send onto the one waiting, the
scheduler putting a job back after it ran, and the API cancelling one. API
servers and schedulers each run as several replicas, so each of those holds
this Redis lock, keyed on the job's id, from its read to its write.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from redis.asyncio.lock import Lock as AsyncRedisLock
from redis.exceptions import LockNotOwnedError

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_KEY_PREFIX = "copilot:at_reset_lock:"
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
async def at_reset_job_lock(job_id: str) -> AsyncIterator[bool]:
    """Hold the at-reset job *job_id* from reading it to setting it.

    Yields whether it is held: not when Redis can't be reached, or when
    another writer kept it past the wait. Each caller decides whether to go
    on without it.
    """
    lock = await _acquire(job_id)
    if lock is None:
        yield False
        return
    renewal = asyncio.create_task(_keep_renewed(lock, job_id))
    _renewals.add(renewal)
    renewal.add_done_callback(_renewals.discard)
    try:
        yield True
    finally:
        renewal.cancel()
        await _release(lock, job_id)


async def _acquire(job_id: str) -> AsyncRedisLock | None:
    try:
        async with asyncio.timeout(_LOCK_WAIT_SECONDS + _REDIS_DEADLINE_SECONDS):
            redis = await get_redis_async()
            lock = redis.lock(
                f"{_KEY_PREFIX}{job_id}",
                timeout=_LOCK_TTL_SECONDS,
                blocking_timeout=_LOCK_WAIT_SECONDS,
            )
            if await lock.acquire():
                return lock
        logger.warning(f"At-reset job {job_id} stayed locked for {_LOCK_WAIT_SECONDS}s")
    except Exception:
        logger.warning(f"Could not lock at-reset job {job_id}", exc_info=True)
    return None


async def _keep_renewed(lock: AsyncRedisLock, job_id: str) -> None:
    """Reset the lock's expiry while its holder works, so a slow scheduler
    call doesn't let it lapse under the holder."""
    while True:
        await asyncio.sleep(_LOCK_RENEW_EVERY_SECONDS)
        try:
            async with asyncio.timeout(_REDIS_DEADLINE_SECONDS):
                await lock.reacquire()
        except LockNotOwnedError:
            logger.warning(f"Lost the lock on at-reset job {job_id} while holding it")
            return
        except Exception:
            # A blip: the lock outlives a few missed renewals.
            logger.warning(
                f"Could not renew the lock on at-reset job {job_id}", exc_info=True
            )


async def _release(lock: AsyncRedisLock, job_id: str) -> None:
    try:
        async with asyncio.timeout(_REDIS_DEADLINE_SECONDS):
            await lock.release()
    except Exception:
        # Held past its expiry, so the next writer may have it now, or Redis
        # went away; the expiry frees it either way.
        logger.warning(
            f"Could not release the lock on at-reset job {job_id}", exc_info=True
        )
