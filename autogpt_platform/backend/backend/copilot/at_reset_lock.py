"""One writer at a time on a chat's turn at the budget reset, on any replica.

A chat has at most one resume and one send at the reset, each a scheduler job
under the chat's own id (``at_reset_schedule_id``). Three things read that job
and then set it: the API joining a new send onto the one waiting, the
scheduler putting a job back after it ran, and the API cancelling one. API
servers and schedulers each run as several replicas, so each of those holds
this Redis lock, keyed on the job's id, from its read to its write.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from redis.asyncio.lock import Lock as AsyncRedisLock

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_KEY_PREFIX = "copilot:at_reset_lock:"
# A read and a write to the scheduler take well under a second; the expiry
# frees a chat whose holder died before letting go.
_LOCK_TTL_SECONDS = 10
# How long a writer waits for the one ahead of it.
_LOCK_WAIT_SECONDS = 5


@asynccontextmanager
async def at_reset_job_lock(job_id: str) -> AsyncIterator[bool]:
    """Hold the at-reset job *job_id* from reading it to setting it.

    Yields whether it is held: not when Redis can't be reached, or when
    another writer kept it past the wait. Each caller decides whether to go
    on without it.
    """
    lock = await _acquire(job_id)
    try:
        yield lock is not None
    finally:
        if lock is not None:
            await _release(lock, job_id)


async def _acquire(job_id: str) -> AsyncRedisLock | None:
    try:
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


async def _release(lock: AsyncRedisLock, job_id: str) -> None:
    try:
        await lock.release()
    except Exception:
        # Held past its expiry, so the next writer may have it now, or Redis
        # went away; the expiry frees it either way.
        logger.warning(
            f"Could not release the lock on at-reset job {job_id}", exc_info=True
        )
