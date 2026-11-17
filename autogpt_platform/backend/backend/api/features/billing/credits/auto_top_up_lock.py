import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from redis.asyncio.lock import Lock as AsyncRedisLock

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_KEY_PREFIX = "billing:auto_top_up_setup:"
# Long enough for the card charge (several cards, each a Stripe round trip);
# a holder that dies frees the lock when this runs out.
_LOCK_TTL_SECONDS = 120
_LOCK_WAIT_SECONDS = 30
_REDIS_DEADLINE_SECONDS = 5


@asynccontextmanager
async def auto_top_up_setup_lock(user_id: str) -> AsyncIterator[bool]:
    """Let one auto top-up save per user run at a time, from reading the
    balance to saving the settings.

    The save charges the card when the balance is under the new threshold, and
    the charge only raises the balance once it succeeds. Two saves at once
    would both see the old balance and both charge; one after the other, the
    second sees the topped-up balance and doesn't.

    Yields whether the lock is held: not when Redis can't be reached, or when
    another save kept it past the wait. The caller must not charge without it.
    """
    lock = await _acquire(user_id)
    if lock is None:
        yield False
        return
    try:
        yield True
    finally:
        await _release(lock, user_id)


async def _acquire(user_id: str) -> AsyncRedisLock | None:
    try:
        async with asyncio.timeout(_LOCK_WAIT_SECONDS + _REDIS_DEADLINE_SECONDS):
            redis = await get_redis_async()
            lock = redis.lock(
                f"{_KEY_PREFIX}{user_id}",
                timeout=_LOCK_TTL_SECONDS,
                blocking_timeout=_LOCK_WAIT_SECONDS,
            )
            if await lock.acquire():
                return lock
        logger.warning(
            f"Auto top-up save for user {user_id} waited {_LOCK_WAIT_SECONDS}s "
            "for another save to finish"
        )
    except Exception:
        logger.warning(
            f"Could not lock auto top-up save for user {user_id}", exc_info=True
        )
    return None


async def _release(lock: AsyncRedisLock, user_id: str) -> None:
    try:
        async with asyncio.timeout(_REDIS_DEADLINE_SECONDS):
            await lock.release()
    except Exception:
        # The TTL frees it.
        logger.warning(
            f"Could not release auto top-up save lock for user {user_id}",
            exc_info=True,
        )
