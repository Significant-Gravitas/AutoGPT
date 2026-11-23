import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from redis.asyncio.lock import Lock as AsyncRedisLock
from redis.exceptions import LockNotOwnedError

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_KEY_PREFIX = "billing:auto_top_up_setup:"
# The charge makes several Stripe calls (and tries each card after a decline),
# so the holder renews the lock while it works; the expiry frees a user whose
# holder died before letting go.
_LOCK_TTL_SECONDS = 60
_LOCK_RENEW_EVERY_SECONDS = 15
# How long a save waits for the one ahead of it. Well under the frontend
# proxy's 30s response timeout, so the 503 reaches the user.
_LOCK_WAIT_SECONDS = 12
_REDIS_DEADLINE_SECONDS = 5

# "busy": another save kept the lock past the wait.
# "unavailable": Redis couldn't be reached.
LockState = Literal["held", "busy", "unavailable"]

# Renewals under way, held so one isn't collected before it ends.
_renewals: set[asyncio.Task[None]] = set()


@asynccontextmanager
async def auto_top_up_setup_lock(user_id: str) -> AsyncIterator[LockState]:
    """Let one auto top-up save per user run at a time, from reading the
    balance to saving the settings.

    The save charges the card when the balance is under the new threshold, and
    the charge only raises the balance once it succeeds. Two saves at once
    would both see the old balance and both charge; one after the other, the
    second sees the topped-up balance and doesn't.

    Yields "held", or why not: "busy" when another save kept it past the
    wait, "unavailable" when Redis can't be reached. The caller must not
    charge without it.
    """
    lock, state = await _acquire(user_id)
    if lock is None:
        yield state
        return
    renewal = asyncio.create_task(_keep_renewed(lock, user_id))
    _renewals.add(renewal)
    renewal.add_done_callback(_renewals.discard)
    try:
        yield "held"
    finally:
        renewal.cancel()
        await _release(lock, user_id)


async def _acquire(user_id: str) -> tuple[AsyncRedisLock | None, LockState]:
    try:
        async with asyncio.timeout(_LOCK_WAIT_SECONDS + _REDIS_DEADLINE_SECONDS):
            redis = await get_redis_async()
            lock = redis.lock(
                f"{_KEY_PREFIX}{user_id}",
                timeout=_LOCK_TTL_SECONDS,
                blocking_timeout=_LOCK_WAIT_SECONDS,
            )
            if await lock.acquire():
                return lock, "held"
        logger.warning(
            f"Auto top-up save for user {user_id} waited {_LOCK_WAIT_SECONDS}s "
            "for another save to finish"
        )
        return None, "busy"
    except Exception:
        logger.warning(
            f"Could not lock auto top-up save for user {user_id}", exc_info=True
        )
        return None, "unavailable"


async def _keep_renewed(lock: AsyncRedisLock, user_id: str) -> None:
    """Reset the lock's expiry while its holder charges, so a slow Stripe call
    doesn't let the next save in before this charge is credited."""
    while True:
        await asyncio.sleep(_LOCK_RENEW_EVERY_SECONDS)
        try:
            async with asyncio.timeout(_REDIS_DEADLINE_SECONDS):
                await lock.reacquire()
        except LockNotOwnedError:
            logger.warning(
                f"Lost the auto top-up save lock for user {user_id} while holding it"
            )
            return
        except Exception:
            # A blip: the lock outlives a few missed renewals.
            logger.warning(
                f"Could not renew the auto top-up save lock for user {user_id}",
                exc_info=True,
            )


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
