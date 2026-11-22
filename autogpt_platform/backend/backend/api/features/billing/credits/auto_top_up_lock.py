import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from redis.asyncio.lock import Lock as AsyncRedisLock
from redis.exceptions import LockNotOwnedError

from backend.data.redis_client import AsyncRedisClient, get_redis_async

logger = logging.getLogger(__name__)

_KEY_PREFIX = "billing:auto_top_up_setup:"
_CHARGED_KEY_PREFIX = "billing:auto_top_up_setup_charged:"
# The charge makes several Stripe calls (and tries each card after a decline),
# so the holder renews the lock while it works; the expiry frees a user whose
# holder died before letting go.
_LOCK_TTL_SECONDS = 60
_LOCK_RENEW_EVERY_SECONDS = 15
# How long a save waits for the one ahead of it. Well under the frontend
# proxy's 30s response timeout, so the 503 reaches the user.
_LOCK_WAIT_SECONDS = 12
_REDIS_DEADLINE_SECONDS = 5
# Long enough for every save queued behind a charge to take the lock and see
# it: each waits at most _LOCK_WAIT_SECONDS.
_CHARGED_TTL_SECONDS = 2 * (_LOCK_WAIT_SECONDS + _REDIS_DEADLINE_SECONDS)

# Renewals under way, held so one isn't collected before it ends.
_renewals: set[asyncio.Task[None]] = set()


class AutoTopUpSetupLock:
    """What one save learned while taking the lock, and how it tells the
    saves queued behind it that it charged."""

    def __init__(
        self,
        redis: AsyncRedisClient | None,
        user_id: str,
        token: str,
        queued_behind_charge: bool,
    ) -> None:
        self._redis = redis
        self._user_id = user_id
        self._token = token
        self.held = redis is not None
        self.queued_behind_charge = queued_behind_charge

    async def record_charge(self) -> None:
        """Mark this save's turn as part of a charge, so a save that queued
        behind it doesn't charge again. Called after a charge, and by a save
        that skipped its charge for one, so a whole queue of duplicates skips."""
        if self._redis is None:
            return
        key = f"{_CHARGED_KEY_PREFIX}{self._user_id}"
        try:
            async with asyncio.timeout(_REDIS_DEADLINE_SECONDS):
                await self._redis.sadd(key, self._token)
                await self._redis.expire(key, _CHARGED_TTL_SECONDS)
        except Exception:
            logger.warning(
                f"Could not record the auto top-up charge for user {self._user_id}",
                exc_info=True,
            )


@asynccontextmanager
async def auto_top_up_setup_lock(user_id: str) -> AsyncIterator[AutoTopUpSetupLock]:
    """Let one auto top-up save per user run at a time, from reading the
    balance to saving the settings.

    The save charges the card when the balance is under the new threshold, and
    the charge only raises the balance once it succeeds. Two saves at once
    would both see the old balance and both charge. One after the other, the
    second sees the topped-up balance, but from a negative balance one refill
    can leave it still under the threshold. So a save that had to queue behind
    a save that charged learns that through ``queued_behind_charge`` instead
    of from the balance; a save that found the lock free is a separate save.

    ``held`` is False when Redis can't be reached, or when another save kept
    the lock past the wait. The caller must not charge without it.
    """
    token = uuid.uuid4().hex
    acquired = await _acquire(user_id, token)
    if acquired is None:
        yield AutoTopUpSetupLock(None, user_id, token, queued_behind_charge=False)
        return
    redis, lock, queued_behind_charge = acquired
    renewal = asyncio.create_task(_keep_renewed(lock, user_id))
    _renewals.add(renewal)
    renewal.add_done_callback(_renewals.discard)
    try:
        yield AutoTopUpSetupLock(redis, user_id, token, queued_behind_charge)
    finally:
        renewal.cancel()
        await _release(lock, user_id)


async def _acquire(
    user_id: str, token: str
) -> tuple[AsyncRedisClient, AsyncRedisLock, bool] | None:
    """Take the lock, and say whether this save queued behind one that charged.

    A save that gets the lock at once queued behind nobody. One that has to
    wait reads the holder's token first and, once it has the lock, looks for
    that token among the saves that charged."""
    try:
        async with asyncio.timeout(_LOCK_WAIT_SECONDS + _REDIS_DEADLINE_SECONDS):
            redis = await get_redis_async()
            key = f"{_KEY_PREFIX}{user_id}"
            lock = redis.lock(
                key, timeout=_LOCK_TTL_SECONDS, blocking_timeout=_LOCK_WAIT_SECONDS
            )
            if await lock.acquire(blocking=False, token=token):
                return redis, lock, False
            ahead = await _holder_token(redis, key)
            if await lock.acquire(token=token):
                charged = f"{_CHARGED_KEY_PREFIX}{user_id}"
                queued_behind_charge = (
                    bool(await redis.sismember(charged, ahead))
                    if ahead is not None
                    # The holder let go between the two tries; any recent
                    # charge is the one this save queued behind.
                    else bool(await redis.scard(charged))
                )
                return redis, lock, queued_behind_charge
        logger.warning(
            f"Auto top-up save for user {user_id} waited {_LOCK_WAIT_SECONDS}s "
            "for another save to finish"
        )
    except Exception:
        logger.warning(
            f"Could not lock auto top-up save for user {user_id}", exc_info=True
        )
    return None


async def _holder_token(redis: AsyncRedisClient, key: str) -> str | None:
    """The token of the save holding the lock right now, if any."""
    token = await redis.get(key)
    if token is None:
        return None
    return token.decode() if isinstance(token, bytes) else str(token)


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
