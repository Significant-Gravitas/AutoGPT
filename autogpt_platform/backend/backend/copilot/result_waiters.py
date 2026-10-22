"""Whether something is waiting on a session's turn result right now.

A delegator waiting on a child (``wait_for_session_result``) is handed the
result by the child's stream; the child's turn end reads this to know it need
not leave a note as well.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_KEY_PREFIX = "copilot:result-waiters:"
# Outlives the longest wait, so a waiter that dies without its decrement cannot
# hold the count up for long.
_SLACK_SECONDS = 60


@asynccontextmanager
async def awaiting_result(session_id: str, timeout: float) -> AsyncIterator[None]:
    """Count a waiter on *session_id*'s result for the duration of the block."""
    key = _key(session_id)
    ttl = int(timeout) + _SLACK_SECONDS
    counted = False
    try:
        redis = await get_redis_async()
        # One transaction, so the count never lacks an expiry; NX then GT sets
        # it on a new key and only ever extends a longer waiter's.
        pipe = redis.pipeline(transaction=True)
        pipe.incr(key)
        pipe.expire(key, ttl, nx=True)
        pipe.expire(key, ttl, gt=True)
        await pipe.execute()
        counted = True
    except Exception:
        logger.warning(f"Could not mark session {session_id} as awaited", exc_info=True)
    try:
        yield
    finally:
        if counted:
            try:
                redis = await get_redis_async()
                await redis.decr(key)
            except Exception:
                logger.warning(
                    f"Could not unmark session {session_id} as awaited", exc_info=True
                )


async def is_result_awaited(session_id: str) -> bool:
    try:
        redis = await get_redis_async()
        count = await redis.get(_key(session_id))
    except Exception:
        logger.warning(f"Could not read waiters on session {session_id}", exc_info=True)
        return False
    return int(count or 0) > 0


def _key(session_id: str) -> str:
    return f"{_KEY_PREFIX}{session_id}"
