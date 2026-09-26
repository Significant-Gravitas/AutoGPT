"""One writer at a time per memory graph, so a forget never overlaps an
ingestion.

graphiti's ``add_episode`` saves the edges and entities it read when it
began (``SET r = edge``, ``SET n = node``): a forget that landed while it
ran would be written over by the older copy. So both writers hold this lock,
the ingestion worker around ``add_episode`` and what it writes after it
(``ingest.py``), and ``recall_forget.retract`` around the whole forget. It is
one Redis key per graph (``MemoryScope.redis_key("write_lock")``), set NX
with a token and a five-minute expiry that the holder keeps renewing, and
released and renewed by the dream lock's compare-and-delete and
compare-and-extend scripts (``dream/locks.py``), so a holder whose key
expired never touches a newer holder's.

A writer waits a bounded time: an ingestion ``INGEST_LOCK_WAIT_SECONDS``,
then its episode goes to the back of the queue once; a forget
``FORGET_LOCK_WAIT_SECONDS``, then it fails as ``busy`` with nothing
written. When Redis cannot be reached both go ahead without the lock and log
a warning: the one window left in which a forget can be written over.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, cast

from backend.copilot.dream.locks import EXTEND_SCRIPT, UNLOCK_SCRIPT
from backend.data.redis_client import get_redis_async

from .scope import write_lock_key

logger = logging.getLogger(__name__)

WRITE_LOCK_TTL_SECONDS = 300
INGEST_LOCK_WAIT_SECONDS = 60
FORGET_LOCK_WAIT_SECONDS = 20
_POLL_SECONDS = 0.25
_RENEW_EVERY_SECONDS = WRITE_LOCK_TTL_SECONDS / 3
# ``get_redis_async`` retries a lost connection for minutes; every call here
# is bounded, so Redis being down costs a writer seconds, not minutes.
_REDIS_TIMEOUT_SECONDS = 2.0


class LockState(str, Enum):
    """What a writer got: the lock, a lock held by another writer for the
    whole wait, or no Redis to take it from."""

    HELD = "held"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"


@asynccontextmanager
async def graph_write_lock(
    group_id: str, *, wait_seconds: float
) -> AsyncIterator[LockState]:
    """Hold graph ``group_id``'s write lock for the block, waiting up to
    ``wait_seconds`` for it. Yields ``BUSY`` (write nothing) or
    ``UNAVAILABLE`` (write without it) instead of raising."""
    key = write_lock_key(group_id)
    token = uuid.uuid4().hex
    state = await _acquire(key, token, wait_seconds)
    if state is not LockState.HELD:
        yield state
        return
    renewal = asyncio.create_task(_renew(key, token))
    try:
        yield state
    finally:
        renewal.cancel()
        await _release(key, token)


async def _acquire(key: str, token: str, wait_seconds: float) -> LockState:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + wait_seconds
    while True:
        try:
            async with asyncio.timeout(_REDIS_TIMEOUT_SECONDS):
                redis = await get_redis_async()
                acquired = await redis.set(
                    key, token, nx=True, px=WRITE_LOCK_TTL_SECONDS * 1000
                )
        except Exception:
            logger.warning(
                f"Redis unreachable for {key[:48]}: writing without the lock",
                exc_info=True,
            )
            return LockState.UNAVAILABLE
        if acquired:
            return LockState.HELD
        if loop.time() >= deadline:
            return LockState.BUSY
        await asyncio.sleep(_POLL_SECONDS)


async def _renew(key: str, token: str) -> None:
    """Keep a held lock from expiring under a long write; stop once the key
    is no longer this holder's."""
    while True:
        await asyncio.sleep(_RENEW_EVERY_SECONDS)
        try:
            renewed = await _script(EXTEND_SCRIPT, key, token, WRITE_LOCK_TTL_SECONDS)
        except Exception:
            logger.warning(f"Renewing {key[:48]} failed", exc_info=True)
            continue
        if not renewed:
            logger.warning(f"{key[:48]} expired while held: writers may overlap")
            return


async def _release(key: str, token: str) -> None:
    try:
        await _script(UNLOCK_SCRIPT, key, token)
    except Exception:
        logger.warning(f"Releasing {key[:48]} failed; it expires", exc_info=True)


async def _script(script: str, key: str, token: str, *args: int) -> int:
    async with asyncio.timeout(_REDIS_TIMEOUT_SECONDS):
        redis = await get_redis_async()
        # redis-py types ``eval`` as returning ``str``; the dream lock casts too.
        return int(await cast(Any, redis.eval(script, 1, key, token, *args)))
