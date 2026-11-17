"""One writer at a time per memory graph, so a forget does not overlap an
ingestion.

graphiti's ``add_episode`` saves the edges and entities it read when it
began, so a forget that landed while it ran would be written over by the
older copy. The driver saves nothing on a soft-forgotten fact but its
episode list (``fact_attributes.py``); graphiti still saves the fact's
entities whole (``SET n = node``), their summaries back, and a
hard-forgotten fact again, live. So these writers hold this lock: the
ingestion worker around ``add_episode`` and what it writes after it
(``ingest.py``), ``recall_forget.retract`` around the whole forget, and the
legacy-forget backfill around each graph's restamp
(``migrations/backfill_legacy_forgets.py``); the weekly community rebuild
and the settings page's erase-all do not (``graphiti/AGENTS.md``). It is one
Redis key per graph (``MemoryScope.redis_key("write_lock")``), set NX with a
token and a five-minute expiry that the holder keeps renewing, and released
and renewed by the dream lock's compare-and-delete and compare-and-extend
scripts (``dream/locks.py``), so a holder whose key expired never touches a
newer holder's.

Waiters take turns. A writer that finds the lock held claims the next turn
(``MemoryScope.redis_key``'s ``write_lock`` key plus ``:next``, set NX with
its token, renewed at every poll and lapsing ``_TURN_CLAIM_SECONDS`` after
its waiter stops), and while another waiter holds that claim no one else
takes the lock, the writer that just let it go included. So a worker
writing a backlog episode after episode hands the lock to a waiter between
two of them instead of taking it straight back within one Redis round trip,
which a poll every ``_POLL_SECONDS`` almost never beat.

A writer waits a bounded time: an ingestion ``INGEST_LOCK_WAIT_SECONDS``,
then its episode goes to the back of the queue after a pause and waits
again, as often as it takes, never dropped for waiting
(``ingest._write_or_requeue``); a forget ``FORGET_LOCK_WAIT_SECONDS``, then
it fails as ``busy`` with nothing written; the backfill skips the graph as
busy. Two windows remain
in which a forget can be written over, and neither has a bound. When Redis
cannot be reached, the ingestion worker and a forget go ahead without the
lock and log a warning (a lock taken by a request that timed out after Redis
applied it is released, best-effort). And a holder that loses its key (its
renewals fail, or its event loop stalls past the expiry) keeps writing
while another writer takes the lock: the renewal only warns, and nothing
fences the graph write.
"""

import asyncio
import logging
import sys
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, cast

from backend.copilot.dream.locks import EXTEND_SCRIPT, UNLOCK_SCRIPT
from backend.data.redis_client import get_redis_async

from .scope import write_lock_key

if sys.version_info >= (3, 11):
    from asyncio import timeout
else:
    from async_timeout import timeout

logger = logging.getLogger(__name__)

WRITE_LOCK_TTL_SECONDS = 300
INGEST_LOCK_WAIT_SECONDS = 60
FORGET_LOCK_WAIT_SECONDS = 20
_POLL_SECONDS = 0.25
# How long a waiter's claim on the next turn outlives its last poll.
_TURN_CLAIM_SECONDS = 2
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
    ``wait_seconds`` for it. Yields ``BUSY`` (another writer kept it for the
    whole wait) or ``UNAVAILABLE`` (Redis could not be reached) instead of
    raising; the caller decides what to do without it: the worker and a
    forget write anyway, the backfill skips the graph."""
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
            async with timeout(_REDIS_TIMEOUT_SECONDS):
                acquired = await _take_or_wait_turn(key, token)
        except Exception:
            logger.warning(
                f"Redis unreachable for {key[:48]}: write lock unavailable",
                exc_info=True,
            )
            # The SET may have landed before the request gave up: free it.
            await _release(key, token)
            return LockState.UNAVAILABLE
        if acquired:
            return LockState.HELD
        if loop.time() >= deadline:
            # Stop waiting, and let the next waiter have the turn at once.
            await _release(f"{key}:next", token)
            return LockState.BUSY
        await asyncio.sleep(_POLL_SECONDS)


async def _take_or_wait_turn(key: str, token: str) -> bool:
    """One attempt at the lock: taken unless another waiter claimed the next
    turn; else the next turn is claimed for this writer, or its claim kept."""
    redis = await get_redis_async()
    turn = f"{key}:next"
    claimed = await redis.get(turn)
    if claimed in (None, token):
        if await redis.set(key, token, nx=True, px=WRITE_LOCK_TTL_SECONDS * 1000):
            if claimed == token:
                await cast(Any, redis.eval(UNLOCK_SCRIPT, 1, turn, token))
            return True
    if claimed is None:
        await redis.set(turn, token, nx=True, px=_TURN_CLAIM_SECONDS * 1000)
    elif claimed == token:
        await cast(Any, redis.eval(EXTEND_SCRIPT, 1, turn, token, _TURN_CLAIM_SECONDS))
    return False


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
    async with timeout(_REDIS_TIMEOUT_SECONDS):
        redis = await get_redis_async()
        # redis-py types ``eval`` as returning ``str``; the dream lock casts too.
        return int(await cast(Any, redis.eval(script, 1, key, token, *args)))
