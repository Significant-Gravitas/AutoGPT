"""One create at a time for a client-named Local PC session."""

from asyncio import sleep
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import monotonic
from uuid import uuid4

from fastapi import HTTPException

from backend.data.redis_client import get_redis_async
from backend.executor.cluster_lock import AsyncClusterLock

_LOCK_PREFIX = "copilot:local-session-create:"
# Outlives the slowest create: attach and activate (10 s each), the data
# channel wait (8 s) and the compensating detach (5 s). A holder that dies
# leaves the key behind for at most this long.
_LOCK_TTL_SECONDS = 120
_WAIT_SECONDS = 45.0
_POLL_SECONDS = 0.25


@asynccontextmanager
async def client_named_local_create(
    user_id: str, session_id: str | None
) -> AsyncIterator[None]:
    """Hold a lock on ``session_id`` while a Local PC create adopts or makes it.

    The browser resends a create whose response it lost, usually while the
    first attempt is still waiting for the data channel. Adopting the row then
    would hand back a session that the first attempt deletes if the wait
    fails, so the resend waits for the first attempt's outcome. An id the
    server generates cannot collide and takes no lock.
    """
    if session_id is None:
        yield
        return
    lock = AsyncClusterLock(
        redis=await get_redis_async(),
        key=f"{_LOCK_PREFIX}{user_id}:{session_id}",
        owner_id=uuid4().hex,
        timeout=_LOCK_TTL_SECONDS,
    )
    deadline = monotonic() + _WAIT_SECONDS
    while await lock.try_acquire() != lock.owner_id:
        if monotonic() >= deadline:
            raise HTTPException(
                status_code=503,
                detail=(
                    "This chat is still being set up on your Local PC. "
                    "Try again in a moment."
                ),
            )
        await sleep(_POLL_SECONDS)
    try:
        yield
    finally:
        await lock.release()
