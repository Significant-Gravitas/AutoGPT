"""Redis SETNX advisory lock for the dream pass.

One in-flight dream per user at a time. Bails if the lock is already
held — the caller surfaces that as ``skipped=True`` in
``DreamPassResult`` so the admin UI can render "another dream is
already running" rather than failing.

TTL is transport-aware per ``dream/p0-spec.md`` §13:
  * Cloud (OpenRouter / Anthropic direct): 1800 s (30 min)
  * Local (Ollama / vLLM / LM Studio): 7200 s (2 hr)
The longer local TTL accommodates CPU-only inference timelines for
the three-phase pipeline.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

DREAM_LOCK_KEY_PREFIX = "dream:inflight:"

# Default TTLs. Callers can override per-transport when local.
DEFAULT_LOCK_TTL_SECONDS = 1800
LOCAL_LOCK_TTL_SECONDS = 7200

# Batch path: the dream pass is async and stays in flight up to the
# BatchExecutor's MAX_BATCH_LIFETIME_SECONDS (24h). The lock must outlive the
# whole batch so apply — which runs hours later in the callback — is still
# covered by "one dream per user". The batch callback releases it on
# terminal/failure; this TTL is only the crash backstop, kept > 24h so the
# lock can't expire before the executor times the batch out.
BATCH_LOCK_TTL_SECONDS = 24 * 60 * 60 + 600


class DreamLockHeld(Exception):
    """Raised when another process is already running a dream for this user."""

    def __init__(self, user_id: str) -> None:
        super().__init__(f"Dream pass already in flight for user {user_id[:12]}")
        self.user_id = user_id


class DreamLockHandle:
    """Yielded by ``dream_lock``. Lets the async batch path hand lock
    ownership to the callback chain: ``extend`` stretches the TTL to the
    batch lifetime and ``disown`` stops the context manager from releasing on
    exit, so the lock survives until ``release_dream_lock`` runs in the batch
    callback (or the TTL expires)."""

    def __init__(self, redis, key: str, user_id: str) -> None:
        self._redis = redis
        self._key = key
        self.user_id = user_id
        self.release_on_exit = True

    def disown(self) -> None:
        self.release_on_exit = False

    async def extend(self, ttl_seconds: int) -> None:
        await self._redis.expire(self._key, ttl_seconds)


@asynccontextmanager
async def dream_lock(user_id: str, ttl_seconds: int = DEFAULT_LOCK_TTL_SECONDS):
    """Acquire a per-user advisory lock for the dream pass.

    Raises ``DreamLockHeld`` if another process holds the lock.
    The lock is released on context exit; on crash the TTL provides
    a fallback release after ``ttl_seconds``.
    """
    # Lazy import so this module is cheap to import in tests that mock redis.
    from backend.data.redis_client import get_redis_async

    redis = await get_redis_async()
    key = f"{DREAM_LOCK_KEY_PREFIX}{user_id}"

    acquired = await redis.set(key, "1", nx=True, ex=ttl_seconds)
    if not acquired:
        raise DreamLockHeld(user_id)

    logger.info("Acquired dream lock for user %s (ttl=%ds)", user_id[:12], ttl_seconds)
    handle = DreamLockHandle(redis, key, user_id)
    try:
        yield handle
    finally:
        if not handle.release_on_exit:
            # Disowned by the batch path — the callback (or the TTL) owns
            # release now, so the lock spans the async batch lifetime.
            logger.info("Dream lock for user %s handed to batch callback", user_id[:12])
        else:
            try:
                await redis.delete(key)
                logger.debug("Released dream lock for user %s", user_id[:12])
            except Exception:
                # If release fails, the TTL will eventually clear the lock.
                logger.warning(
                    "Failed to release dream lock for user %s — TTL %ds will clear",
                    user_id[:12],
                    ttl_seconds,
                    exc_info=True,
                )


async def release_dream_lock(user_id: str) -> None:
    """Release a disowned dream lock (batch path) once the pass terminates.

    Blind delete is safe: the batch lock's TTL (>= the BatchExecutor's
    MAX_BATCH_LIFETIME_SECONDS) can't expire before the batch terminates, so
    the key is still ours. A failed delete falls back to the TTL.
    """
    from backend.data.redis_client import get_redis_async

    try:
        redis = await get_redis_async()
        await redis.delete(f"{DREAM_LOCK_KEY_PREFIX}{user_id}")
        logger.debug("Released disowned dream lock for user %s", user_id[:12])
    except Exception:
        logger.warning(
            "Failed to release disowned dream lock for user %s — TTL will clear",
            user_id[:12],
            exc_info=True,
        )
