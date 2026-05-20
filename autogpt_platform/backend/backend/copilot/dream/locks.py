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


class DreamLockHeld(Exception):
    """Raised when another process is already running a dream for this user."""

    def __init__(self, user_id: str) -> None:
        super().__init__(f"Dream pass already in flight for user {user_id[:12]}")
        self.user_id = user_id


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
    try:
        yield
    finally:
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
