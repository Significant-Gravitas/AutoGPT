"""Per-request idempotency lock for the /stream endpoint.

Prevents duplicate executor tasks from concurrent or retried POSTs (e.g. k8s
rolling-deploy retries, nginx upstream retries, rapid double-clicks).

Lifecycle
---------
1. ``acquire()`` — computes a stable hash of (session_id, message, file_ids)
   and atomically sets a Redis NX key. Returns a ``_DedupLock`` on success or
   ``None`` when the key already exists (duplicate request).
2. ``release()`` — deletes the key. Must be called on turn completion or turn
   error so the next legitimate send is never blocked.
3. On client disconnect (``GeneratorExit``) the lock must NOT be released —
   the backend turn is still running, and releasing would reopen the duplicate
   window for infra-level retries. The 30 s TTL is the safety net.
"""

import hashlib
import logging

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

_KEY_PREFIX = "chat:msg_dedup"
_TTL_SECONDS = 30


class _DedupLock:
    def __init__(self, key: str, redis) -> None:
        self._key = key
        self._redis = redis

    async def release(self) -> None:
        """Best-effort key deletion. The TTL handles failures silently."""
        try:
            await self._redis.delete(self._key)
        except Exception:
            pass


async def acquire_dedup_lock(
    session_id: str,
    message: str | None,
    file_ids: list[str] | None,
) -> _DedupLock | None:
    """Acquire the idempotency lock for this (session, message, files) tuple.

    Returns a ``_DedupLock`` when the lock is freshly acquired (first request).
    Returns ``None`` when a duplicate is detected (lock already held).
    Returns ``None`` when there is nothing to deduplicate (no message, no files).
    """
    if not message and not file_ids:
        return None

    sorted_ids = ":".join(sorted(file_ids or []))
    content_hash = hashlib.sha256(
        f"{session_id}:{message or ''}:{sorted_ids}".encode()
    ).hexdigest()[:16]
    key = f"{_KEY_PREFIX}:{session_id}:{content_hash}"

    redis = await get_redis_async()
    acquired = await redis.set(key, "1", ex=_TTL_SECONDS, nx=True)
    if not acquired:
        logger.warning(
            f"[STREAM] Duplicate user message blocked for session {session_id}, "
            f"hash={content_hash} — returning empty SSE",
        )
        return None

    return _DedupLock(key, redis)
