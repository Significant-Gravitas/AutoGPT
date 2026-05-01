"""Redis-based distributed locking for cluster coordination."""

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from backend.data.redis_client import AsyncRedisClient, RedisClient

logger = logging.getLogger(__name__)

# CAS release: DEL only when the stored owner still matches — guards against
# wiping a successor's lock after an external force-release.
_RELEASE_LUA = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) "
    "else return 0 end"
)


class ClusterLock:
    """Simple Redis-based distributed lock for preventing duplicate execution."""

    def __init__(
        self, redis: "RedisClient", key: str, owner_id: str, timeout: int = 300
    ):
        self.redis = redis
        self.key = key
        self.owner_id = owner_id
        self.timeout = timeout
        self._last_refresh = 0.0
        self._refresh_lock = threading.Lock()

    def try_acquire(self) -> str | None:
        """Try to acquire the lock.

        Returns:
            - owner_id (self.owner_id) if successfully acquired
            - different owner_id if someone else holds the lock
            - None if Redis is unavailable or other error
        """
        try:
            success = self.redis.set(self.key, self.owner_id, nx=True, ex=self.timeout)
            if success:
                with self._refresh_lock:
                    self._last_refresh = time.time()
                return self.owner_id  # Successfully acquired

            # Failed to acquire, get current owner
            current_value = self.redis.get(self.key)
            if current_value:
                current_owner = (
                    current_value.decode("utf-8")
                    if isinstance(current_value, bytes)
                    else str(current_value)
                )
                return current_owner

            # Key doesn't exist but we failed to set it - race condition or Redis issue
            return None

        except Exception as e:
            logger.error(f"ClusterLock.try_acquire failed for key {self.key}: {e}")
            return None

    def refresh(self) -> bool:
        """Refresh lock TTL if we still own it.

        Rate limited to at most once every timeout/10 seconds (minimum 1 second).
        During rate limiting, still verifies lock existence but skips TTL extension.
        Setting _last_refresh to 0 bypasses rate limiting for testing.

        Thread-safe: uses _refresh_lock to protect _last_refresh access.
        """
        # Calculate refresh interval: max(timeout // 10, 1)
        refresh_interval = max(self.timeout // 10, 1)
        current_time = time.time()

        # Check if we're within the rate limit period (thread-safe read)
        # _last_refresh == 0 forces a refresh (bypasses rate limiting for testing)
        with self._refresh_lock:
            last_refresh = self._last_refresh
        is_rate_limited = (
            last_refresh > 0 and (current_time - last_refresh) < refresh_interval
        )

        try:
            # Always verify lock existence, even during rate limiting
            current_value = self.redis.get(self.key)
            if not current_value:
                with self._refresh_lock:
                    self._last_refresh = 0
                return False

            stored_owner = (
                current_value.decode("utf-8")
                if isinstance(current_value, bytes)
                else str(current_value)
            )
            if stored_owner != self.owner_id:
                with self._refresh_lock:
                    self._last_refresh = 0
                return False

            # If rate limited, return True but don't update TTL or timestamp
            if is_rate_limited:
                return True

            # Perform actual refresh
            if self.redis.expire(self.key, self.timeout):
                with self._refresh_lock:
                    self._last_refresh = current_time
                return True

            with self._refresh_lock:
                self._last_refresh = 0
            return False

        except Exception as e:
            logger.error(f"ClusterLock.refresh failed for key {self.key}: {e}")
            with self._refresh_lock:
                self._last_refresh = 0
            return False

    def release(self):
        """Release the lock.

        Owner-checked: only deletes the Redis key if the stored value still
        matches our owner_id. Prevents wiping a successor's lock when the
        original key was force-released externally and re-acquired.
        """
        with self._refresh_lock:
            if self._last_refresh == 0:
                return

        try:
            self.redis.eval(_RELEASE_LUA, 1, self.key, self.owner_id)
        except Exception:
            pass

        with self._refresh_lock:
            self._last_refresh = 0.0


class AsyncClusterLock:
    """Async Redis-based distributed lock for preventing duplicate execution."""

    def __init__(
        self, redis: "AsyncRedisClient", key: str, owner_id: str, timeout: int = 300
    ):
        self.redis = redis
        self.key = key
        self.owner_id = owner_id
        self.timeout = timeout
        self._last_refresh = 0.0
        self._refresh_lock = asyncio.Lock()

    async def try_acquire(self) -> str | None:
        """Try to acquire the lock.

        Returns:
            - owner_id (self.owner_id) if successfully acquired
            - different owner_id if someone else holds the lock
            - None if Redis is unavailable or other error
        """
        try:
            success = await self.redis.set(
                self.key, self.owner_id, nx=True, ex=self.timeout
            )
            if success:
                async with self._refresh_lock:
                    self._last_refresh = time.time()
                return self.owner_id  # Successfully acquired

            # Failed to acquire, get current owner
            current_value = await self.redis.get(self.key)
            if current_value:
                current_owner = (
                    current_value.decode("utf-8")
                    if isinstance(current_value, bytes)
                    else str(current_value)
                )
                return current_owner

            # Key doesn't exist but we failed to set it - race condition or Redis issue
            return None

        except Exception as e:
            logger.error(f"AsyncClusterLock.try_acquire failed for key {self.key}: {e}")
            return None

    async def refresh(self) -> bool:
        """Refresh lock TTL if we still own it.

        Rate limited to at most once every timeout/10 seconds (minimum 1 second).
        During rate limiting, still verifies lock existence but skips TTL extension.
        Setting _last_refresh to 0 bypasses rate limiting for testing.

        Async-safe: uses asyncio.Lock to protect _last_refresh access.
        """
        # Calculate refresh interval: max(timeout // 10, 1)
        refresh_interval = max(self.timeout // 10, 1)
        current_time = time.time()

        # Check if we're within the rate limit period (async-safe read)
        # _last_refresh == 0 forces a refresh (bypasses rate limiting for testing)
        async with self._refresh_lock:
            last_refresh = self._last_refresh
        is_rate_limited = (
            last_refresh > 0 and (current_time - last_refresh) < refresh_interval
        )

        try:
            # Always verify lock existence, even during rate limiting
            current_value = await self.redis.get(self.key)
            if not current_value:
                async with self._refresh_lock:
                    self._last_refresh = 0
                return False

            stored_owner = (
                current_value.decode("utf-8")
                if isinstance(current_value, bytes)
                else str(current_value)
            )
            if stored_owner != self.owner_id:
                async with self._refresh_lock:
                    self._last_refresh = 0
                return False

            # If rate limited, return True but don't update TTL or timestamp
            if is_rate_limited:
                return True

            # Perform actual refresh
            if await self.redis.expire(self.key, self.timeout):
                async with self._refresh_lock:
                    self._last_refresh = current_time
                return True

            async with self._refresh_lock:
                self._last_refresh = 0
            return False

        except Exception as e:
            logger.error(f"AsyncClusterLock.refresh failed for key {self.key}: {e}")
            async with self._refresh_lock:
                self._last_refresh = 0
            return False

    async def release(self):
        """Release the lock.

        Owner-checked: only deletes the Redis key if the stored value still
        matches our owner_id. Prevents wiping a successor's lock when the
        original key was force-released externally and re-acquired.
        """
        async with self._refresh_lock:
            if self._last_refresh == 0:
                return

        try:
            await cast(Any, self.redis.eval(_RELEASE_LUA, 1, self.key, self.owner_id))
        except Exception:
            pass

        async with self._refresh_lock:
            self._last_refresh = 0.0
