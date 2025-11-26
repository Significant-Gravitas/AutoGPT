"""Redis-based distributed locking for cluster coordination."""

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis import Redis

logger = logging.getLogger(__name__)


class ClusterLock:
    """Simple Redis-based distributed lock for preventing duplicate execution."""

    def __init__(self, redis: "Redis", key: str, owner_id: str, timeout: int = 300):
        self.redis = redis
        self.key = key
        self.owner_id = owner_id
        self.timeout = timeout
        self._last_refresh = 0.0

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
        """
        # Calculate refresh interval: max(timeout // 10, 1)
        refresh_interval = max(self.timeout // 10, 1)
        current_time = time.time()

        # Check if we're within the rate limit period
        # _last_refresh == 0 forces a refresh (bypasses rate limiting for testing)
        is_rate_limited = (
            self._last_refresh > 0
            and (current_time - self._last_refresh) < refresh_interval
        )

        try:
            # Always verify lock existence, even during rate limiting
            current_value = self.redis.get(self.key)
            if not current_value:
                self._last_refresh = 0
                return False

            stored_owner = (
                current_value.decode("utf-8")
                if isinstance(current_value, bytes)
                else str(current_value)
            )
            if stored_owner != self.owner_id:
                self._last_refresh = 0
                return False

            # If rate limited, return True but don't update TTL or timestamp
            if is_rate_limited:
                return True

            # Perform actual refresh
            if self.redis.expire(self.key, self.timeout):
                self._last_refresh = current_time
                return True

            self._last_refresh = 0
            return False

        except Exception as e:
            logger.error(f"ClusterLock.refresh failed for key {self.key}: {e}")
            self._last_refresh = 0
            return False

    def release(self):
        """Release the lock."""
        if self._last_refresh == 0:
            return

        try:
            self.redis.delete(self.key)
        except Exception:
            pass

        self._last_refresh = 0.0
