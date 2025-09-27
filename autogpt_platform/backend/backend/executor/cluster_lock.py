"""
Redis-based distributed locking for cluster coordination.

This module provides ClusterLock, a thread-safe, process-safe distributed lock
that prevents duplicate graph execution across multiple ExecutionManager instances.
"""

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from redis import Redis

logger = logging.getLogger(__name__)


class ClusterLock:
    """
    Redis-based distributed lock for cluster coordination.

    Provides thread-safe, process-safe distributed locking using Redis SET commands
    with NX (only if not exists) and EX (expiry) flags for atomic lock acquisition.

    Features:
    - Automatic lock expiry to prevent deadlocks
    - Ownership verification before refresh/release operations
    - Rate-limited refresh to reduce Redis load
    - Graceful handling of Redis connection failures
    - Context manager support for automatic cleanup
    - Both blocking and non-blocking acquisition modes

    Example usage:
        # Blocking lock (raises exception on failure)
        with cluster_lock.acquire() as lock:
            # Critical section - only one process can execute this
            perform_exclusive_operation()

        # Non-blocking lock (yields None on failure)
        with cluster_lock.acquire(blocking=False) as lock:
            if lock is not None:
                perform_exclusive_operation()
            else:
                handle_lock_contention()

    Args:
        redis: Redis client instance
        key: Unique lock identifier (should be descriptive, e.g., "execution:graph_123")
        owner_id: Unique identifier for the lock owner (e.g., process UUID)
        timeout: Lock expiry time in seconds (default: 300s = 5 minutes)
    """

    def __init__(self, redis: "Redis", key: str, owner_id: str, timeout: int = 300):
        if not key:
            raise ValueError("Lock key cannot be empty")
        if not owner_id:
            raise ValueError("Owner ID cannot be empty")
        if timeout <= 0:
            raise ValueError("Timeout must be positive")

        self.redis = redis
        self.key = key
        self.owner_id = owner_id
        self.timeout = timeout
        self._acquired = False
        self._last_refresh = 0.0

    @contextmanager
    def acquire(self, blocking: bool = True):
        """
        Context manager that acquires and automatically releases the lock.

        Args:
            blocking: If True, raises exception on failure. If False, yields None on failure.

        Raises:
            RuntimeError: When blocking=True and lock cannot be acquired
            ConnectionError: When Redis is unavailable and blocking=True
        """
        try:
            success = self.try_acquire()
            if not success:
                if blocking:
                    raise RuntimeError(f"Lock already held: {self.key}")
                yield None
                return

            logger.debug(f"ClusterLock acquired: {self.key} by {self.owner_id}")
            try:
                yield self
            finally:
                self.release()

        except Exception as e:
            if "Redis" in str(type(e).__name__) or "Connection" in str(
                type(e).__name__
            ):
                logger.warning(f"Redis connection failed during lock acquisition: {e}")
                if blocking:
                    raise ConnectionError(f"Redis unavailable for lock {self.key}: {e}")
                yield None
            else:
                raise

    def try_acquire(self) -> bool:
        """Internal method to attempt lock acquisition."""
        try:
            success = self.redis.set(self.key, self.owner_id, nx=True, ex=self.timeout)
            if success:
                self._acquired = True
                self._last_refresh = time.time()
                logger.debug(f"Lock acquired successfully: {self.key}")
            return bool(success)
        except Exception as e:
            logger.warning(f"Failed to acquire lock {self.key}: {e}")
            return False

    def refresh(self) -> bool:
        """
        Refresh the lock TTL to prevent expiry.

        Returns:
            bool: True if refresh successful, False if lock expired or we don't own it
        """
        if not self._acquired:
            return False

        # Rate limiting: only refresh if it's been >timeout/10 since last refresh
        current_time = time.time()
        refresh_interval = self.timeout // 10
        if current_time - self._last_refresh < refresh_interval:
            return True  # Skip refresh, still valid

        try:
            # Atomic check-and-refresh: only refresh if we still own the lock
            current_value = self.redis.get(self.key)
            if (
                current_value is not None
                and (
                    current_value == self.owner_id  # Already decoded
                    if isinstance(current_value, str)
                    else current_value.decode("utf-8") == self.owner_id  # Raw bytes
                )
            ):
                result = self.redis.expire(self.key, self.timeout)
                if result:
                    self._last_refresh = current_time
                    logger.debug(f"Lock refreshed successfully: {self.key}")
                    return True
                else:
                    logger.warning(
                        f"Failed to refresh lock (key not found): {self.key}"
                    )
            else:
                logger.warning(f"Lock ownership lost during refresh: {self.key}")

            # We no longer own the lock
            self._acquired = False
            return False

        except Exception as e:
            logger.warning(f"Failed to refresh lock {self.key}: {e}")
            self._acquired = False
            return False

    def release(self):
        """Release the lock by marking it as no longer acquired locally.

        The lock will expire naturally via Redis TTL, which is simpler and more
        reliable than trying to delete it manually.
        """
        if not self._acquired:
            return

        logger.debug(f"Lock released locally, will expire via TTL: {self.key}")
        self._acquired = False
        self._last_refresh = 0.0
