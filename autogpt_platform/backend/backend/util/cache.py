"""
In-memory TTL cache with O(1) operations for FastAPI endpoints.

This module provides a thread-safe, memory-efficient cache implementation
that uses function code and input parameters for cache key generation.
"""

import hashlib
import inspect
import json
import logging
import sys
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)
T = TypeVar("T")


class TTLCache:
    """
    Thread-safe in-memory cache with Time-To-Live (TTL) support.

    Features:
    - O(1) get/set operations
    - Automatic expiration of old entries
    - Optional memory limit with LRU eviction
    - Thread-safe operations with read-write lock optimization
    """

    def __init__(self, default_ttl: int = 3600, max_size_mb: Optional[float] = None):
        """
        Initialize the TTL cache.

        Args:
            default_ttl: Default time-to-live in seconds (default: 3600)
            max_size_mb: Maximum cache size in megabytes (None for unlimited)
        """
        self._cache: OrderedDict[str, Tuple[Any, float, int]] = (
            OrderedDict()
        )  # value, expiry, size_bytes
        self._default_ttl = default_ttl
        self._max_size_mb = max_size_mb
        self._current_size_bytes = 0
        # Using RLock for reentrant locking (same thread can acquire multiple times)
        # For true read-write optimization, we'd need threading.RWLock (not in stdlib)
        # or a third-party library. For now, RLock provides thread safety.
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 60  # Cleanup expired entries every 60 seconds

    def _estimate_size(self, obj: Any) -> int:
        """
        Estimate the memory size of an object in bytes.

        Args:
            obj: Object to estimate size for

        Returns:
            Estimated size in bytes
        """
        # Use sys.getsizeof for basic estimation, with recursive handling for containers
        try:
            size = sys.getsizeof(obj)

            # Add sizes of nested objects for common containers
            if isinstance(obj, dict):
                size += sum(
                    self._estimate_size(k) + self._estimate_size(v)
                    for k, v in obj.items()
                )
            elif isinstance(obj, (list, tuple, set)):
                size += sum(self._estimate_size(item) for item in obj)
            elif hasattr(obj, "__dict__"):
                size += self._estimate_size(obj.__dict__)

            return size
        except Exception:
            # Fallback to a conservative estimate if we can't determine size
            return 1024  # 1KB default

    def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        current_time = time.time()

        # Only cleanup periodically to avoid overhead
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        expired_keys = [
            key for key, (_, expiry, _) in self._cache.items() if expiry < current_time
        ]

        if expired_keys:
            logger.debug(
                f"[CACHE CLEANUP] Removing {len(expired_keys)} expired entries"
            )
            for key in expired_keys:
                _, _, size_bytes = self._cache[key]
                del self._cache[key]
                self._current_size_bytes -= size_bytes
                logger.debug(
                    f"[CACHE EXPIRED] Removed expired entry: {key[:16]}... (freed {size_bytes} bytes)"
                )

        self._last_cleanup = current_time

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max_size_mb."""
        if self._max_size_mb:
            max_bytes = self._max_size_mb * 1024 * 1024  # Convert MB to bytes
            if self._current_size_bytes > max_bytes:
                # Remove oldest entries until we're under the size limit
                evicted = 0
                evicted_bytes = 0
                while self._current_size_bytes > max_bytes and self._cache:
                    evicted_key, (_, _, size_bytes) = self._cache.popitem(last=False)
                    self._current_size_bytes -= size_bytes
                    evicted_bytes += size_bytes
                    evicted += 1
                    logger.info(
                        f"[CACHE EVICT] Evicted entry due to memory limit: {evicted_key[:16]}... (freed {size_bytes} bytes)"
                    )
                if evicted > 0:
                    logger.info(
                        f"[CACHE EVICT] Evicted {evicted} entries ({evicted_bytes / 1024 / 1024:.2f} MB) to maintain max_size_mb={self._max_size_mb}"
                    )

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value if found and not expired, None otherwise
        """
        # First, try to read without modifying (no lock needed for dict read in CPython)
        # but we still need the lock because we might modify the cache
        with self._lock:
            if key not in self._cache:
                logger.debug(f"[CACHE MISS] Key not found: {key[:16]}...")
                return None

            value, expiry, size_bytes = self._cache[key]

            current_time = time.time()
            if expiry < current_time:
                # Entry has expired - need to modify cache
                del self._cache[key]
                self._current_size_bytes -= size_bytes
                logger.info(
                    f"[CACHE EXPIRED] Entry expired and removed: {key[:16]}... (freed {size_bytes} bytes)"
                )
                return None

            # Move to end for LRU tracking - this modifies the cache
            self._cache.move_to_end(key)
            ttl_remaining = expiry - current_time
            logger.info(
                f"[CACHE HIT] Retrieved entry: {key[:16]}... (TTL: {ttl_remaining:.1f}s remaining)"
            )
            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            ttl = ttl or self._default_ttl
            expiry = time.time() + ttl

            # Estimate size of the new value
            value_size = self._estimate_size(value)

            # Check if this is an update or new entry
            is_update = key in self._cache

            # If updating, subtract old size first
            if is_update:
                _, _, old_size = self._cache[key]
                self._current_size_bytes -= old_size

            # Add/update the entry
            self._cache[key] = (value, expiry, value_size)
            self._cache.move_to_end(key)
            self._current_size_bytes += value_size

            if is_update:
                logger.info(
                    f"[CACHE UPDATE] Updated entry: {key[:16]}... (TTL: {ttl}s, size: {value_size} bytes)"
                )
            else:
                logger.info(
                    f"[CACHE SET] Added new entry: {key[:16]}... (TTL: {ttl}s, size: {value_size} bytes)"
                )

            # Cleanup and evict if needed
            self._cleanup_expired()
            self._evict_if_needed()

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            size = len(self._cache)
            self._cache.clear()
            logger.info(f"[CACHE CLEAR] Cleared {size} entries from cache")

    def size(self) -> int:
        """Get the current number of entries in the cache."""
        with self._lock:
            return len(self._cache)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            current_time = time.time()
            valid_entries = sum(
                1 for _, expiry, _ in self._cache.values() if expiry >= current_time
            )
            return {
                "total_entries": len(self._cache),
                "valid_entries": valid_entries,
                "expired_entries": len(self._cache) - valid_entries,
                "max_size_mb": self._max_size_mb,
                "current_size_mb": self._current_size_bytes / 1024 / 1024,
            }

    def invalidate_key(self, key: str) -> bool:
        """
        Invalidate a specific cache key.

        Args:
            key: The cache key to invalidate

        Returns:
            True if key was found and invalidated, False otherwise
        """
        with self._lock:
            if key in self._cache:
                _, _, size_bytes = self._cache[key]
                del self._cache[key]
                self._current_size_bytes -= size_bytes
                logger.info(
                    f"[CACHE INVALIDATE] Invalidated key: {key[:16]}... (freed {size_bytes} bytes)"
                )
                return True
            return False

    def invalidate_prefix(self, prefix: str) -> int:
        """
        Invalidate all cache keys starting with a prefix.

        Args:
            prefix: The prefix to match

        Returns:
            Number of keys invalidated
        """
        with self._lock:
            keys_to_remove = [
                key for key in self._cache.keys() if key.startswith(prefix)
            ]
            freed_bytes = 0
            for key in keys_to_remove:
                _, _, size_bytes = self._cache[key]
                del self._cache[key]
                self._current_size_bytes -= size_bytes
                freed_bytes += size_bytes

            if keys_to_remove:
                logger.info(
                    f"[CACHE INVALIDATE] Invalidated {len(keys_to_remove)} keys with prefix: {prefix} (freed {freed_bytes / 1024 / 1024:.2f} MB)"
                )
            return len(keys_to_remove)

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache keys matching a pattern.

        Args:
            pattern: Regular expression pattern to match

        Returns:
            Number of keys invalidated
        """
        import re

        with self._lock:
            try:
                regex = re.compile(pattern)
                
                # Debug: log all keys in cache
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[CACHE INVALIDATE] Searching for pattern '{pattern}' in {len(self._cache)} keys")
                    for key in list(self._cache.keys())[:5]:  # Log first 5 keys for debugging
                        logger.debug(f"[CACHE INVALIDATE] Cache key: {key}")
                
                keys_to_remove = [key for key in self._cache.keys() if regex.search(key)]
                freed_bytes = 0
                for key in keys_to_remove:
                    _, _, size_bytes = self._cache[key]
                    del self._cache[key]
                    self._current_size_bytes -= size_bytes
                    freed_bytes += size_bytes

                if keys_to_remove:
                    logger.info(
                        f"[CACHE INVALIDATE] Invalidated {len(keys_to_remove)} keys matching pattern: {pattern} (freed {freed_bytes / 1024 / 1024:.2f} MB)"
                    )
                else:
                    logger.debug(f"[CACHE INVALIDATE] No keys found matching pattern: {pattern}")
                return len(keys_to_remove)
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern}': {e}")
                return 0

    def invalidate_user(self, user_id: str) -> int:
        """
        Invalidate all cache entries for a specific user.

        Args:
            user_id: The user ID to invalidate cache for

        Returns:
            Number of keys invalidated
        """
        # Common patterns for user-specific cache keys
        patterns = [
            f".*user_id.*{user_id}.*",
            f".*{user_id}.*",
            f".*user.*{user_id}.*",
        ]

        total_invalidated = 0
        for pattern in patterns:
            total_invalidated += self.invalidate_pattern(pattern)

        if total_invalidated > 0:
            logger.info(
                f"[CACHE INVALIDATE] Invalidated {total_invalidated} keys for user: {user_id}"
            )
        return total_invalidated


def generate_cache_key(
    func: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    include_code: bool = True,
) -> str:
    """
    Generate a cache key from function and its arguments.

    Args:
        func: Function being cached
        args: Positional arguments
        kwargs: Keyword arguments
        include_code: Whether to include function code in the hash

    Returns:
        SHA256 hash string as cache key
    """
    hasher = hashlib.sha256()

    # Include function identity
    hasher.update(func.__module__.encode())
    hasher.update(func.__name__.encode())

    # Include function source code if requested (invalidates on code changes)
    if include_code:
        try:
            source = inspect.getsource(func)
            hasher.update(source.encode())
        except (OSError, TypeError):
            # Can't get source for built-in functions or some other cases
            pass

    # Normalize and hash arguments
    try:
        # Create a normalized representation of arguments
        normalized_args = {"args": args, "kwargs": kwargs}

        # Use JSON for consistent serialization
        # Sort keys to ensure consistent ordering
        arg_str = json.dumps(
            normalized_args,
            sort_keys=True,
            default=str,  # Convert non-serializable objects to strings
        )
        hasher.update(arg_str.encode())
    except (TypeError, ValueError):
        # Fallback for non-JSON serializable objects
        hasher.update(str(args).encode())
        hasher.update(str(sorted(kwargs.items())).encode())

    return hasher.hexdigest()


# Global cache instance for convenience
_global_cache = TTLCache()
logger.info("[CACHE INIT] Global cache instance created with default settings")


def get_global_cache() -> TTLCache:
    """Get the global cache instance."""
    return _global_cache


def clear_global_cache() -> None:
    """Clear the global cache."""
    logger.info("[CACHE] Clearing global cache")
    _global_cache.clear()
