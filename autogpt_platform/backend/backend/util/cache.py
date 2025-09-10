"""
In-memory TTL cache with O(1) operations for FastAPI endpoints.

This module provides a thread-safe, memory-efficient cache implementation
that uses function code and input parameters for cache key generation.
"""

import hashlib
import inspect
import json
import logging
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

    def __init__(self, default_ttl: int = 3600, max_size: Optional[int] = None):
        """
        Initialize the TTL cache.

        Args:
            default_ttl: Default time-to-live in seconds (default: 3600)
            max_size: Maximum number of entries (None for unlimited)
        """
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._default_ttl = default_ttl
        self._max_size = max_size
        # Using RLock for reentrant locking (same thread can acquire multiple times)
        # For true read-write optimization, we'd need threading.RWLock (not in stdlib)
        # or a third-party library. For now, RLock provides thread safety.
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._cleanup_interval = 60  # Cleanup expired entries every 60 seconds

    def _cleanup_expired(self) -> None:
        """Remove expired entries from the cache."""
        current_time = time.time()

        # Only cleanup periodically to avoid overhead
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        expired_keys = [
            key for key, (_, expiry) in self._cache.items() if expiry < current_time
        ]

        if expired_keys:
            logger.debug(
                f"[CACHE CLEANUP] Removing {len(expired_keys)} expired entries"
            )
            for key in expired_keys:
                del self._cache[key]
                logger.debug(f"[CACHE EXPIRED] Removed expired entry: {key[:16]}...")

        self._last_cleanup = current_time

    def _evict_if_needed(self) -> None:
        """Evict oldest entries if cache exceeds max_size."""
        if self._max_size and len(self._cache) > self._max_size:
            # Remove oldest entries until we're at max_size
            evicted = 0
            while len(self._cache) > self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                evicted += 1
                logger.info(
                    f"[CACHE EVICT] Evicted entry due to size limit: {evicted_key[:16]}..."
                )
            if evicted > 0:
                logger.info(
                    f"[CACHE EVICT] Evicted {evicted} entries to maintain max_size={self._max_size}"
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

            value, expiry = self._cache[key]

            current_time = time.time()
            if expiry < current_time:
                # Entry has expired - need to modify cache
                del self._cache[key]
                logger.info(f"[CACHE EXPIRED] Entry expired and removed: {key[:16]}...")
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

            # Check if this is an update or new entry
            is_update = key in self._cache

            # Add/update the entry
            self._cache[key] = (value, expiry)
            self._cache.move_to_end(key)

            if is_update:
                logger.info(
                    f"[CACHE UPDATE] Updated entry: {key[:16]}... (TTL: {ttl}s)"
                )
            else:
                logger.info(f"[CACHE SET] Added new entry: {key[:16]}... (TTL: {ttl}s)")

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
                1 for _, expiry in self._cache.values() if expiry >= current_time
            )
            return {
                "total_entries": len(self._cache),
                "valid_entries": valid_entries,
                "expired_entries": len(self._cache) - valid_entries,
                "max_size": self._max_size,
            }


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
