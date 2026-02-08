"""
Caching utilities for the AutoGPT platform.

Provides decorators for caching function results with support for:
- In-memory caching with TTL
- Shared Redis-backed caching across processes
- Thread-local caching for request-scoped data
- Thundering herd protection
- LRU eviction with optional TTL refresh
"""

import asyncio
import inspect
import logging
import pickle
import threading
import time
from dataclasses import dataclass
from functools import cache, wraps
from typing import Any, Callable, ParamSpec, Protocol, TypeVar, cast, runtime_checkable

from redis import ConnectionPool, Redis

from backend.util.retry import conn_retry
from backend.util.settings import Settings

P = ParamSpec("P")
R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)
T = TypeVar("T")

logger = logging.getLogger(__name__)
settings = Settings()

# RECOMMENDED REDIS CONFIGURATION FOR PRODUCTION:
# Configure Redis with the following settings for optimal caching performance:
#   maxmemory-policy allkeys-lru    # Evict least recently used keys when memory limit reached
#   maxmemory 2gb                   # Set memory limit (adjust based on your needs)
#   save ""                         # Disable persistence if using Redis purely for caching


@cache
def _get_cache_pool() -> ConnectionPool:
    """Get or create a connection pool for cache operations (lazy, thread-safe)."""
    return ConnectionPool(
        host=settings.config.redis_host,
        port=settings.config.redis_port,
        password=settings.config.redis_password or None,
        decode_responses=False,  # Binary mode for pickle
        max_connections=50,
        socket_keepalive=True,
        socket_connect_timeout=5,
        retry_on_timeout=True,
    )


@cache
@conn_retry("Redis", "Acquiring cache connection")
def _get_redis() -> Redis:
    """
    Get the lazily-initialized Redis client for shared cache operations.
    Uses @cache for thread-safe singleton behavior - connection is only
    established when first accessed, allowing services that only use
    in-memory caching to work without Redis configuration.
    """
    r = Redis(connection_pool=_get_cache_pool())
    r.ping()  # Verify connection
    return r


@dataclass
class CachedValue:
    """Wrapper for cached values with timestamp to avoid tuple ambiguity."""

    result: Any
    timestamp: float


def _make_hashable_key(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, ...]:
    """
    Convert args and kwargs into a hashable cache key.

    Handles unhashable types like dict, list, set by converting them to
    their sorted string representations.
    """

    def make_hashable(obj: Any) -> Any:
        """Recursively convert an object to a hashable representation."""
        if isinstance(obj, dict):
            # Sort dict items to ensure consistent ordering
            return (
                "__dict__",
                tuple(sorted((k, make_hashable(v)) for k, v in obj.items())),
            )
        elif isinstance(obj, (list, tuple)):
            return ("__list__", tuple(make_hashable(item) for item in obj))
        elif isinstance(obj, set):
            return ("__set__", tuple(sorted(make_hashable(item) for item in obj)))
        elif hasattr(obj, "__dict__"):
            # Handle objects with __dict__ attribute
            return ("__obj__", obj.__class__.__name__, make_hashable(obj.__dict__))
        else:
            # For basic hashable types (str, int, bool, None, etc.)
            try:
                hash(obj)
                return obj
            except TypeError:
                # Fallback: convert to string representation
                return ("__str__", str(obj))

    hashable_args = tuple(make_hashable(arg) for arg in args)
    hashable_kwargs = tuple(sorted((k, make_hashable(v)) for k, v in kwargs.items()))
    return (hashable_args, hashable_kwargs)


def _make_redis_key(key: tuple[Any, ...], func_name: str) -> str:
    """Convert a hashable key tuple to a Redis key string."""
    # Ensure key is already hashable
    hashable_key = key if isinstance(key, tuple) else (key,)
    return f"cache:{func_name}:{hash(hashable_key)}"


@runtime_checkable
class CachedFunction(Protocol[P, R_co]):
    """Protocol for cached functions with cache management methods."""

    def cache_clear(self, pattern: str | None = None) -> None:
        """Clear cached entries. If pattern provided, clear matching entries only."""
        return None

    def cache_info(self) -> dict[str, int | None]:
        """Get cache statistics."""
        return {}

    def cache_delete(self, *args: P.args, **kwargs: P.kwargs) -> bool:
        """Delete a specific cache entry by its arguments. Returns True if entry existed."""
        return False

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co:
        """Call the cached function."""
        return None  # type: ignore


def cached(
    *,
    maxsize: int = 128,
    ttl_seconds: int,
    shared_cache: bool = False,
    refresh_ttl_on_get: bool = False,
) -> Callable[[Callable[P, R]], CachedFunction[P, R]]:
    """
    Thundering herd safe cache decorator for both sync and async functions.

    Uses double-checked locking to prevent multiple threads/coroutines from
    executing the expensive operation simultaneously during cache misses.

    Args:
        maxsize: Maximum number of cached entries (only for in-memory cache)
        ttl_seconds: Time to live in seconds. Required - entries must expire.
        shared_cache: If True, use Redis for cross-process caching
        refresh_ttl_on_get: If True, refresh TTL when cache entry is accessed (LRU behavior)

    Returns:
        Decorated function with caching capabilities

    Example:
        @cached(ttl_seconds=300)  # 5 minute TTL
        def expensive_sync_operation(param: str) -> dict:
            return {"result": param}

        @cached(ttl_seconds=600, shared_cache=True, refresh_ttl_on_get=True)
        async def expensive_async_operation(param: str) -> dict:
            return {"result": param}
    """

    def decorator(target_func: Callable[P, R]) -> CachedFunction[P, R]:
        cache_storage: dict[tuple, CachedValue] = {}
        _event_loop_locks: dict[Any, asyncio.Lock] = {}

        def _get_from_redis(redis_key: str) -> Any | None:
            """Get value from Redis, optionally refreshing TTL."""
            try:
                if refresh_ttl_on_get:
                    # Use GETEX to get value and refresh expiry atomically
                    cached_bytes = _get_redis().getex(redis_key, ex=ttl_seconds)
                else:
                    cached_bytes = _get_redis().get(redis_key)

                if cached_bytes and isinstance(cached_bytes, bytes):
                    return pickle.loads(cached_bytes)
            except Exception as e:
                logger.error(
                    f"Redis error during cache check for {target_func.__name__}: {e}"
                )
            return None

        def _set_to_redis(redis_key: str, value: Any) -> None:
            """Set value in Redis with TTL."""
            try:
                pickled_value = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                _get_redis().setex(redis_key, ttl_seconds, pickled_value)
            except Exception as e:
                logger.error(
                    f"Redis error storing cache for {target_func.__name__}: {e}"
                )

        def _get_from_memory(key: tuple) -> Any | None:
            """Get value from in-memory cache, checking TTL."""
            if key in cache_storage:
                cached_data = cache_storage[key]
                if time.time() - cached_data.timestamp < ttl_seconds:
                    logger.debug(
                        f"Cache hit for {target_func.__name__} args: {key[0]} kwargs: {key[1]}"
                    )
                    return cached_data.result
            return None

        def _set_to_memory(key: tuple, value: Any) -> None:
            """Set value in in-memory cache with timestamp."""
            cache_storage[key] = CachedValue(result=value, timestamp=time.time())

            # Cleanup if needed
            if len(cache_storage) > maxsize:
                cutoff = maxsize // 2
                oldest_keys = list(cache_storage.keys())[:-cutoff] if cutoff > 0 else []
                for old_key in oldest_keys:
                    cache_storage.pop(old_key, None)

        if inspect.iscoroutinefunction(target_func):

            def _get_cache_lock():
                """Get or create an asyncio.Lock for the current event loop."""
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop not in _event_loop_locks:
                    return _event_loop_locks.setdefault(loop, asyncio.Lock())
                return _event_loop_locks[loop]

            @wraps(target_func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs):
                key = _make_hashable_key(args, kwargs)
                redis_key = (
                    _make_redis_key(key, target_func.__name__) if shared_cache else ""
                )

                # Fast path: check cache without lock
                if shared_cache:
                    result = _get_from_redis(redis_key)
                    if result is not None:
                        return result
                else:
                    result = _get_from_memory(key)
                    if result is not None:
                        return result

                # Slow path: acquire lock for cache miss/expiry
                async with _get_cache_lock():
                    # Double-check: another coroutine might have populated cache
                    if shared_cache:
                        result = _get_from_redis(redis_key)
                        if result is not None:
                            return result
                    else:
                        result = _get_from_memory(key)
                        if result is not None:
                            return result

                    # Cache miss - execute function
                    logger.debug(f"Cache miss for {target_func.__name__}")
                    result = await target_func(*args, **kwargs)

                    # Store result
                    if shared_cache:
                        _set_to_redis(redis_key, result)
                    else:
                        _set_to_memory(key, result)

                    return result

            wrapper = async_wrapper

        else:
            # Sync function with threading.Lock
            cache_lock = threading.Lock()

            @wraps(target_func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs):
                key = _make_hashable_key(args, kwargs)
                redis_key = (
                    _make_redis_key(key, target_func.__name__) if shared_cache else ""
                )

                # Fast path: check cache without lock
                if shared_cache:
                    result = _get_from_redis(redis_key)
                    if result is not None:
                        return result
                else:
                    result = _get_from_memory(key)
                    if result is not None:
                        return result

                # Slow path: acquire lock for cache miss/expiry
                with cache_lock:
                    # Double-check: another thread might have populated cache
                    if shared_cache:
                        result = _get_from_redis(redis_key)
                        if result is not None:
                            return result
                    else:
                        result = _get_from_memory(key)
                        if result is not None:
                            return result

                    # Cache miss - execute function
                    logger.debug(f"Cache miss for {target_func.__name__}")
                    result = target_func(*args, **kwargs)

                    # Store result
                    if shared_cache:
                        _set_to_redis(redis_key, result)
                    else:
                        _set_to_memory(key, result)

                    return result

            wrapper = sync_wrapper

        # Add cache management methods
        def cache_clear(pattern: str | None = None) -> None:
            """Clear cache entries. If pattern provided, clear matching entries."""
            if shared_cache:
                if pattern:
                    # Clear entries matching pattern
                    keys = list(
                        _get_redis().scan_iter(
                            f"cache:{target_func.__name__}:{pattern}"
                        )
                    )
                else:
                    # Clear all cache keys
                    keys = list(
                        _get_redis().scan_iter(f"cache:{target_func.__name__}:*")
                    )

                if keys:
                    pipeline = _get_redis().pipeline()
                    for key in keys:
                        pipeline.delete(key)
                    pipeline.execute()
            else:
                if pattern:
                    # For in-memory cache, pattern matching not supported
                    logger.warning(
                        "Pattern-based clearing not supported for in-memory cache"
                    )
                else:
                    cache_storage.clear()

        def cache_info() -> dict[str, int | None]:
            if shared_cache:
                cache_keys = list(
                    _get_redis().scan_iter(f"cache:{target_func.__name__}:*")
                )
                return {
                    "size": len(cache_keys),
                    "maxsize": None,  # Redis manages its own size
                    "ttl_seconds": ttl_seconds,
                }
            else:
                return {
                    "size": len(cache_storage),
                    "maxsize": maxsize,
                    "ttl_seconds": ttl_seconds,
                }

        def cache_delete(*args, **kwargs) -> bool:
            """Delete a specific cache entry. Returns True if entry existed."""
            key = _make_hashable_key(args, kwargs)
            if shared_cache:
                redis_key = _make_redis_key(key, target_func.__name__)
                deleted_count = cast(int, _get_redis().delete(redis_key))
                return deleted_count > 0
            else:
                if key in cache_storage:
                    del cache_storage[key]
                    return True
                return False

        setattr(wrapper, "cache_clear", cache_clear)
        setattr(wrapper, "cache_info", cache_info)
        setattr(wrapper, "cache_delete", cache_delete)

        return cast(CachedFunction[P, R], wrapper)

    return decorator


def thread_cached(func):
    """
    Thread-local cache decorator for both sync and async functions.

    Each thread gets its own cache, which is useful for request-scoped caching
    in web applications where you want to cache within a single request but
    not across requests.

    Args:
        func: The function to cache

    Returns:
        Decorated function with thread-local caching

    Example:
        @thread_cached
        def expensive_operation(param: str) -> dict:
            return {"result": param}

        @thread_cached  # Works with async too
        async def expensive_async_operation(param: str) -> dict:
            return {"result": param}
    """
    thread_local = threading.local()

    def _clear():
        if hasattr(thread_local, "cache"):
            del thread_local.cache

    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = getattr(thread_local, "cache", None)
            if cache is None:
                cache = thread_local.cache = {}
            key = _make_hashable_key(args, kwargs)
            if key not in cache:
                cache[key] = await func(*args, **kwargs)
            return cache[key]

        setattr(async_wrapper, "clear_cache", _clear)
        return async_wrapper

    else:

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache = getattr(thread_local, "cache", None)
            if cache is None:
                cache = thread_local.cache = {}
            key = _make_hashable_key(args, kwargs)
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]

        setattr(sync_wrapper, "clear_cache", _clear)
        return sync_wrapper


def clear_thread_cache(func: Callable) -> None:
    """Clear thread-local cache for a function."""
    if clear := getattr(func, "clear_cache", None):
        clear()
