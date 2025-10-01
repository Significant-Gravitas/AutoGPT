import asyncio
import inspect
import logging
import os
import pickle
import threading
import time
from functools import wraps
from typing import (
    Any,
    Callable,
    ParamSpec,
    Protocol,
    TypeVar,
    cast,
    runtime_checkable,
)

from dotenv import load_dotenv
from redis import ConnectionPool, Redis

from autogpt_libs.utils.retry import conn_retry

P = ParamSpec("P")
R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)

logger = logging.getLogger(__name__)


load_dotenv()
HOST = os.getenv("REDIS_HOST", "localhost")
PORT = int(os.getenv("REDIS_PORT", "6379"))
PASSWORD = os.getenv("REDIS_PASSWORD", None)

logger = logging.getLogger(__name__)


@conn_retry("Redis", "Acquiring connection")
def connect() -> Redis:
    # Configure connection pool for optimal performance
    pool = ConnectionPool(
        host=HOST,
        port=PORT,
        password=PASSWORD,
        decode_responses=False,  # Store binary data for pickle
        max_connections=50,  # Allow up to 50 concurrent connections
        socket_keepalive=True,  # Keep connections alive
        socket_connect_timeout=5,
        retry_on_timeout=True,
    )
    c = Redis(connection_pool=pool)
    c.ping()
    return c


redis_client = connect()


def _make_redis_key(key: tuple[Any, ...]) -> str:
    """Convert a hashable key tuple to a Redis key string."""
    return f"cache:{hash(key)}"


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


@runtime_checkable
class CachedFunction(Protocol[P, R_co]):
    """Protocol for cached functions with cache management methods."""

    def cache_clear(self) -> None:
        """Clear all cached entries."""
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
    ttl_seconds: int | None = None,
    shared_cache: bool = False,
) -> Callable[[Callable], CachedFunction]:
    """
    Thundering herd safe cache decorator for both sync and async functions.

    Uses double-checked locking to prevent multiple threads/coroutines from
    executing the expensive operation simultaneously during cache misses.

    Args:
        func: The function to cache (when used without parentheses)
        maxsize: Maximum number of cached entries
        ttl_seconds: Time to live in seconds. If None, entries never expire

    Returns:
        Decorated function or decorator

    Example:
        @cache()  # Default: maxsize=128, no TTL
        def expensive_sync_operation(param: str) -> dict:
            return {"result": param}

        @cache()  # Works with async too
        async def expensive_async_operation(param: str) -> dict:
            return {"result": param}

        @cache(maxsize=1000, ttl_seconds=300)  # Custom maxsize and TTL
        def another_operation(param: str) -> dict:
            return {"result": param}
    """

    def decorator(target_func):
        cache_storage = {}
        _event_loop_locks = {}  # Maps event loop to its asyncio.Lock

        if inspect.iscoroutinefunction(target_func):

            def _get_cache_lock():
                """Get or create an asyncio.Lock for the current event loop."""
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    # No event loop, use None as default key
                    loop = None

                if loop not in _event_loop_locks:
                    return _event_loop_locks.setdefault(loop, asyncio.Lock())
                return _event_loop_locks[loop]

            @wraps(target_func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs):
                key = _make_hashable_key(args, kwargs)
                current_time = time.time()

                # Compute redis_key once if using shared cache
                redis_key = _make_redis_key(key)

                # Fast path: check cache without lock
                if shared_cache:
                    try:
                        # Use GET directly instead of EXISTS + GET (saves 1 round trip)
                        cached_bytes = redis_client.get(redis_key)
                        if cached_bytes is not None and isinstance(cached_bytes, bytes):
                            logger.debug(
                                f"Cache hit for {target_func.__name__} args: {args} kwargs: {kwargs}"
                            )
                            return pickle.loads(cached_bytes)
                    except Exception as e:
                        logger.error(
                            f"Redis error during cache check for {target_func.__name__}: {e}"
                        )
                        # Fall through to execute function
                        return await target_func(*args, **kwargs)
                else:
                    if key in cache_storage:
                        if ttl_seconds is None:
                            logger.debug(
                                f"Cache hit for {target_func.__name__} args: {args} kwargs: {kwargs}"
                            )
                            return cache_storage[key]
                        else:
                            cached_data = cache_storage[key]
                            if isinstance(cached_data, tuple):
                                result, timestamp = cached_data
                                if current_time - timestamp < ttl_seconds:
                                    logger.debug(
                                        f"Cache hit for {target_func.__name__} args: {args} kwargs: {kwargs}"
                                    )
                                    return result

                # Slow path: acquire lock for cache miss/expiry
                async with _get_cache_lock():
                    # Double-check: another coroutine might have populated cache
                    if shared_cache:
                        try:
                            # Use GET directly (saves 1 round trip)
                            cached_bytes = redis_client.get(redis_key)
                            if cached_bytes is not None and isinstance(
                                cached_bytes, bytes
                            ):
                                return pickle.loads(cached_bytes)
                        except Exception as e:
                            logger.error(
                                f"Redis error during double-check for {target_func.__name__}: {e}"
                            )
                            # Continue to execute function
                    else:
                        if key in cache_storage:
                            if ttl_seconds is None:
                                return cache_storage[key]
                            else:
                                cached_data = cache_storage[key]
                                if isinstance(cached_data, tuple):
                                    result, timestamp = cached_data
                                    if current_time - timestamp < ttl_seconds:
                                        return result

                    # Cache miss - execute function
                    logger.debug(f"Cache miss for {target_func.__name__}")
                    result = await target_func(*args, **kwargs)

                    # Store result
                    if shared_cache:
                        try:
                            pickled_result = pickle.dumps(
                                result, protocol=pickle.HIGHEST_PROTOCOL
                            )
                            if ttl_seconds is None:
                                redis_client.set(redis_key, pickled_result)
                            else:
                                redis_client.setex(
                                    redis_key, ttl_seconds, pickled_result
                                )
                        except Exception as e:
                            logger.error(
                                f"Redis error storing cache for {target_func.__name__}: {e}"
                            )
                            # Continue without caching
                    else:
                        if ttl_seconds is None:
                            cache_storage[key] = result
                        else:
                            cache_storage[key] = (result, current_time)

                    # Cleanup if needed (only for local cache)
                    if not shared_cache and len(cache_storage) > maxsize:
                        cutoff = maxsize // 2
                        oldest_keys = (
                            list(cache_storage.keys())[:-cutoff] if cutoff > 0 else []
                        )
                        for old_key in oldest_keys:
                            cache_storage.pop(old_key, None)

                    return result

            wrapper = async_wrapper

        else:
            # Sync function with threading.Lock
            cache_lock = threading.Lock()

            @wraps(target_func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs):
                key = _make_hashable_key(args, kwargs)
                current_time = time.time()

                # Compute redis_key once if using shared cache
                redis_key = _make_redis_key(key)

                # Fast path: check cache without lock
                if shared_cache:
                    try:
                        # Use GET directly instead of EXISTS + GET (saves 1 round trip)
                        cached_bytes = redis_client.get(redis_key)
                        if cached_bytes is not None and isinstance(cached_bytes, bytes):
                            logger.debug(
                                f"Cache hit for {target_func.__name__} args: {args} kwargs: {kwargs}"
                            )
                            return pickle.loads(cached_bytes)
                    except Exception as e:
                        logger.error(
                            f"Redis error during cache check for {target_func.__name__}: {e}"
                        )
                        # Fall through to execute function
                        return target_func(*args, **kwargs)
                else:
                    if key in cache_storage:
                        if ttl_seconds is None:
                            logger.debug(
                                f"Cache hit for {target_func.__name__} args: {args} kwargs: {kwargs}"
                            )
                            return cache_storage[key]
                        else:
                            cached_data = cache_storage[key]
                            if isinstance(cached_data, tuple):
                                result, timestamp = cached_data
                                if current_time - timestamp < ttl_seconds:
                                    logger.debug(
                                        f"Cache hit for {target_func.__name__} args: {args} kwargs: {kwargs}"
                                    )
                                    return result

                # Slow path: acquire lock for cache miss/expiry
                with cache_lock:
                    # Double-check: another thread might have populated cache
                    if shared_cache:
                        try:
                            # Use GET directly (saves 1 round trip)
                            cached_bytes = redis_client.get(redis_key)
                            if cached_bytes is not None and isinstance(
                                cached_bytes, bytes
                            ):
                                return pickle.loads(cached_bytes)
                        except Exception as e:
                            logger.error(
                                f"Redis error during double-check for {target_func.__name__}: {e}"
                            )
                            # Continue to execute function
                    else:
                        if key in cache_storage:
                            if ttl_seconds is None:
                                return cache_storage[key]
                            else:
                                cached_data = cache_storage[key]
                                if isinstance(cached_data, tuple):
                                    result, timestamp = cached_data
                                    if current_time - timestamp < ttl_seconds:
                                        return result

                    # Cache miss - execute function
                    logger.debug(f"Cache miss for {target_func.__name__}")
                    result = target_func(*args, **kwargs)

                    # Store result
                    if shared_cache:
                        try:
                            pickled_result = pickle.dumps(
                                result, protocol=pickle.HIGHEST_PROTOCOL
                            )
                            if ttl_seconds is None:
                                redis_client.set(redis_key, pickled_result)
                            else:
                                redis_client.setex(
                                    redis_key, ttl_seconds, pickled_result
                                )
                        except Exception as e:
                            logger.error(
                                f"Redis error storing cache for {target_func.__name__}: {e}"
                            )
                            # Continue without caching
                    else:
                        if ttl_seconds is None:
                            cache_storage[key] = result
                        else:
                            cache_storage[key] = (result, current_time)

                    # Cleanup if needed (only for local cache)
                    if not shared_cache and len(cache_storage) > maxsize:
                        cutoff = maxsize // 2
                        oldest_keys = (
                            list(cache_storage.keys())[:-cutoff] if cutoff > 0 else []
                        )
                        for old_key in oldest_keys:
                            cache_storage.pop(old_key, None)

                    return result

            wrapper = sync_wrapper

        # Add cache management methods
        def cache_clear() -> None:
            if shared_cache:
                # Clear only cache keys (prefixed with "cache:") using pipeline for efficiency
                keys = list(redis_client.scan_iter("cache:*", count=100))
                if keys:
                    pipeline = redis_client.pipeline()
                    for key in keys:
                        pipeline.delete(key)
                    pipeline.execute()
            else:
                cache_storage.clear()

        def cache_info() -> dict[str, int | None]:
            if shared_cache:
                # Count only cache keys
                cache_keys = list(redis_client.scan_iter("cache:*"))
                return {
                    "size": len(cache_keys),
                    "maxsize": maxsize,
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
                redis_key = _make_redis_key(key)
                if redis_client.exists(redis_key):
                    redis_client.delete(redis_key)
                    return True
                return False
            else:
                if key in cache_storage:
                    del cache_storage[key]
                    return True
                return False

        setattr(wrapper, "cache_clear", cache_clear)
        setattr(wrapper, "cache_info", cache_info)
        setattr(wrapper, "cache_delete", cache_delete)

        return cast(CachedFunction, wrapper)

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
