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

P = ParamSpec("P")
R = TypeVar("R")
R_co = TypeVar("R_co", covariant=True)

logger = logging.getLogger(__name__)

# Redis client providers (can be set externally)
_redis_client_provider = None
_async_redis_client_provider = None


def set_redis_client_provider(sync_provider=None, async_provider=None):
    """
    Set external Redis client providers.

    This allows the backend to inject its Redis clients into the cache system.

    Args:
        sync_provider: A callable that returns a sync Redis client
        async_provider: A callable that returns an async Redis client
    """
    global _redis_client_provider, _async_redis_client_provider
    if sync_provider:
        _redis_client_provider = sync_provider
    if async_provider:
        _async_redis_client_provider = async_provider


def _get_redis_client():
    """Get Redis client from provider or create a default one."""
    if _redis_client_provider:
        try:
            client = _redis_client_provider()
            if client:
                return client
        except Exception as e:
            logger.warning(f"Failed to get Redis client from provider: {e}")

    # Fallback to creating our own client
    try:
        from redis import Redis

        client = Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD", None),
            decode_responses=False,  # We'll use pickle for serialization
        )
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"Failed to connect to Redis: {e}")
        return None


async def _get_async_redis_client():
    """Get async Redis client from provider or create a default one."""
    if _async_redis_client_provider:
        try:
            # Provider is an async function, we need to await it
            if inspect.iscoroutinefunction(_async_redis_client_provider):
                client = await _async_redis_client_provider()
            else:
                client = _async_redis_client_provider()
            if client:
                return client
        except Exception as e:
            logger.warning(f"Failed to get async Redis client from provider: {e}")

    # Fallback to creating our own client
    try:
        from redis.asyncio import Redis as AsyncRedis

        client = AsyncRedis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_PASSWORD", None),
            decode_responses=False,  # We'll use pickle for serialization
        )
        return client
    except Exception as e:
        logger.warning(f"Failed to create async Redis client: {e}")
        return None


def _make_redis_key(func_name: str, key: tuple) -> str:
    """Create a Redis key from function name and cache key."""
    # Convert the key to a string representation
    key_str = str(key)
    # Add a prefix to avoid collisions with other Redis usage
    return f"cache:{func_name}:{key_str}"


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

    When shared_cache=True, uses Redis for distributed caching across instances.

    Args:
        func: The function to cache (when used without parentheses)
        maxsize: Maximum number of cached entries (ignored when using Redis)
        ttl_seconds: Time to live in seconds. If None, entries never expire
        shared_cache: If True, use Redis for distributed caching

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
        # Cache storage and locks
        cache_storage = {}

        if inspect.iscoroutinefunction(target_func):
            # Async function with asyncio.Lock
            cache_lock = asyncio.Lock()

            @wraps(target_func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs):
                key = _make_hashable_key(args, kwargs)
                current_time = time.time()

                # Try Redis first if shared_cache is enabled
                if shared_cache:
                    redis_client = await _get_async_redis_client()
                    if redis_client:
                        redis_key = _make_redis_key(target_func.__name__, key)
                        try:
                            # Check Redis cache
                            await redis_client.ping()  # Ensure connection is alive
                            cached_value = await redis_client.get(redis_key)
                            if cached_value:
                                result = pickle.loads(cast(bytes, cached_value))
                                logger.info(
                                    f"Redis cache hit for {target_func.__name__}, args: {args}, kwargs: {kwargs}"
                                )
                                return result
                        except Exception as e:
                            logger.warning(f"Redis cache read failed: {e}")
                            # Fall through to execute function
                else:
                    # Fast path: check local cache without lock
                    if key in cache_storage:
                        if ttl_seconds is None:
                            logger.info(
                                f"Cache hit for {target_func.__name__}, args: {args}, kwargs: {kwargs}"
                            )
                            return cache_storage[key]
                        else:
                            cached_data = cache_storage[key]
                            if isinstance(cached_data, tuple):
                                result, timestamp = cached_data
                                if current_time - timestamp < ttl_seconds:
                                    logger.info(
                                        f"Cache hit for {target_func.__name__}, args: {args}, kwargs: {kwargs}"
                                    )
                                    return result

                # Slow path: acquire lock for cache miss/expiry
                async with cache_lock:
                    # Double-check: another coroutine might have populated cache
                    if shared_cache:
                        redis_client = await _get_async_redis_client()
                        if redis_client:
                            redis_key = _make_redis_key(target_func.__name__, key)
                            try:
                                cached_value = await redis_client.get(redis_key)
                                if cached_value:
                                    return pickle.loads(cast(bytes, cached_value))
                            except Exception as e:
                                logger.warning(f"Redis cache read failed in lock: {e}")
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
                    logger.info(f"Cache miss for {target_func.__name__}")
                    result = await target_func(*args, **kwargs)

                    # Store result
                    if shared_cache:
                        redis_client = await _get_async_redis_client()
                        if redis_client:
                            redis_key = _make_redis_key(target_func.__name__, key)
                            try:
                                serialized = pickle.dumps(result)
                                await redis_client.set(
                                    redis_key,
                                    serialized,
                                    ex=ttl_seconds if ttl_seconds else None,
                                )
                            except Exception as e:
                                logger.warning(f"Redis cache write failed: {e}")
                    else:
                        if ttl_seconds is None:
                            cache_storage[key] = result
                        else:
                            cache_storage[key] = (result, current_time)

                        # Cleanup if needed
                        if len(cache_storage) > maxsize:
                            cutoff = maxsize // 2
                            oldest_keys = (
                                list(cache_storage.keys())[:-cutoff]
                                if cutoff > 0
                                else []
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

                # Try Redis first if shared_cache is enabled
                if shared_cache:
                    redis_client = _get_redis_client()
                    if redis_client:
                        redis_key = _make_redis_key(target_func.__name__, key)
                        try:
                            # Check Redis cache
                            cached_value = redis_client.get(redis_key)
                            if cached_value:
                                result = pickle.loads(cast(bytes, cached_value))
                                logger.info(
                                    f"Redis cache hit for {target_func.__name__}, args: {args}, kwargs: {kwargs}"
                                )
                                return result
                        except Exception as e:
                            logger.warning(f"Redis cache read failed: {e}")
                            # Fall through to execute function
                else:
                    # Fast path: check local cache without lock
                    if key in cache_storage:
                        if ttl_seconds is None:
                            logger.info(
                                f"Cache hit for {target_func.__name__}, args: {args}, kwargs: {kwargs}"
                            )
                            return cache_storage[key]
                        else:
                            cached_data = cache_storage[key]
                            if isinstance(cached_data, tuple):
                                result, timestamp = cached_data
                                if current_time - timestamp < ttl_seconds:
                                    logger.info(
                                        f"Cache hit for {target_func.__name__}, args: {args}, kwargs: {kwargs}"
                                    )
                                    return result

                # Slow path: acquire lock for cache miss/expiry
                with cache_lock:
                    # Double-check: another thread might have populated cache
                    if shared_cache:
                        redis_client = _get_redis_client()
                        if redis_client:
                            redis_key = _make_redis_key(target_func.__name__, key)
                            try:
                                cached_value = redis_client.get(redis_key)
                                if cached_value:
                                    return pickle.loads(cast(bytes, cached_value))
                            except Exception as e:
                                logger.warning(f"Redis cache read failed in lock: {e}")
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
                    logger.info(f"Cache miss for {target_func.__name__}")
                    result = target_func(*args, **kwargs)

                    # Store result
                    if shared_cache:
                        redis_client = _get_redis_client()
                        if redis_client:
                            redis_key = _make_redis_key(target_func.__name__, key)
                            try:
                                serialized = pickle.dumps(result)
                                redis_client.set(
                                    redis_key,
                                    serialized,
                                    ex=ttl_seconds if ttl_seconds else None,
                                )
                            except Exception as e:
                                logger.warning(f"Redis cache write failed: {e}")
                    else:
                        if ttl_seconds is None:
                            cache_storage[key] = result
                        else:
                            cache_storage[key] = (result, current_time)

                        # Cleanup if needed
                        if len(cache_storage) > maxsize:
                            cutoff = maxsize // 2
                            oldest_keys = (
                                list(cache_storage.keys())[:-cutoff]
                                if cutoff > 0
                                else []
                            )
                            for old_key in oldest_keys:
                                cache_storage.pop(old_key, None)

                    return result

            wrapper = sync_wrapper

        # Add cache management methods
        def cache_clear() -> None:
            """Clear all cached entries."""
            if shared_cache:
                redis_client = (
                    _get_redis_client()
                    if not inspect.iscoroutinefunction(target_func)
                    else None
                )
                if redis_client:
                    try:
                        # Clear all cache entries for this function
                        pattern = f"cache:{target_func.__name__}:*"
                        for key in redis_client.scan_iter(match=pattern):
                            redis_client.delete(key)
                    except Exception as e:
                        logger.warning(f"Redis cache clear failed: {e}")
            else:
                cache_storage.clear()

        def cache_info() -> dict[str, Any]:
            """Get cache statistics."""
            if shared_cache:
                redis_client = (
                    _get_redis_client()
                    if not inspect.iscoroutinefunction(target_func)
                    else None
                )
                if redis_client:
                    try:
                        pattern = f"cache:{target_func.__name__}:*"
                        count = sum(1 for _ in redis_client.scan_iter(match=pattern))
                        return {
                            "size": count,
                            "maxsize": None,  # Not applicable for Redis
                            "ttl_seconds": ttl_seconds,
                            "shared_cache": True,
                        }
                    except Exception as e:
                        logger.warning(f"Redis cache info failed: {e}")
                        return {
                            "size": 0,
                            "maxsize": None,
                            "ttl_seconds": ttl_seconds,
                            "shared_cache": True,
                            "error": str(e),
                        }
            return {
                "size": len(cache_storage),
                "maxsize": maxsize,
                "ttl_seconds": ttl_seconds,
                "shared_cache": False,
            }

        # Create appropriate cache_delete based on whether function is async
        if inspect.iscoroutinefunction(target_func):

            async def async_cache_delete(*args, **kwargs) -> bool:
                """Delete a specific cache entry. Returns True if entry existed."""
                key = _make_hashable_key(args, kwargs)

                if shared_cache:
                    redis_client = await _get_async_redis_client()
                    if redis_client:
                        redis_key = _make_redis_key(target_func.__name__, key)
                        try:
                            result = await redis_client.delete(redis_key)
                            return cast(int, result) > 0
                        except Exception as e:
                            logger.warning(f"Redis cache delete failed: {e}")
                            return False
                else:
                    if key in cache_storage:
                        del cache_storage[key]
                        return True
                return False

            cache_delete = async_cache_delete
        else:

            def sync_cache_delete(*args, **kwargs) -> bool:
                """Delete a specific cache entry. Returns True if entry existed."""
                key = _make_hashable_key(args, kwargs)

                if shared_cache:
                    redis_client = _get_redis_client()
                    if redis_client:
                        redis_key = _make_redis_key(target_func.__name__, key)
                        try:
                            result = redis_client.delete(redis_key)
                            return cast(int, result) > 0
                        except Exception as e:
                            logger.warning(f"Redis cache delete failed: {e}")
                            return False
                else:
                    if key in cache_storage:
                        del cache_storage[key]
                        return True
                return False

            cache_delete = sync_cache_delete

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
