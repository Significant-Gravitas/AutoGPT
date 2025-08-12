import inspect
import logging
import threading
import time
from functools import wraps
from typing import (
    Awaitable,
    Callable,
    ParamSpec,
    Protocol,
    Tuple,
    TypeVar,
    cast,
    overload,
    runtime_checkable,
)

P = ParamSpec("P")
R = TypeVar("R")

logger = logging.getLogger(__name__)


@overload
def thread_cached(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    pass


@overload
def thread_cached(func: Callable[P, R]) -> Callable[P, R]:
    pass


def thread_cached(
    func: Callable[P, R] | Callable[P, Awaitable[R]],
) -> Callable[P, R] | Callable[P, Awaitable[R]]:
    thread_local = threading.local()

    def _clear():
        if hasattr(thread_local, "cache"):
            del thread_local.cache

    if inspect.iscoroutinefunction(func):

        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            cache = getattr(thread_local, "cache", None)
            if cache is None:
                cache = thread_local.cache = {}
            key = (args, tuple(sorted(kwargs.items())))
            if key not in cache:
                cache[key] = await cast(Callable[P, Awaitable[R]], func)(
                    *args, **kwargs
                )
            return cache[key]

        setattr(async_wrapper, "clear_cache", _clear)
        return async_wrapper

    else:

        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            cache = getattr(thread_local, "cache", None)
            if cache is None:
                cache = thread_local.cache = {}
            key = (args, tuple(sorted(kwargs.items())))
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]

        setattr(sync_wrapper, "clear_cache", _clear)
        return sync_wrapper


def clear_thread_cache(func: Callable) -> None:
    if clear := getattr(func, "clear_cache", None):
        clear()


FuncT = TypeVar("FuncT")


R_co = TypeVar("R_co", covariant=True)


@runtime_checkable
class AsyncCachedFunction(Protocol[P, R_co]):
    """Protocol for async functions with cache management methods."""

    def cache_clear(self) -> None:
        """Clear all cached entries."""
        return None

    def cache_info(self) -> dict[str, int | None]:
        """Get cache statistics."""
        return {}

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R_co:
        """Call the cached function."""
        return None  # type: ignore


def async_ttl_cache(
    maxsize: int = 128, ttl_seconds: int | None = None
) -> Callable[[Callable[P, Awaitable[R]]], AsyncCachedFunction[P, R]]:
    """
    TTL (Time To Live) cache decorator for async functions.

    Similar to functools.lru_cache but works with async functions and includes optional TTL.

    Args:
        maxsize: Maximum number of cached entries
        ttl_seconds: Time to live in seconds. If None, entries never expire (like lru_cache)

    Returns:
        Decorator function

    Example:
        # With TTL
        @async_ttl_cache(maxsize=1000, ttl_seconds=300)
        async def api_call(param: str) -> dict:
            return {"result": param}

        # Without TTL (permanent cache like lru_cache)
        @async_ttl_cache(maxsize=1000)
        async def expensive_computation(param: str) -> dict:
            return {"result": param}
    """

    def decorator(
        async_func: Callable[P, Awaitable[R]],
    ) -> AsyncCachedFunction[P, R]:
        # Cache storage - use union type to handle both cases
        cache_storage: dict[tuple, R | Tuple[R, float]] = {}

        @wraps(async_func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Create cache key from arguments
            key = (args, tuple(sorted(kwargs.items())))
            current_time = time.time()

            # Check if we have a valid cached entry
            if key in cache_storage:
                if ttl_seconds is None:
                    # No TTL - return cached result directly
                    logger.debug(
                        f"Cache hit for {async_func.__name__} with key: {str(key)[:50]}"
                    )
                    return cast(R, cache_storage[key])
                else:
                    # With TTL - check expiration
                    cached_data = cache_storage[key]
                    if isinstance(cached_data, tuple):
                        result, timestamp = cached_data
                        if current_time - timestamp < ttl_seconds:
                            logger.debug(
                                f"Cache hit for {async_func.__name__} with key: {str(key)[:50]}"
                            )
                            return cast(R, result)
                        else:
                            # Expired entry
                            del cache_storage[key]
                            logger.debug(
                                f"Cache entry expired for {async_func.__name__}"
                            )

            # Cache miss or expired - fetch fresh data
            logger.debug(
                f"Cache miss for {async_func.__name__} with key: {str(key)[:50]}"
            )
            result = await async_func(*args, **kwargs)

            # Store in cache
            if ttl_seconds is None:
                cache_storage[key] = result
            else:
                cache_storage[key] = (result, current_time)

            # Simple cleanup when cache gets too large
            if len(cache_storage) > maxsize:
                # Remove oldest entries (simple FIFO cleanup)
                cutoff = maxsize // 2
                oldest_keys = list(cache_storage.keys())[:-cutoff] if cutoff > 0 else []
                for old_key in oldest_keys:
                    cache_storage.pop(old_key, None)
                logger.debug(
                    f"Cache cleanup: removed {len(oldest_keys)} entries for {async_func.__name__}"
                )

            return result

        # Add cache management methods (similar to functools.lru_cache)
        def cache_clear() -> None:
            cache_storage.clear()

        def cache_info() -> dict[str, int | None]:
            return {
                "size": len(cache_storage),
                "maxsize": maxsize,
                "ttl_seconds": ttl_seconds,
            }

        # Attach methods to wrapper
        setattr(wrapper, "cache_clear", cache_clear)
        setattr(wrapper, "cache_info", cache_info)

        return cast(AsyncCachedFunction[P, R], wrapper)

    return decorator


@overload
def async_cache(
    func: Callable[P, Awaitable[R]],
) -> AsyncCachedFunction[P, R]:
    pass


@overload
def async_cache(
    func: None = None,
    *,
    maxsize: int = 128,
) -> Callable[[Callable[P, Awaitable[R]]], AsyncCachedFunction[P, R]]:
    pass


def async_cache(
    func: Callable[P, Awaitable[R]] | None = None,
    *,
    maxsize: int = 128,
) -> (
    AsyncCachedFunction[P, R]
    | Callable[[Callable[P, Awaitable[R]]], AsyncCachedFunction[P, R]]
):
    """
    Process-level cache decorator for async functions (no TTL).

    Similar to functools.lru_cache but works with async functions.
    This is a convenience wrapper around async_ttl_cache with ttl_seconds=None.

    Args:
        func: The async function to cache (when used without parentheses)
        maxsize: Maximum number of cached entries

    Returns:
        Decorated function or decorator

    Example:
        # Without parentheses (uses default maxsize=128)
        @async_cache
        async def get_data(param: str) -> dict:
            return {"result": param}

        # With parentheses and custom maxsize
        @async_cache(maxsize=1000)
        async def expensive_computation(param: str) -> dict:
            # Expensive computation here
            return {"result": param}
    """
    if func is None:
        # Called with parentheses @async_cache() or @async_cache(maxsize=...)
        return async_ttl_cache(maxsize=maxsize, ttl_seconds=None)
    else:
        # Called without parentheses @async_cache
        decorator = async_ttl_cache(maxsize=maxsize, ttl_seconds=None)
        return decorator(func)
