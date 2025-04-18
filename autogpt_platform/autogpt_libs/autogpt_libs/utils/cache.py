import inspect
import threading
from typing import Any, Awaitable, Callable, ParamSpec, TypeVar, cast, overload

P = ParamSpec("P")
R = TypeVar("R")


@overload
def thread_cached(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]: ...


@overload
def thread_cached(func: Callable[P, R]) -> Callable[P, R]: ...


def thread_cached(
    func: Callable[P, R] | Callable[P, Awaitable[R]],
) -> Callable[P, R] | Callable[P, Awaitable[R]]:
    thread_local = threading.local()

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

        return async_wrapper
    else:

        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            cache = getattr(thread_local, "cache", None)
            if cache is None:
                cache = thread_local.cache = {}
            # Include function in the key to prevent collisions between different functions
            key = (args, tuple(sorted(kwargs.items())))
            if key not in cache:
                cache[key] = func(*args, **kwargs)
            return cache[key]

        return sync_wrapper


def clear_thread_cache(func: Callable[..., Any]) -> None:
    """Clear the cache for a thread-cached function."""
    thread_local = threading.local()
    cache = getattr(thread_local, "cache", None)
    if cache is not None:
        # Clear all entries that match the function
        for key in list(cache.keys()):
            if key and len(key) > 0 and key[0] == func:
                del cache[key]
