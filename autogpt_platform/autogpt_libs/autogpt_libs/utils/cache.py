import inspect
import threading
from typing import Awaitable, Callable, ParamSpec, TypeVar, cast, overload

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
