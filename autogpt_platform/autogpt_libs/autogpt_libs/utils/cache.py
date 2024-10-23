import threading
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")


def thread_cached(func: Callable[P, R]) -> Callable[P, R]:
    thread_local = threading.local()

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        cache = getattr(thread_local, "cache", None)
        if cache is None:
            cache = thread_local.cache = {}
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper


def thread_cached_property(func: Callable[[T], R]) -> property:
    return property(thread_cached(func))
