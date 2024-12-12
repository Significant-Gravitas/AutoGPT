import threading
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


def thread_cached(func: Callable[P, R]) -> Callable[P, R]:
    thread_local = threading.local()

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        cache = getattr(thread_local, "cache", None)
        if cache is None:
            cache = thread_local.cache = {}
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return wrapper
