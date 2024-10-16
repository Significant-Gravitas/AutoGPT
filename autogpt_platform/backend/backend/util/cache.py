import threading
from functools import wraps
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def thread_cached_property(func: Callable[[T], R]) -> property:
    local_cache = threading.local()

    @wraps(func)
    def wrapper(self: T) -> R:
        if not hasattr(local_cache, "cache"):
            local_cache.cache = {}
        key = id(self)
        if key not in local_cache.cache:
            local_cache.cache[key] = func(self)
        return local_cache.cache[key]

    return property(wrapper)
