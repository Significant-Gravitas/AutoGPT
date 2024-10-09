from contextlib import contextmanager
from threading import Lock
from typing import TYPE_CHECKING, Any

from expiringdict import ExpiringDict

if TYPE_CHECKING:
    from redis import Redis
    from redis.lock import Lock as RedisLock


class RedisKeyedMutex:
    """
    This class provides a mutex that can be locked and unlocked by a specific key,
    using Redis as a distributed locking provider.
    It uses an ExpiringDict to automatically clear the mutex after a specified timeout,
    in case the key is not unlocked for a specified duration, to prevent memory leaks.
    """

    TIMEOUT = 60

    locks: dict[Any, tuple["RedisLock", int]] = ExpiringDict(
        max_len=6000, max_age_seconds=TIMEOUT
    )
    locks_lock = Lock()

    def __init__(self, redis: "Redis"):
        self.redis = redis

    @contextmanager
    def locked(self, key: Any):
        self.lock(key)
        try:
            yield
        finally:
            self.unlock(key)

    def lock(self, key: Any):
        with self.locks_lock:
            if key not in self.locks:
                self.locks[key] = (
                    self.redis.lock(str(key), self.TIMEOUT, thread_local=False),
                    0,
                )
            lock, request_count = self.locks[key]
            self.locks[key] = (lock, request_count + 1)
        lock.acquire()

    def unlock(self, key: Any):
        with self.locks_lock:
            lock, request_count = self.locks.pop(key)
            if request_count > 1:
                self.locks[key] = (lock, request_count - 1)
        lock.release()
