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

    def __init__(self, redis: "Redis", timeout: int | None = 60):
        self.redis = redis
        self.timeout = timeout
        self.locks: dict[Any, "RedisLock"] = ExpiringDict(
            max_len=6000, max_age_seconds=self.timeout
        )
        self.locks_lock = Lock()

    @contextmanager
    def locked(self, key: Any):
        lock = self.acquire(key)
        try:
            yield
        finally:
            lock.release()

    def acquire(self, key: Any) -> "RedisLock":
        """Acquires and returns a lock with the given key"""
        with self.locks_lock:
            if key not in self.locks:
                self.locks[key] = self.redis.lock(
                    str(key), self.timeout, thread_local=False
                )
            lock = self.locks[key]
        lock.acquire()
        return lock

    def release(self, key: Any):
        if lock := self.locks.get(key):
            lock.release()

    def release_all_locks(self):
        """Call this on process termination to ensure all locks are released"""
        self.locks_lock.acquire(blocking=False)
        for lock in self.locks.values():
            if lock.locked() and lock.owned():
                lock.release()
