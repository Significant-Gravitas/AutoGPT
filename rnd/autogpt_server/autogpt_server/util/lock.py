from threading import Lock
from typing import Any

from expiringdict import ExpiringDict


class KeyedMutex:
    """
    This class provides a mutex that can be locked and unlocked by a specific key.
    It uses an ExpiringDict to automatically clear the mutex after a specified timeout,
    in case the key is not unlocked for a specified duration, to prevent memory leaks.
    """

    def __init__(self):
        self.locks: dict[Any, tuple[Lock, int]] = ExpiringDict(
            max_len=6000, max_age_seconds=60
        )
        self.locks_lock = Lock()

    def lock(self, key: Any):
        with self.locks_lock:
            lock, request_count = self.locks.get(key, (Lock(), 0))
            self.locks[key] = (lock, request_count + 1)
        lock.acquire()

    def unlock(self, key: Any):
        with self.locks_lock:
            lock, request_count = self.locks.pop(key)
            if request_count > 1:
                self.locks[key] = (lock, request_count - 1)
        lock.release()
