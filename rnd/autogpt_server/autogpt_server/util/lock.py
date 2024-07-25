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
        self.locks: dict[Any, Lock] = ExpiringDict(max_len=6000, max_age_seconds=60)
        self.locks_lock = Lock()

    def lock(self, key: Any):
        with self.locks_lock:
            if key not in self.locks:
                self.locks[key] = (lock := Lock())
            else:
                lock = self.locks[key]
        lock.acquire()

    def unlock(self, key: Any):
        with self.locks_lock:
            if key in self.locks:
                lock = self.locks.pop(key)
            else:
                return
        lock.release()
