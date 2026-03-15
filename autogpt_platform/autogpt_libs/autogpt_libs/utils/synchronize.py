import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from expiringdict import ExpiringDict

if TYPE_CHECKING:
    from redis.asyncio import Redis as AsyncRedis
    from redis.asyncio.lock import Lock as AsyncRedisLock


class AsyncRedisKeyedMutex:
    """
    This class provides a mutex that can be locked and unlocked by a specific key,
    using Redis as a distributed locking provider.
    It uses an ExpiringDict to automatically clear the mutex after a specified timeout,
    in case the key is not unlocked for a specified duration, to prevent memory leaks.
    """

    def __init__(self, redis: "AsyncRedis", timeout: int | None = 60):
        self.redis = redis
        self.timeout = timeout
        self.locks: dict[Any, "AsyncRedisLock"] = ExpiringDict(
            max_len=6000, max_age_seconds=self.timeout
        )
        self.locks_lock = asyncio.Lock()

    @asynccontextmanager
    async def locked(self, key: Any):
        lock = await self.acquire(key)
        try:
            yield
        finally:
            if (await lock.locked()) and (await lock.owned()):
                await lock.release()

    async def acquire(self, key: Any) -> "AsyncRedisLock":
        """Acquires and returns a lock with the given key"""
        async with self.locks_lock:
            if key not in self.locks:
                self.locks[key] = self.redis.lock(
                    str(key), self.timeout, thread_local=False
                )
            lock = self.locks[key]
        await lock.acquire()
        return lock

    async def release(self, key: Any):
        if (
            (lock := self.locks.get(key))
            and (await lock.locked())
            and (await lock.owned())
        ):
            await lock.release()

    async def release_all_locks(self):
        """Call this on process termination to ensure all locks are released"""
        async with self.locks_lock:
            for lock in self.locks.values():
                if (await lock.locked()) and (await lock.owned()):
                    await lock.release()
