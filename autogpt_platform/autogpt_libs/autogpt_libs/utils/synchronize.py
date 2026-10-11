import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Union

from expiringdict import ExpiringDict

if TYPE_CHECKING:
    from redis.asyncio import Redis as AsyncRedis
    from redis.asyncio.cluster import RedisCluster as AsyncRedisCluster
    from redis.asyncio.lock import Lock as AsyncRedisLock

    AsyncRedisLike = Union[AsyncRedis, AsyncRedisCluster]


class AsyncRedisKeyedMutex:
    """
    This class provides a mutex that can be locked and unlocked by a specific key,
    using Redis as a distributed locking provider.

    Every acquisition gets its own redis-py ``Lock``. A ``Lock`` keeps its owner
    token on the object, and since redis-py 7 ``release()`` clears that token
    only after the Redis round-trip. One ``Lock`` shared by every coroutine
    waiting on a key let a finishing holder wipe the token of the waiter that
    had just taken the key. That waiter then skipped its own release, leaving
    the key held until its TTL ran out, and stalling every other request for
    that key for up to ``timeout`` seconds.

    It tracks the latest lock per key in an ExpiringDict, so that ``release``
    and ``release_all_locks`` can find it, and so the dict can't leak memory.
    """

    def __init__(self, redis: "AsyncRedisLike", timeout: int | None = 60):
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
        lock = self.redis.lock(str(key), self.timeout, thread_local=False)
        await lock.acquire()
        async with self.locks_lock:
            self.locks[key] = lock
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
            for lock in list(self.locks.values()):
                if (await lock.locked()) and (await lock.owned()):
                    await lock.release()
