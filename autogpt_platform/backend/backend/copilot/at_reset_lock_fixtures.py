"""Redis as the at-reset tests see it, in memory: each chat's job lock, and
the cancels the scheduler shares between its replicas."""

import asyncio
from datetime import timedelta


class FakeJobLock:
    """redis-py's async ``Lock``: one holder per name, waited for up to its
    ``blocking_timeout``, its renewals counted."""

    def __init__(
        self, held: asyncio.Lock, blocking_timeout: float, renewals: list[float]
    ) -> None:
        self._held = held
        self._wait = blocking_timeout
        self._renewals = renewals

    async def acquire(self) -> bool:
        try:
            async with asyncio.timeout(self._wait):
                await self._held.acquire()
        except TimeoutError:
            return False
        return True

    async def reacquire(self) -> bool:
        self._renewals.append(asyncio.get_running_loop().time())
        return True

    async def release(self) -> None:
        self._held.release()


class FakeAtResetRedis:
    """What every replica sees in Redis, held in this process."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expiries: dict[str, timedelta] = {}
        self.lock_options: dict[str, dict[str, float]] = {}
        # When each lock's holder renewed it, by name.
        self.renewals: dict[str, list[float]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def lock(
        self, name: str, *, timeout: float, blocking_timeout: float
    ) -> FakeJobLock:
        self.lock_options[name] = {
            "timeout": timeout,
            "blocking_timeout": blocking_timeout,
        }
        return FakeJobLock(
            self._locks.setdefault(name, asyncio.Lock()),
            blocking_timeout,
            self.renewals.setdefault(name, []),
        )

    def is_locked(self, name: str) -> bool:
        lock = self._locks.get(name)
        return lock is not None and lock.locked()

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def set(self, key: str, value: str, *, ex: timedelta) -> bool:
        self.values[key] = value
        self.expiries[key] = ex
        return True
