"""Redis as the at-reset tests see it, in memory: each chat's job lock, and
the cancels the scheduler shares between its replicas."""

import asyncio
from collections.abc import Callable
from datetime import timedelta
from types import TracebackType


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


class FakeTransaction:
    """A ``pipeline(transaction=True)``: its commands run together on
    ``execute``."""

    def __init__(self, redis: "FakeAtResetRedis") -> None:
        self._redis = redis
        self._queued: list[Callable[[], object]] = []

    async def __aenter__(self) -> "FakeTransaction":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._queued.clear()

    def zadd(self, key: str, mapping: dict[str, int], *, gt: bool) -> None:
        self._queued.append(lambda: self._redis.add_scores(key, mapping, gt=gt))

    def expire(self, key: str, ttl: timedelta) -> None:
        self._queued.append(lambda: self._redis.expiries.update({key: ttl}))

    async def execute(self) -> list[object]:
        results = [run() for run in self._queued]
        self._queued.clear()
        return results


class FakeAtResetRedis:
    """What every replica sees in Redis, held in this process."""

    def __init__(self) -> None:
        # Each sorted set's members and scores, by key.
        self.scores: dict[str, dict[str, float]] = {}
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

    def pipeline(self, *, transaction: bool) -> FakeTransaction:
        assert transaction
        return FakeTransaction(self)

    def add_scores(self, key: str, mapping: dict[str, int], *, gt: bool) -> int:
        """``ZADD``, with ``GT`` only ever moving a member's score up."""
        members = self.scores.setdefault(key, {})
        added = 0
        for member, score in mapping.items():
            if member not in members:
                added += 1
            elif gt and score <= members[member]:
                continue
            members[member] = score
        return added

    async def zscore(self, key: str, member: str) -> float | None:
        return self.scores.get(key, {}).get(member)
