"""Holding a shared-cache read in Redis, as a shard that stopped answering
would, for tests of what the rest of the process does meanwhile."""

import asyncio
import threading
from typing import Any

import pytest
import redis.asyncio.cluster as async_cluster
import redis.cluster as sync_cluster

from backend.util.cache import _make_hashable_key, _make_redis_key


def cache_key(function_name: str, **arguments: Any) -> str:
    """The Redis key of a ``@cached(shared_cache=True)`` call, by argument
    name, as the decorator binds them."""
    return _make_redis_key(_make_hashable_key((), arguments), function_name)


class HeldRead:
    """The first read of one key, held until ``release()``, whichever client
    makes it: a sync read blocks its thread and an async one awaits. A sync
    read also goes on after ``hang_s``, so code that reads on the event
    loop's thread, which the test can't release from, fails it instead of
    hanging it.

    ``answered`` turns true once the held read goes on to Redis; a read its
    caller gave up on never does. ``commands`` lists every command sent for
    the key since, held or not."""

    def __init__(self, redis_key: str, hang_s: float) -> None:
        self.answered = False
        self.commands: list[str] = []
        self._redis_key = redis_key
        self._hang_s = hang_s
        self._claimed = False
        self._loop = asyncio.get_running_loop()
        self._started = asyncio.Event()
        self._released = threading.Event()
        self._released_here = asyncio.Event()

    async def started(self) -> None:
        await self._started.wait()

    def release(self) -> None:
        self._released.set()
        self._released_here.set()

    def claims(self, command: Any, key: Any = None, *_: Any) -> bool:
        if (key.decode() if isinstance(key, bytes) else key) != self._redis_key:
            return False
        self.commands.append(str(command))
        if self._claimed or command not in ("GET", "GETEX"):
            return False
        self._claimed = True
        return True

    def hold_thread(self) -> None:
        self._loop.call_soon_threadsafe(self._started.set)
        self._released.wait(self._hang_s)
        self.answered = True

    async def hold_coroutine(self) -> None:
        self._started.set()
        await self._released_here.wait()
        self.answered = True


def hold_cache_read(
    monkeypatch: pytest.MonkeyPatch, redis_key: str, hang_s: float
) -> HeldRead:
    """Hold the next read of ``redis_key`` until the returned handle is
    released. Release it in a ``finally``."""
    held = HeldRead(redis_key, hang_s)
    execute = sync_cluster.RedisCluster.execute_command
    execute_async = async_cluster.RedisCluster.execute_command

    def held_execute(self, *args, **kwargs):
        if held.claims(*args):
            held.hold_thread()
        return execute(self, *args, **kwargs)

    async def held_execute_async(self, *args, **kwargs):
        if held.claims(*args):
            await held.hold_coroutine()
        return await execute_async(self, *args, **kwargs)

    monkeypatch.setattr(sync_cluster.RedisCluster, "execute_command", held_execute)
    monkeypatch.setattr(
        async_cluster.RedisCluster, "execute_command", held_execute_async
    )
    return held
