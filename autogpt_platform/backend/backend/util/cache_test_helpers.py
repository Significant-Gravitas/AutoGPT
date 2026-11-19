"""Helpers for tests of the shared cache: an entry's Redis key, a read held in
Redis as a shard that stopped answering would hold it, and the callers that
have reached a key's computation."""

import asyncio
import threading
from typing import Any

import pytest
import redis.asyncio.cluster as async_cluster
import redis.cluster as sync_cluster

from backend.util import cache as cache_module
from backend.util.cache import _make_hashable_key, _make_redis_key


def cache_key(function_name: str, **arguments: Any) -> str:
    """The Redis key of a ``@cached(shared_cache=True)`` call, by argument
    name, as the decorator binds them."""
    return _make_redis_key(_make_hashable_key((), arguments), function_name)


class HeldRead:
    """The first read of one key (or the first of ``commands``), held until
    ``release()``, whichever client makes it: a sync read blocks its thread
    and an async one awaits. A sync read also goes on after ``hang_s``, so
    code that reads on the event loop's thread, which the test can't release
    from, fails it instead of hanging it.

    ``answered`` turns true once the held read goes on to Redis; a read its
    caller gave up on never does. ``commands`` lists every command sent for
    the key since, held or not."""

    def __init__(
        self, redis_key: str, hang_s: float, held_commands: tuple[str, ...]
    ) -> None:
        self.answered = False
        self.commands: list[str] = []
        self._redis_key = redis_key
        self._hang_s = hang_s
        self._held_commands = held_commands
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
        if self._claimed or command not in self._held_commands:
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
    monkeypatch: pytest.MonkeyPatch,
    redis_key: str,
    hang_s: float,
    commands: tuple[str, ...] = ("GET", "GETEX"),
) -> HeldRead:
    """Hold the next read of ``redis_key`` (or the next of ``commands``)
    until the returned handle is released. Release it in a ``finally``."""
    held = HeldRead(redis_key, hang_s, commands)
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


def callers_in_flight(monkeypatch: pytest.MonkeyPatch, expected: int) -> asyncio.Event:
    """Set once ``expected`` callers of ``@cached`` coroutines have missed and
    reached their key's computation: computing it, waiting for a turn to,
    or waiting for another caller's."""
    entered = 0
    everyone_in = asyncio.Event()
    run = cache_module._InFlight.run

    async def counted_run(self, *args, **kwargs):
        nonlocal entered
        entered += 1
        if entered == expected:
            everyone_in.set()
        return await run(self, *args, **kwargs)

    monkeypatch.setattr(cache_module._InFlight, "run", counted_run)
    return everyone_in


def forget_connections(client: Any) -> None:
    """Close what's left of an async cluster client whose loop has closed, on
    a loop of its own. Each connection's transport belongs to the closed
    loop, so closing one raises, and redis-py forgets it all the same."""
    loop = asyncio.new_event_loop()
    try:
        for close in (
            client.aclose,
            client.nodes_manager.aclose,
            lambda: client.nodes_manager.aclose("startup_nodes"),
        ):
            try:
                loop.run_until_complete(close())
            except Exception:
                pass
    finally:
        loop.close()
