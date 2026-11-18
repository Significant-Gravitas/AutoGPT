"""Lightweight helpers for tests that must inspect local infrastructure, break
it on purpose, or prove that reads run side by side."""

import asyncio
import socket
from asyncio import open_connection
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from functools import cache
from typing import Self, TypeVar
from unittest.mock import patch

from backend.data import redis_client

T = TypeVar("T")


@cache
def is_tcp_port_reachable(
    host: str,
    port: int,
    *,
    timeout: float = 1.0,
) -> bool:
    """Return whether a TCP endpoint accepts a connection within ``timeout``.

    This deliberately avoids service clients with retry policies so it is safe
    to call from module-level pytest skip markers during collection. Results
    are cached so modules that share an endpoint only wait for one probe.
    """
    try:
        connection = socket.create_connection((host, port), timeout=timeout)
    except OSError:
        return False

    connection.close()
    return True


async def unreachable_tcp_endpoints(
    endpoints: dict[str, tuple[str, int]], *, within: float
) -> list[str]:
    """The ``endpoints`` that accepted no TCP connection within ``within``
    seconds, each named with its address.

    Each is tried again until then, so a service still starting gets its
    time, and one that is down is reported instead of leaving its clients
    retrying forever. Not cached, unlike :func:`is_tcp_port_reachable`.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + within
    waiting = dict(endpoints)
    while True:
        for name, (host, port) in list(waiting.items()):
            try:
                _, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port), timeout=1.0
                )
            except (OSError, TimeoutError):
                continue
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass
            del waiting[name]
        if not waiting or loop.time() >= deadline:
            break
        await asyncio.sleep(0.5)
    return [f"{name} at {host}:{port}" for name, (host, port) in waiting.items()]


class Rendezvous:
    """Fakes that answer only once all ``parties`` of them are being awaited.

    Proves calls run side by side without timing them: code that awaits the
    fakes one after another never gets past the first, which fails with
    ``TimeoutError`` instead of letting the test pass. ``timeout`` only bounds
    that failure; side-by-side callers meet at once.
    """

    def __init__(self, parties: int, *, timeout: float = 10.0) -> None:
        self._barrier = asyncio.Barrier(parties)
        self._timeout = timeout

    async def arrive(self) -> None:
        async with asyncio.timeout(self._timeout):
            await self._barrier.wait()

    def returning(self, value: T) -> Callable[..., Awaitable[T]]:
        """An async stand-in that arrives here, then answers ``value``."""

        async def fake(*_args: object, **_kwargs: object) -> T:
            await self.arrive()
            return value

        return fake


class TcpProxy:
    """Forwards each TCP connection to ``upstream`` until a test breaks it.

    ``drop()`` closes every connection it carries, as a restarting server
    does. ``black_hole`` keeps them open but forwards nothing either way, as a
    network partition does: no FIN, no RST, so only a reply that never comes
    gives it away.
    """

    def __init__(self, upstream: tuple[str, int]) -> None:
        self.upstream = upstream
        self.black_hole = False
        self.address: tuple[str, int] = ("127.0.0.1", 0)
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()
        # Accept handlers and pipes, so leaving the context stops them all.
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    async def __aenter__(self) -> Self:
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
        self.address = self._server.sockets[0].getsockname()[:2]
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self._closed = True
        if self._server is not None:
            self._server.close()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        try:
            # Bounded, so a task that outlives its cancel fails the test
            # instead of hanging it: the suite has no per-test timeout.
            async with asyncio.timeout(5):
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self.drop()
            if self._server is not None:
                async with asyncio.timeout(5):
                    await self._server.wait_closed()

    def drop(self) -> None:
        for writer in self._writers:
            writer.close()
        self._writers.clear()

    async def _accept(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # Registered first, so drop() closes it even if this is cancelled.
        self._writers.add(writer)
        if self._closed:
            writer.close()
            return
        self._track(asyncio.current_task())
        try:
            up_reader, up_writer = await open_connection(*self.upstream)
        except OSError:
            writer.close()
            return
        if self._closed:  # left the context while connecting
            writer.close()
            up_writer.close()
            return
        self._writers.add(up_writer)
        for source, sink in ((reader, up_writer), (up_reader, writer)):
            self._track(asyncio.create_task(self._pipe(source, sink)))

    def _track(self, task: asyncio.Task | None) -> None:
        if task is None:
            return
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _pipe(
        self, source: asyncio.StreamReader, sink: asyncio.StreamWriter
    ) -> None:
        try:
            while data := await source.read(65536):
                if self.black_hole:
                    continue
                sink.write(data)
                await sink.drain()
        except OSError:
            pass
        finally:
            sink.close()


@asynccontextmanager
async def redis_shard_proxies() -> AsyncIterator[Callable[[str], Awaitable[TcpProxy]]]:
    """Yield ``route(channel)``, which sends that channel's shard-pinned
    pub/sub connections through a new ``TcpProxy`` to the shard that really
    owns it, and returns the proxy to break. The cluster client and every
    other channel still go direct."""
    real_resolve = redis_client.resolve_shard_for_channel
    routes: dict[str, TcpProxy] = {}

    def resolve(channel: str) -> tuple[str, int]:
        proxy = routes.get(channel)
        return proxy.address if proxy else real_resolve(channel)

    async with AsyncExitStack() as stack:

        async def route(channel: str) -> TcpProxy:
            proxy = TcpProxy(real_resolve(channel))
            routes[channel] = await stack.enter_async_context(proxy)
            return proxy

        with patch.object(redis_client, "resolve_shard_for_channel", resolve):
            yield route
