"""Lightweight helpers for tests that must inspect local infrastructure, break
it on purpose, or prove that reads run side by side."""

import asyncio
import socket
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
        self._pipes: set[asyncio.Task] = set()

    async def __aenter__(self) -> Self:
        self._server = await asyncio.start_server(self._accept, "127.0.0.1", 0)
        self.address = self._server.sockets[0].getsockname()[:2]
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._server is not None:
            self._server.close()
        self.drop()
        for pipe in list(self._pipes):
            pipe.cancel()
        await asyncio.gather(*self._pipes, return_exceptions=True)
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
        try:
            up_reader, up_writer = await asyncio.open_connection(*self.upstream)
        except OSError:
            writer.close()
            return
        self._writers |= {writer, up_writer}
        for source, sink in ((reader, up_writer), (up_reader, writer)):
            pipe = asyncio.create_task(self._pipe(source, sink))
            self._pipes.add(pipe)
            pipe.add_done_callback(self._pipes.discard)

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
