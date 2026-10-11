"""Lightweight helpers for tests that must inspect local infrastructure."""

import asyncio
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from functools import cache
from typing import TYPE_CHECKING, Any, cast

from redis import Redis

import backend.data.redis_client as _redis_client
from backend.data.redis_client import PASSWORD

if TYPE_CHECKING:
    import pytest


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


class ShardProxy:
    """TCP proxy in front of one real shard, so a test can cut the network
    between the WS server and Redis without touching Redis itself."""

    def __init__(self, upstream: tuple[str, int]) -> None:
        self.upstream = upstream
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()
        self.address: tuple[str, int] = ("127.0.0.1", 0)

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle, "127.0.0.1", self.address[1]
        )
        self.address = self._server.sockets[0].getsockname()[:2]

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        up_reader, up_writer = await asyncio.open_connection(*self.upstream)
        self._writers.update((writer, up_writer))

        async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                while data := await src.read(65536):
                    dst.write(data)
                    await dst.drain()
            except (ConnectionError, OSError):
                pass
            finally:
                dst.close()

        await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))

    def drop_connections(self) -> None:
        """Reset every proxied connection; new ones are still accepted."""
        for w in self._writers:
            w.transport.abort()
        self._writers.clear()

    async def stop(self) -> None:
        """Refuse new connections and reset the open ones: the shard is gone."""
        if self._server is not None:
            self._server.close()
            self.drop_connections()
            await self._server.wait_closed()
            self._server = None
        self.drop_connections()


def shard_subscribers(upstream: tuple[str, int], channel: str) -> int:
    """SSUBSCRIBE count on the owning shard itself, bypassing the proxy."""
    node = Redis(host=upstream[0], port=upstream[1], password=PASSWORD)
    try:
        reply = cast(list[Any], node.execute_command("PUBSUB", "SHARDNUMSUB", channel))
        return int(reply[1])
    finally:
        node.close()


def spublish_on_shard(upstream: tuple[str, int], channel: str, payload: str) -> int:
    node = Redis(host=upstream[0], port=upstream[1], password=PASSWORD)
    try:
        return int(cast(int, node.execute_command("SPUBLISH", channel, payload)))
    finally:
        node.close()


@asynccontextmanager
async def shard_proxies_routed(
    monkeypatch: "pytest.MonkeyPatch",
) -> AsyncIterator[Callable[[str], Awaitable[ShardProxy]]]:
    """Route every shard-pinned pubsub connection through a ``ShardProxy``;
    yields ``proxy_for(channel)``, which returns the proxy for its shard."""
    _redis_client.get_redis.cache_clear()
    real_resolve = _redis_client.resolve_shard_for_channel
    proxies: dict[tuple[str, int], ShardProxy] = {}

    def _resolve(channel: str) -> tuple[str, int]:
        return proxies[real_resolve(channel)].address

    async def proxy_for(channel: str) -> ShardProxy:
        upstream = real_resolve(channel)
        if upstream not in proxies:
            proxies[upstream] = ShardProxy(upstream)
            await proxies[upstream].start()
        return proxies[upstream]

    monkeypatch.setattr(_redis_client, "resolve_shard_for_channel", _resolve)
    try:
        yield proxy_for
    finally:
        for proxy in proxies.values():
            await proxy.stop()
        _redis_client.disconnect()
