import asyncio
import inspect
import socket
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import pytest
from prisma import Prisma
from prisma.engine.errors import EngineConnectionError

from backend.data import db as db_module
from backend.data.db import DISCONNECT_TIMEOUT, disconnect
from backend.util.retry import conn_retry


# The root conftest spins a full test server for every test via an autouse
# session fixture. None of these tests needs the app stack, so shadow it here the way `backend/util/conftest.py` does for its directory
# -- narrowed to this module so the rest of `backend/data` is unaffected.
@pytest.fixture(scope="session")
def server():
    yield None


async def test_disconnect_bounds_the_engine_shutdown(monkeypatch):
    """Prisma's shutdown wait is unbounded unless a timeout is handed to it.

    The blocking wait sits inside an async function, so an engine that ignores
    the SIGINT takes the event loop down with it.
    """
    prisma = MagicMock()
    prisma.is_connected.side_effect = [True, False]
    prisma.disconnect = AsyncMock()
    monkeypatch.setattr(db_module, "prisma", prisma)

    await disconnect()

    prisma.disconnect.assert_awaited_once_with(DISCONNECT_TIMEOUT)


async def test_disconnect_is_a_noop_when_not_connected(monkeypatch):
    prisma = MagicMock()
    prisma.is_connected.return_value = False
    prisma.disconnect = AsyncMock()
    monkeypatch.setattr(db_module, "prisma", prisma)

    await disconnect()

    prisma.disconnect.assert_not_awaited()


async def test_connect_retries_an_unreachable_database_then_raises(monkeypatch):
    client = _CountingPrisma(_url_at_port(_closed_port()), timedelta(seconds=1))
    monkeypatch.setattr(db_module, "prisma", client)

    with pytest.raises(EngineConnectionError):
        await _connect(max_retry=2)

    assert client.attempts == 3
    assert not db_module.is_connected()


async def test_connect_succeeds_once_the_database_comes_up(monkeypatch):
    """The case the retry exists for: Postgres refuses the first attempt and
    accepts the second."""
    port = _closed_port()
    proxies: list[asyncio.Server] = []

    async def start_database_on_second_attempt(attempt: int) -> None:
        if attempt == 2:
            proxies.append(await _forward_to_database(port))

    client = _CountingPrisma(
        _url_at_port(port), timedelta(seconds=5), start_database_on_second_attempt
    )
    monkeypatch.setattr(db_module, "prisma", client)
    try:
        await _connect(max_retry=2)

        assert await client.query_raw("SELECT 1 AS one") == [{"one": 1}]
        assert client.attempts == 2
    finally:
        await client.disconnect(DISCONNECT_TIMEOUT)
        for proxy in proxies:
            proxy.close()
            await proxy.wait_closed()


class _CountingPrisma(Prisma):
    def __init__(
        self,
        url: str,
        connect_timeout: timedelta,
        before_attempt: Callable[[int], Awaitable[None]] | None = None,
    ):
        super().__init__(datasource={"url": url}, connect_timeout=connect_timeout)
        self.attempts = 0
        self._before_attempt = before_attempt

    async def connect(self, *args: Any, **kwargs: Any) -> None:
        self.attempts += 1
        if self._before_attempt:
            await self._before_attempt(self.attempts)
        await super().connect(*args, **kwargs)


async def _connect(max_retry: int) -> None:
    """The real `connect()`, under a smaller retry budget than production's."""
    retried = conn_retry(
        "Prisma", "Acquiring connection", max_retry=max_retry, max_wait=0.1
    )
    await retried(inspect.unwrap(db_module.connect))()


def _closed_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _url_at_port(port: int) -> str:
    url = urlparse(db_module.DATABASE_URL)
    userinfo, _, _ = url.netloc.rpartition("@")
    host = f"127.0.0.1:{port}"
    return url._replace(netloc=f"{userinfo}@{host}" if userinfo else host).geturl()


async def _forward_to_database(port: int) -> asyncio.Server:
    url = urlparse(db_module.DATABASE_URL)

    async def forward(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        upstream_reader, upstream_writer = await asyncio.open_connection(
            url.hostname, url.port or 5432
        )
        await asyncio.gather(
            _pipe(reader, upstream_writer), _pipe(upstream_reader, writer)
        )

    return await asyncio.start_server(forward, "127.0.0.1", port)


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
    except ConnectionError:
        pass
    finally:
        writer.close()
