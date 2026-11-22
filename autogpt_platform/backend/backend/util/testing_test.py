import asyncio
import socket
from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util import testing
from backend.util.testing import (
    Rendezvous,
    TcpProxy,
    is_tcp_port_reachable,
    unreachable_tcp_endpoints,
)


@pytest.fixture(autouse=True)
def _clear_port_probe_cache() -> Iterator[None]:
    is_tcp_port_reachable.cache_clear()
    yield
    is_tcp_port_reachable.cache_clear()


def test_is_tcp_port_reachable_closes_probe_socket() -> None:
    connection = MagicMock()

    with patch(
        "backend.util.testing.socket.create_connection",
        return_value=connection,
    ) as create_connection:
        assert is_tcp_port_reachable("redis.example", 6379)

    create_connection.assert_called_once_with(("redis.example", 6379), timeout=1.0)
    connection.close.assert_called_once_with()


def test_is_tcp_port_reachable_caches_repeated_probe() -> None:
    connection = MagicMock()

    with patch(
        "backend.util.testing.socket.create_connection",
        return_value=connection,
    ) as create_connection:
        assert is_tcp_port_reachable("redis.example", 6379)
        assert is_tcp_port_reachable("redis.example", 6379)

    create_connection.assert_called_once_with(("redis.example", 6379), timeout=1.0)


def test_is_tcp_port_reachable_returns_false_for_os_error() -> None:
    with patch(
        "backend.util.testing.socket.create_connection",
        side_effect=TimeoutError,
    ):
        assert not is_tcp_port_reachable("redis.example", 6379)


def test_is_tcp_port_reachable_propagates_unexpected_errors() -> None:
    with patch(
        "backend.util.testing.socket.create_connection",
        side_effect=ValueError("invalid endpoint"),
    ):
        with pytest.raises(ValueError, match="invalid endpoint"):
            is_tcp_port_reachable("redis.example", 6379)


@pytest.mark.asyncio
async def test_unreachable_tcp_endpoints_names_only_the_one_that_is_down() -> None:
    """The test server's preflight: a listening service passes, and a closed
    port is named once the wait runs out, not retried forever. Each is
    probed on its own, so a slow runner can only slow the listening probe
    within its long window, never turn it into a miss."""
    listening = await asyncio.start_server(lambda _r, w: w.close(), "127.0.0.1", 0)
    up = listening.sockets[0].getsockname()[1]
    try:
        reached = await unreachable_tcp_endpoints(
            {"Postgres": ("127.0.0.1", up)}, within=30
        )
    finally:
        listening.close()
        await listening.wait_closed()
    # Bound but never listening: it refuses, and nothing else can take it.
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        down = reserved.getsockname()[1]
        missing = await unreachable_tcp_endpoints(
            {"Redis": ("127.0.0.1", down)}, within=0.2
        )

    assert reached == []
    assert missing == [f"Redis at 127.0.0.1:{down}"]


@pytest.mark.asyncio
async def test_rendezvous_answers_callers_awaited_side_by_side() -> None:
    meeting = Rendezvous(2)

    assert await asyncio.gather(
        meeting.returning("a")(), meeting.returning("b")(key="ignored")
    ) == ["a", "b"]


@pytest.mark.asyncio
async def test_rendezvous_fails_callers_awaited_one_after_another() -> None:
    meeting = Rendezvous(2, timeout=0.05)

    with pytest.raises(TimeoutError):
        await meeting.returning("a")()


async def test_leaving_a_proxy_stops_a_connection_it_was_still_setting_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connection still waiting on its upstream when the test leaves the
    proxy must not outlive it, or it starts pipes and sockets nobody closes."""
    real_open_connection = asyncio.open_connection
    upstream = ("127.0.0.1", 9)
    connecting = asyncio.Event()
    dial_cancelled = asyncio.Event()

    async def open_connection(host: str, port: int, *args, **kwargs):
        if (host, port) == upstream:
            connecting.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                dial_cancelled.set()
                raise
        return await real_open_connection(host, port, *args, **kwargs)

    monkeypatch.setattr(testing, "open_connection", open_connection)
    async with TcpProxy(upstream) as proxy:
        reader, writer = await real_open_connection(*proxy.address)
        async with asyncio.timeout(5):
            await connecting.wait()
    try:
        # The handler was stopped, not left dialling, and its client closed.
        assert dial_cancelled.is_set()
        async with asyncio.timeout(5):
            assert await reader.read() == b""
    finally:
        writer.close()


async def test_a_connection_accepted_after_the_proxy_closed_is_turned_away(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dial = AsyncMock(return_value=(MagicMock(), MagicMock()))
    monkeypatch.setattr(testing, "open_connection", dial)
    async with TcpProxy(("127.0.0.1", 9)) as proxy:
        pass
    client = MagicMock()

    await proxy._accept(MagicMock(), client)

    client.close.assert_called_once()
    dial.assert_not_awaited()


async def test_an_upstream_that_answers_after_the_proxy_closed_is_hung_up_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy = TcpProxy(("127.0.0.1", 9))
    client, upstream_writer = MagicMock(), MagicMock()

    async def dial(*_args: object, **_kwargs: object) -> tuple:
        proxy._closed = True  # the test left the proxy while this dial ran
        return MagicMock(), upstream_writer

    monkeypatch.setattr(testing, "open_connection", dial)
    await proxy._accept(MagicMock(), client)

    client.close.assert_called_once()
    upstream_writer.close.assert_called_once()
    assert not [task for task in proxy._tasks if task is not asyncio.current_task()]
