import asyncio
import socket
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from backend.util.testing import (
    Rendezvous,
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
