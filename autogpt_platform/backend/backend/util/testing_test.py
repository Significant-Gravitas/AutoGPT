from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from backend.util.testing import is_tcp_port_reachable


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
