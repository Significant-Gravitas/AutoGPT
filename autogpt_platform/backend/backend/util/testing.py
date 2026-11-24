"""Lightweight helpers for tests that must inspect local infrastructure, or
prove that reads run side by side."""

import asyncio
import socket
from collections.abc import Awaitable, Callable
from functools import cache
from typing import TypeVar

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
