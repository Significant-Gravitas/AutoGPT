import asyncio
from typing import Optional, cast

from .tcp_helpers import tcp_nodelay


class BaseProtocol(asyncio.Protocol):
    __slots__ = (
        "_loop",
        "_paused",
        "_drain_waiter",
        "_connection_lost",
        "_reading_paused",
        "transport",
    )

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop: asyncio.AbstractEventLoop = loop
        self._paused = False
        self._drain_waiter: Optional[asyncio.Future[None]] = None
        self._reading_paused = False

        self.transport: Optional[asyncio.Transport] = None

    @property
    def connected(self) -> bool:
        """Return True if the connection is open."""
        return self.transport is not None

    def pause_writing(self) -> None:
        assert not self._paused
        self._paused = True

    def resume_writing(self) -> None:
        assert self._paused
        self._paused = False

        waiter = self._drain_waiter
        if waiter is not None:
            self._drain_waiter = None
            if not waiter.done():
                waiter.set_result(None)

    def pause_reading(self) -> None:
        if not self._reading_paused and self.transport is not None:
            try:
                self.transport.pause_reading()
            except (AttributeError, NotImplementedError, RuntimeError):
                pass
            self._reading_paused = True

    def resume_reading(self) -> None:
        if self._reading_paused and self.transport is not None:
            try:
                self.transport.resume_reading()
            except (AttributeError, NotImplementedError, RuntimeError):
                pass
            self._reading_paused = False

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        tr = cast(asyncio.Transport, transport)
        tcp_nodelay(tr, True)
        self.transport = tr

    def connection_lost(self, exc: Optional[BaseException]) -> None:
        # Wake up the writer if currently paused.
        self.transport = None
        if not self._paused:
            return
        waiter = self._drain_waiter
        if waiter is None:
            return
        self._drain_waiter = None
        if waiter.done():
            return
        if exc is None:
            waiter.set_result(None)
        else:
            waiter.set_exception(exc)

    async def _drain_helper(self) -> None:
        if not self.connected:
            raise ConnectionResetError("Connection lost")
        if not self._paused:
            return
        waiter = self._drain_waiter
        if waiter is None:
            waiter = self._loop.create_future()
            self._drain_waiter = waiter
        await asyncio.shield(waiter)
