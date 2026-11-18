import asyncio
from unittest.mock import AsyncMock

import pytest

from backend.api.ws_outbox import Delivery, WSOutbox

# Only a step that never comes keeps a wait going this long.
_HANG_SECONDS = 10.0


class _Client:
    """The socket's client as its deliveries see it: each waits while the
    client isn't reading, the way a send waits on a paused transport."""

    def __init__(self) -> None:
        self.reading = asyncio.Event()
        self.waited_on = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.got: list[str] = []
        self.sending = 0
        self.most_at_once = 0

    def delivery(self, event: str) -> Delivery:
        async def deliver() -> None:
            self.sending += 1
            self.most_at_once = max(self.most_at_once, self.sending)
            try:
                if not self.reading.is_set():
                    self.waited_on.set()
                await self.reading.wait()
                self.got.append(event)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            finally:
                self.sending -= 1

        return deliver


async def _until(predicate, timeout: float = _HANG_SECONDS) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


@pytest.fixture
def small_bounds(request, monkeypatch: pytest.MonkeyPatch) -> int:
    """Bounds the queue at three events, by count or by bytes; returns the
    size to give each event."""
    if request.param == "events":
        monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_EVENTS", 3)
        return 1
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_BYTES", 30)
    return 10


async def test_events_reach_the_client_in_order_one_at_a_time() -> None:
    client = _Client()
    outbox = WSOutbox(on_overflow=AsyncMock())
    try:
        for i in range(50):
            outbox.put(client.delivery(f"e{i}"), 10)
        async with asyncio.timeout(_HANG_SECONDS):
            await client.waited_on.wait()
        client.reading.set()
        await _until(lambda: len(client.got) == 50)
        assert client.got == [f"e{i}" for i in range(50)]
        assert client.most_at_once == 1
    finally:
        await outbox.stop()


@pytest.mark.parametrize("small_bounds", ["events", "bytes"], indirect=True)
async def test_a_client_that_catches_up_within_the_bound_gets_everything(
    small_bounds: int,
) -> None:
    on_overflow = AsyncMock()
    client = _Client()
    outbox = WSOutbox(on_overflow=on_overflow)
    try:
        for i in range(3):
            outbox.put(client.delivery(f"e{i}"), small_bounds)
        async with asyncio.timeout(_HANG_SECONDS):
            await client.waited_on.wait()
        client.reading.set()
        await _until(lambda: len(client.got) == 3)
        # Caught up, so the bound starts over.
        for i in range(3, 6):
            outbox.put(client.delivery(f"e{i}"), small_bounds)
        await _until(lambda: len(client.got) == 6)
        assert client.got == [f"e{i}" for i in range(6)]
        on_overflow.assert_not_awaited()
    finally:
        await outbox.stop()


@pytest.mark.parametrize("small_bounds", ["events", "bytes"], indirect=True)
async def test_a_client_past_the_bound_is_closed_and_its_backlog_dropped(
    small_bounds: int,
) -> None:
    """Its next events would be lost anyway, so it reconnects and refetches
    instead; what was queued for it is let go at once."""
    overflowed = asyncio.Event()

    async def on_overflow() -> None:
        overflowed.set()

    client = _Client()
    outbox = WSOutbox(on_overflow=on_overflow)
    try:
        outbox.put(client.delivery("e0"), small_bounds)
        async with asyncio.timeout(_HANG_SECONDS):
            await client.waited_on.wait()
        for i in range(1, 4):
            outbox.put(client.delivery(f"e{i}"), small_bounds)
        async with asyncio.timeout(_HANG_SECONDS):
            await overflowed.wait()
            # The send waiting on the client is abandoned, not left to finish.
            await client.cancelled.wait()
        assert len(outbox._queue) == 0

        for i in range(4, 100):
            outbox.put(client.delivery(f"e{i}"), small_bounds)
        client.reading.set()
        await asyncio.sleep(0)
        assert client.got == []
        assert client.sending == 0
        assert len(outbox._queue) == 0
    finally:
        # Lets a send left waiting end, so a failure here can't hang stop().
        client.reading.set()
        await outbox.stop()


@pytest.mark.parametrize("stuck", ["sending", "closing"])
async def test_stop_ends_whatever_the_outbox_still_waits_on(
    stuck: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A socket closing can't leave a task waiting on its client, whether a
    send or the overflow's close of the socket."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_EVENTS", 1)
    closing, closer_ended = asyncio.Event(), asyncio.Event()

    async def on_overflow() -> None:
        closing.set()
        try:
            await asyncio.Event().wait()
        finally:
            closer_ended.set()

    client = _Client()
    outbox = WSOutbox(on_overflow=on_overflow)
    outbox.put(client.delivery("e0"), 1)
    async with asyncio.timeout(_HANG_SECONDS):
        await client.waited_on.wait()
        if stuck == "closing":
            outbox.put(client.delivery("e1"), 1)
            await closing.wait()
        await outbox.stop()

    assert client.cancelled.is_set()
    assert closer_ended.is_set() == (stuck == "closing")
    outbox.put(client.delivery("late"), 1)
    client.reading.set()
    await asyncio.sleep(0)
    assert client.got == []
