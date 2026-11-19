import asyncio

import pytest
from prometheus_client import REGISTRY

from backend.api.ws_outbox import Delivery, WSBacklog, WSOutbox

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


class _Socket:
    """An outbox, its client, and why the outbox closed it, if it did."""

    def __init__(self, backlog: WSBacklog) -> None:
        self.client = _Client()
        self.closed = asyncio.Event()
        self.closes = 0
        self.why = ""

        async def on_overflow(why: str) -> None:
            self.why = why
            self.closes += 1
            self.closed.set()

        self.outbox = WSOutbox(on_overflow=on_overflow, backlog=backlog)

    async def stuck_behind(self, sizes: list[int]) -> None:
        """Queue events of ``sizes`` bytes, the first waiting on the client."""
        self.outbox.put(self.client.delivery("e0"), sizes[0])
        async with asyncio.timeout(_HANG_SECONDS):
            await self.client.waited_on.wait()
        for i, size in enumerate(sizes[1:], 1):
            self.outbox.put(self.client.delivery(f"e{i}"), size)

    async def close(self) -> None:
        await _stop(self.outbox, self.client)


async def _stop(outbox: WSOutbox, client: _Client) -> None:
    """Let a send left waiting end, then stop the outbox, bounded so a failed
    test can't hang its teardown."""
    client.reading.set()
    async with asyncio.timeout(_HANG_SECONDS):
        await outbox.stop()


async def _until(predicate, timeout: float = _HANG_SECONDS) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0)


def _queued_bytes_gauge() -> float:
    return REGISTRY.get_sample_value("autogpt_websocket_queued_bytes") or 0.0


@pytest.fixture
def small_bound(request, monkeypatch: pytest.MonkeyPatch) -> tuple[int, str]:
    """Bounds the queue at three events, by count or by bytes; returns the
    size to give each event and the bound's name."""
    if request.param == "events":
        monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_EVENTS", 3)
        return 1, "MAX_QUEUED_EVENTS"
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_BYTES", 30)
    return 10, "MAX_QUEUED_BYTES"


async def test_events_reach_the_client_in_order_one_at_a_time() -> None:
    socket = _Socket(WSBacklog())
    client = socket.client
    try:
        for i in range(50):
            socket.outbox.put(client.delivery(f"e{i}"), 10)
        async with asyncio.timeout(_HANG_SECONDS):
            await client.waited_on.wait()
        client.reading.set()
        await _until(lambda: len(client.got) == 50)
        assert client.got == [f"e{i}" for i in range(50)]
        assert client.most_at_once == 1
    finally:
        await socket.close()


async def test_a_delivery_that_raises_is_skipped_and_later_events_still_go_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def raises() -> None:
        raise RuntimeError("send failed")

    socket = _Socket(WSBacklog())
    client = socket.client
    client.reading.set()
    try:
        socket.outbox.put(raises, 10)
        socket.outbox.put(client.delivery("next"), 10)
        await _until(lambda: client.got == ["next"])
        socket.outbox.put(client.delivery("later"), 10)
        await _until(lambda: client.got == ["next", "later"])
        assert "Failed to send an event to a WebSocket" in caplog.text
    finally:
        await socket.close()


async def test_a_delivery_that_swallows_its_cancel_ends_with_nothing_counted() -> None:
    """A send can swallow the cancel that stops it, as redis-py's wait_for can
    on Python 3.11. The sender then wakes in a stopped outbox: it must just
    end, with nothing raised and nothing counted, below zero or above."""
    backlog = WSBacklog()
    waiting, swallowed, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def swallows_its_cancel() -> None:
        waiting.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            swallowed.set()

    socket = _Socket(backlog)
    gauge_before = _queued_bytes_gauge()
    try:
        socket.outbox.put(swallows_its_cancel, 10)
        async with asyncio.timeout(_HANG_SECONDS):
            await waiting.wait()
        sender = socket.outbox._sender
        assert sender is not None
        async with asyncio.timeout(_HANG_SECONDS):
            await socket.outbox.stop()
        assert swallowed.is_set()
        assert sender.done() and not sender.cancelled()
        assert sender.exception() is None
        assert socket.outbox.queued_bytes == 0
        assert backlog.queued_bytes == 0
        assert _queued_bytes_gauge() == gauge_before
    finally:
        # Lets the delivery end if the stop failed to, before closing.
        release.set()
        await socket.close()


@pytest.mark.parametrize("small_bound", ["events", "bytes"], indirect=True)
async def test_a_client_that_catches_up_within_the_bound_gets_everything(
    small_bound: tuple[int, str],
) -> None:
    size, _ = small_bound
    socket = _Socket(WSBacklog())
    client = socket.client
    try:
        await socket.stuck_behind([size] * 3)
        client.reading.set()
        await _until(lambda: len(client.got) == 3)
        # Caught up, so the bound starts over.
        for i in range(3, 6):
            socket.outbox.put(client.delivery(f"e{i}"), size)
        await _until(lambda: len(client.got) == 6)
        assert client.got == [f"e{i}" for i in range(6)]
        assert not socket.closed.is_set()
    finally:
        await socket.close()


@pytest.mark.parametrize("small_bound", ["events", "bytes"], indirect=True)
async def test_a_client_past_the_bound_is_closed_and_its_backlog_dropped(
    small_bound: tuple[int, str],
) -> None:
    """Its next events would be lost anyway, so it reconnects and refetches
    instead; what was queued for it is let go at once. The reason names the
    bound and how far behind it was, for tuning the bounds."""
    size, bound = small_bound
    socket = _Socket(WSBacklog())
    client = socket.client
    try:
        await socket.stuck_behind([size] * 4)
        async with asyncio.timeout(_HANG_SECONDS):
            await socket.closed.wait()
            # The send waiting on the client is abandoned, not left to finish.
            await client.cancelled.wait()
        assert socket.why == f"{bound}: 3 events, {3 * size} bytes waiting"
        assert len(socket.outbox._queue) == 0

        sender = socket.outbox._sender
        assert sender is not None
        async with asyncio.timeout(_HANG_SECONDS):
            await asyncio.wait([sender])
        assert sender.cancelled()
        for i in range(4, 100):
            socket.outbox.put(client.delivery(f"e{i}"), size)
        client.reading.set()
        # Only a sender sends, and no new one started for what came after.
        assert socket.outbox._sender is sender
        assert client.got == []
        assert client.sending == 0
        assert len(socket.outbox._queue) == 0
    finally:
        await socket.close()


@pytest.mark.parametrize(
    "sizes, closed",
    [
        ([10, 15, 5], False),
        ([10, 15, 6], True),
        ([10, 25], True),
        ([50], False),
        ([50, 1], True),
    ],
    ids=["exact-fit", "one-byte-over", "under-then-over", "big-alone", "behind-big"],
)
async def test_the_byte_bound_counts_the_event_being_put(
    sizes: list[int], closed: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The queue never holds more than the bound, unless one event alone is
    bigger: with nothing waiting the client isn't behind, so that one is
    still sent. In memory an event can take up to four times the bytes Redis
    sent, so closing on it would cut off messages Redis delivered before."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_BYTES", 30)
    socket = _Socket(WSBacklog())
    client = socket.client
    try:
        await socket.stuck_behind(sizes)
        if closed:
            async with asyncio.timeout(_HANG_SECONDS):
                await socket.closed.wait()
            assert socket.why.startswith("MAX_QUEUED_BYTES:")
            return
        client.reading.set()
        await _until(lambda: len(client.got) == len(sizes))
        assert not socket.closed.is_set()
    finally:
        await socket.close()


async def test_a_full_process_closes_the_client_furthest_behind_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sockets together can't hold more than the process allows: the client
    furthest behind is closed to make room, not whichever puts next."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_BYTES_PER_PROCESS", 100)
    backlog = WSBacklog()
    furthest, behind, next_up = (_Socket(backlog) for _ in range(3))
    try:
        await furthest.stuck_behind([30, 30])
        await behind.stuck_behind([30])
        assert backlog.queued_bytes == 90

        next_up.outbox.put(next_up.client.delivery("fits now"), 20)
        async with asyncio.timeout(_HANG_SECONDS):
            await furthest.closed.wait()
        assert furthest.why.startswith("MAX_QUEUED_BYTES_PER_PROCESS:")
        assert backlog.queued_bytes == 50
        next_up.client.reading.set()
        await _until(lambda: next_up.client.got == ["fits now"])
        assert not behind.closed.is_set()
        assert not next_up.closed.is_set()
    finally:
        for socket in (furthest, behind, next_up):
            await socket.close()


async def test_a_client_furthest_behind_that_puts_more_is_the_one_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_BYTES_PER_PROCESS", 100)
    backlog = WSBacklog()
    furthest, behind = _Socket(backlog), _Socket(backlog)
    try:
        await behind.stuck_behind([30])
        await furthest.stuck_behind([30, 30, 20])
        async with asyncio.timeout(_HANG_SECONDS):
            await furthest.closed.wait()
        assert backlog.queued_bytes == 30
        assert not behind.closed.is_set()
    finally:
        for socket in (furthest, behind):
            await socket.close()


async def test_room_is_made_by_closing_the_largest_backlogs_first_and_no_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A socket that keeps up gets its event: the clients furthest behind,
    most bytes first, are closed only until it fits. A smaller backlog and a
    socket with nothing waiting stay open, and no client is closed twice."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_BYTES_PER_PROCESS", 100)
    backlog = WSBacklog()
    small, large, medium, idle, keeping_up = (_Socket(backlog) for _ in range(5))
    sockets = (small, large, medium, idle, keeping_up)
    try:
        await small.stuck_behind([10])
        await large.stuck_behind([25, 25])
        await medium.stuck_behind([30])
        assert backlog.queued_bytes == 90

        # 90 + 75 is over: closing large (50) leaves 115, medium (30) 85.
        keeping_up.outbox.put(keeping_up.client.delivery("fits"), 75)
        async with asyncio.timeout(_HANG_SECONDS):
            await large.closed.wait()
            await medium.closed.wait()
        assert backlog.queued_bytes == 85
        assert [s.closes for s in sockets] == [0, 1, 1, 0, 0]
        keeping_up.client.reading.set()
        await _until(lambda: keeping_up.client.got == ["fits"])
    finally:
        for socket in sockets:
            await socket.close()


async def test_an_event_that_needs_every_backlog_gone_still_gets_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Making room ends after each client behind is closed once, and an event
    that fits the process at all always gets in: nothing loops or starves."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_BYTES_PER_PROCESS", 100)
    backlog = WSBacklog()
    behind = [_Socket(backlog) for _ in range(5)]
    keeping_up = _Socket(backlog)
    try:
        for socket in behind:
            await socket.stuck_behind([20])
        assert backlog.queued_bytes == 100

        keeping_up.outbox.put(keeping_up.client.delivery("fits exactly"), 100)
        async with asyncio.timeout(_HANG_SECONDS):
            for socket in behind:
                await socket.closed.wait()
        assert [s.closes for s in behind] == [1] * 5
        assert backlog.queued_bytes == 100
        keeping_up.client.reading.set()
        await _until(lambda: keeping_up.client.got == ["fits exactly"])
        assert not keeping_up.closed.is_set()
    finally:
        for socket in (*behind, keeping_up):
            await socket.close()


async def test_an_event_bigger_than_the_process_budget_closes_only_its_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closing others could never make it fit, so they keep their queues."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_BYTES_PER_PROCESS", 100)
    backlog = WSBacklog()
    behind, oversized = _Socket(backlog), _Socket(backlog)
    try:
        await behind.stuck_behind([30, 30])
        oversized.outbox.put(oversized.client.delivery("too big"), 101)
        async with asyncio.timeout(_HANG_SECONDS):
            await oversized.closed.wait()
        assert oversized.why.startswith(
            "MAX_QUEUED_BYTES_PER_PROCESS, one event of 101 bytes"
        )
        assert not behind.closed.is_set()
        assert backlog.queued_bytes == 60
        # Never queued, so nothing was ever there to send it.
        assert oversized.outbox._sender is None
        assert oversized.client.got == []
    finally:
        for socket in (behind, oversized):
            await socket.close()


@pytest.mark.parametrize("ending", ["sent", "closed", "stopped"])
async def test_a_backlog_counts_nothing_once_its_events_are_gone(
    ending: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Whether its events were sent, dropped as the client fell behind, or
    dropped as the socket closed, an outbox leaves nothing counted, in the
    process's room or on the queued-bytes gauge."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_EVENTS", 3)
    backlog = WSBacklog()
    socket = _Socket(backlog)
    gauge_before = _queued_bytes_gauge()
    try:
        await socket.stuck_behind([10, 10])
        assert backlog.queued_bytes == 20
        assert _queued_bytes_gauge() == gauge_before + 20
        if ending == "sent":
            socket.client.reading.set()
            await _until(lambda: len(socket.client.got) == 2)
        elif ending == "closed":
            socket.outbox.put(socket.client.delivery("e2"), 10)
            socket.outbox.put(socket.client.delivery("e3"), 10)
            async with asyncio.timeout(_HANG_SECONDS):
                await socket.closed.wait()
        else:
            await socket.outbox.stop()
        assert backlog.queued_bytes == 0
        assert not backlog._behind
        assert _queued_bytes_gauge() == gauge_before
    finally:
        await socket.close()


@pytest.mark.parametrize("stuck", ["sending", "closing"])
async def test_stop_ends_whatever_the_outbox_still_waits_on(
    stuck: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A socket closing can't leave a task waiting on its client, whether a
    send or the overflow's close of the socket."""
    monkeypatch.setattr("backend.api.ws_outbox.MAX_QUEUED_EVENTS", 1)
    closing, closer_ended = asyncio.Event(), asyncio.Event()

    async def on_overflow(why: str) -> None:
        closing.set()
        try:
            await asyncio.Event().wait()
        finally:
            closer_ended.set()

    client = _Client()
    outbox = WSOutbox(on_overflow=on_overflow, backlog=WSBacklog())
    try:
        outbox.put(client.delivery("e0"), 1)
        async with asyncio.timeout(_HANG_SECONDS):
            await client.waited_on.wait()
            if stuck == "closing":
                outbox.put(client.delivery("e1"), 1)
                await closing.wait()
            await outbox.stop()

        assert client.cancelled.is_set()
        assert closer_ended.is_set() == (stuck == "closing")
        sender = outbox._sender
        assert sender is not None and sender.done()
        outbox.put(client.delivery("late"), 1)
        client.reading.set()
        # Only a sender sends, and none started for the late event, which
        # wasn't kept either.
        assert outbox._sender is sender
        assert outbox.queued_bytes == 0
        assert client.got == []
    finally:
        await _stop(outbox, client)
