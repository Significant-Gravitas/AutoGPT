"""Unit tests for the silence watchdog.

The watchdog is a small async helper that emits ``StreamStatus`` events
during long silent gaps in a copilot turn (model deep-thinking before any
chunk, inter-tool gaps, slow tool execution). Tests here use a tight
schedule (sub-second thresholds) so they run quickly without flakiness.
"""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator

import pytest
import pytest_asyncio

from backend.copilot.response_model import (
    StreamBaseResponse,
    StreamFinish,
    StreamStatus,
    StreamTextDelta,
)
from backend.copilot.stream_heartbeat import SilenceWatchdog, wrap_stream_with_heartbeat


# Override the heavy session-scope fixtures from ``backend/conftest.py`` —
# the watchdog is a pure async helper and doesn't need the real backend
# server / graph cleanup machinery. Mirrors the pattern in
# ``backend/copilot/sdk/conftest.py``.
@pytest_asyncio.fixture(scope="session", loop_scope="session", name="server")
async def _server_noop() -> None:
    return None


@pytest_asyncio.fixture(
    scope="session", loop_scope="session", autouse=True, name="graph_cleanup"
)
async def _graph_cleanup_noop() -> AsyncIterator[None]:
    yield


@pytest.mark.asyncio
async def test_emits_status_after_silent_threshold():
    """When no event is bumped, the watchdog fires the first scheduled
    status message after its threshold elapses."""
    received: list[StreamStatus] = []

    async def emit(status: StreamStatus) -> None:
        received.append(status)

    schedule = [(0.05, "Working on it…")]
    async with SilenceWatchdog(
        emit_status=emit,
        schedule=schedule,
        tick_s=0.01,
        suppression_window_s=0.0,
    ):
        await asyncio.sleep(0.15)

    assert any(s.message == "Working on it…" for s in received)


@pytest.mark.asyncio
async def test_bump_resets_silence_timer():
    """Periodic ``bump()`` calls keep the watchdog quiet — the threshold
    is measured from the last bump, not from watchdog start."""
    received: list[StreamStatus] = []

    async def emit(status: StreamStatus) -> None:
        received.append(status)

    schedule = [(0.10, "Working on it…")]
    async with SilenceWatchdog(
        emit_status=emit,
        schedule=schedule,
        tick_s=0.01,
        suppression_window_s=0.0,
    ) as watchdog:
        for _ in range(5):
            await asyncio.sleep(0.04)
            watchdog.bump()
        # 0.20s elapsed but each gap was < 0.10s.
        assert received == []


@pytest.mark.asyncio
async def test_escalates_through_schedule():
    """Each threshold fires its own message, in order."""
    received: list[StreamStatus] = []

    async def emit(status: StreamStatus) -> None:
        received.append(status)

    schedule = [
        (0.05, "Working on it…"),
        (0.10, "Still working…"),
        (0.20, "This is taking longer…"),
    ]
    async with SilenceWatchdog(
        emit_status=emit,
        schedule=schedule,
        tick_s=0.01,
        suppression_window_s=0.0,
    ):
        await asyncio.sleep(0.30)

    messages = [s.message for s in received]
    assert "Working on it…" in messages
    assert "Still working…" in messages
    assert "This is taking longer…" in messages
    # Order matches the schedule.
    assert messages.index("Working on it…") < messages.index("Still working…")
    assert messages.index("Still working…") < messages.index("This is taking longer…")


@pytest.mark.asyncio
async def test_each_threshold_fires_only_once_per_silent_gap():
    """A given threshold message fires once per silent gap. After
    ``bump()`` resets, the schedule re-arms."""
    received: list[StreamStatus] = []

    async def emit(status: StreamStatus) -> None:
        received.append(status)

    schedule = [(0.05, "Working on it…")]
    async with SilenceWatchdog(
        emit_status=emit,
        schedule=schedule,
        tick_s=0.01,
        suppression_window_s=0.0,
    ) as watchdog:
        await asyncio.sleep(0.15)
        first_count = sum(1 for s in received if s.message == "Working on it…")
        # Past first threshold, more ticks should not multiply emissions.
        await asyncio.sleep(0.10)
        second_count = sum(1 for s in received if s.message == "Working on it…")
        assert first_count == 1
        assert second_count == 1
        # After bump, schedule re-arms.
        watchdog.bump()
        await asyncio.sleep(0.10)
        third_count = sum(1 for s in received if s.message == "Working on it…")
        assert third_count == 2


@pytest.mark.asyncio
async def test_suppression_window_skips_when_recent_status():
    """If a non-watchdog ``StreamStatus`` was emitted recently, the
    watchdog stays quiet to avoid talking over the driver's own status
    messages (e.g. 'Optimizing conversation context…')."""
    received: list[StreamStatus] = []

    async def emit(status: StreamStatus) -> None:
        received.append(status)

    schedule = [(0.05, "Working on it…")]
    async with SilenceWatchdog(
        emit_status=emit,
        schedule=schedule,
        tick_s=0.01,
        suppression_window_s=0.20,
    ) as watchdog:
        watchdog.note_status_emitted()
        await asyncio.sleep(0.15)
        # Suppression window 0.20s > elapsed 0.15s → quiet.
        assert received == []


@pytest.mark.asyncio
async def test_cleanup_on_context_exit():
    """The internal task is cancelled and awaited when the context
    manager exits — no orphaned tasks."""
    received: list[StreamStatus] = []

    async def emit(status: StreamStatus) -> None:
        received.append(status)

    schedule = [(10.0, "Working on it…")]  # never fires in this test
    async with SilenceWatchdog(
        emit_status=emit,
        schedule=schedule,
        tick_s=0.01,
    ) as watchdog:
        task = watchdog._task
        assert task is not None and not task.done()

    assert task.done()


async def _delayed_stream(
    items: list[tuple[float, StreamBaseResponse]],
) -> AsyncGenerator[StreamBaseResponse, None]:
    """Helper: yield ``items`` with the given pre-yield sleeps."""
    for delay, item in items:
        if delay > 0:
            await asyncio.sleep(delay)
        yield item


@pytest.mark.asyncio
async def test_wrap_passes_through_driver_events_in_order():
    """The wrapper must not reorder or drop the driver's events."""
    events = [
        (0.0, StreamTextDelta(id="t1", delta="a")),
        (0.0, StreamTextDelta(id="t1", delta="b")),
        (0.0, StreamFinish()),
    ]
    received = []
    schedule = [(60.0, "Working on it…")]  # never fires
    async for event in wrap_stream_with_heartbeat(
        _delayed_stream(events),
        schedule=schedule,
        tick_s=0.05,
    ):
        received.append(event)

    assert len(received) == 3
    assert received[0].delta == "a"  # type: ignore[union-attr]
    assert received[1].delta == "b"  # type: ignore[union-attr]
    assert isinstance(received[2], StreamFinish)


@pytest.mark.asyncio
async def test_wrap_emits_status_during_silent_gap():
    """When the driver pauses, the watchdog interleaves status events."""
    events = [
        (0.0, StreamTextDelta(id="t1", delta="hello")),
        (0.20, StreamFinish()),
    ]
    schedule = [(0.05, "Working on it…")]
    received = []
    async for event in wrap_stream_with_heartbeat(
        _delayed_stream(events),
        schedule=schedule,
        tick_s=0.01,
        suppression_window_s=0.0,
    ):
        received.append(event)

    statuses = [e for e in received if isinstance(e, StreamStatus)]
    assert any(s.message == "Working on it…" for s in statuses)


@pytest.mark.asyncio
async def test_wrap_driver_status_suppresses_watchdog():
    """A driver-emitted ``StreamStatus`` should keep the watchdog quiet
    for ``suppression_window_s``. Otherwise both would talk at once."""
    events = [
        (0.0, StreamStatus(message="Optimizing conversation context…")),
        (0.10, StreamFinish()),
    ]
    schedule = [(0.05, "Working on it…")]
    received = []
    async for event in wrap_stream_with_heartbeat(
        _delayed_stream(events),
        schedule=schedule,
        tick_s=0.01,
        suppression_window_s=0.20,
    ):
        received.append(event)

    watchdog_msgs = [
        e
        for e in received
        if isinstance(e, StreamStatus) and e.message == "Working on it…"
    ]
    assert watchdog_msgs == []


@pytest.mark.asyncio
async def test_emit_status_failure_does_not_kill_watchdog():
    """If the emit callback raises (e.g. SSE channel went away), the
    watchdog logs and keeps running so subsequent bumps still work."""
    call_count = 0

    async def emit(status: StreamStatus) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("emit failed")

    schedule = [(0.05, "Working on it…")]
    async with SilenceWatchdog(
        emit_status=emit,
        schedule=schedule,
        tick_s=0.01,
        suppression_window_s=0.0,
    ) as watchdog:
        await asyncio.sleep(0.10)
        # First emit raised — but the watchdog must still be alive.
        assert watchdog._task is not None
        assert not watchdog._task.done()


@pytest.mark.asyncio
async def test_wrap_underlying_stream_closed_on_early_consumer_exit():
    """Early aclose() on the wrapper must propagate to the underlying stream
    so its cleanup (e.g. SDK lock release) runs deterministically."""
    closed = False

    async def _closeable_stream() -> AsyncGenerator[StreamBaseResponse, None]:
        nonlocal closed
        try:
            for _ in range(100):
                await asyncio.sleep(0.01)
                yield StreamTextDelta(id="t1", delta="x")
        finally:
            closed = True

    schedule = [(60.0, "Working on it…")]  # never fires in this test
    gen = wrap_stream_with_heartbeat(
        _closeable_stream(), schedule=schedule, tick_s=0.05
    )
    await gen.__anext__()  # consume one event
    await gen.aclose()  # exit early
    await asyncio.sleep(0.05)  # let cleanup propagate
    assert closed, "Underlying stream was not closed on early consumer exit"
