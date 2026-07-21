"""Silence watchdog — keeps the FE informed during long copilot turns.

A copilot turn can stay silent for many seconds at a time:

* deep ``ThinkingBlock`` reasoning before any text/tool chunk lands,
* slow tool execution (e.g. ``browser_act`` navigation, video gen),
* inter-tool gaps where the model is computing the next call.

The FE's ``ThinkingIndicator`` cycles generic phrases and surfaces elapsed
time after 20s, but a backend-emitted ``StreamStatus`` overrides the
phrase rotation with a more specific message. Existing emissions only
fire at boundaries ("Contacting the model…", "Analyzing result…"); this
watchdog fills the silent gaps in between with escalating reassurance.

Usage:

    async def emit(status):
        await out_queue.put(status)

    async with SilenceWatchdog(emit_status=emit) as watchdog:
        async for event in driver_stream():
            if not isinstance(event, StreamStatus):
                watchdog.bump()
            else:
                watchdog.note_status_emitted()
            yield event

Each non-status yield resets the silence timer; the watchdog only fires
its escalation messages when the gap between bumps grows past a
threshold. ``note_status_emitted()`` lets driver-emitted status messages
suppress the watchdog for a short window so the two don't talk over each
other.
"""

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

from backend.copilot.response_model import StreamBaseResponse, StreamStatus

logger = logging.getLogger(__name__)


# Default escalation schedule: (silent_seconds, status_message).
# Thresholds are measured from the last ``bump()``.  Each tier fires
# once per silent gap; ``bump()`` re-arms the schedule.
DEFAULT_SCHEDULE: list[tuple[float, str]] = [
    (10.0, "Working on it…"),
    (30.0, "Still working — complex requests can take a moment…"),
    (60.0, "This is taking longer than usual…"),
]


# When the driver itself emits a ``StreamStatus`` (e.g. "Optimizing
# conversation context…"), the watchdog stays quiet for this many
# seconds so it doesn't immediately overwrite that message.
DEFAULT_SUPPRESSION_WINDOW_S = 5.0


# How often the internal loop wakes to check thresholds.  1s is fine
# for human-perceivable cadence and keeps wakeups cheap.
DEFAULT_TICK_S = 1.0


class SilenceWatchdog:
    """Async context manager that emits ``StreamStatus`` during silence.

    The watchdog is fully passive when there's regular activity — it only
    fires when ``bump()`` hasn't been called within a threshold.
    """

    def __init__(
        self,
        emit_status: Callable[[StreamStatus], Awaitable[None]],
        *,
        schedule: list[tuple[float, str]] | None = None,
        suppression_window_s: float = DEFAULT_SUPPRESSION_WINDOW_S,
        tick_s: float = DEFAULT_TICK_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._emit_status = emit_status
        self._schedule = sorted(schedule if schedule is not None else DEFAULT_SCHEDULE)
        self._suppression_window_s = suppression_window_s
        self._tick_s = tick_s
        self._clock = clock

        now = self._clock()
        self._last_event_at = now
        self._last_status_emitted_at: float = float("-inf")
        # Fired thresholds since the last ``bump()``.  Uses index into the
        # sorted schedule so we can quickly check whether a tier already
        # fired during the current silent gap.
        self._fired_tiers: set[int] = set()
        self._task: asyncio.Task[None] | None = None

    def bump(self) -> None:
        """Mark a non-status event as just emitted.

        Resets the silence timer and re-arms the escalation schedule.
        """
        self._last_event_at = self._clock()
        self._fired_tiers.clear()

    def note_status_emitted(self) -> None:
        """Mark that a non-watchdog ``StreamStatus`` was just emitted.

        Activates the suppression window so the watchdog doesn't fire
        immediately on top of the driver's own status message.
        """
        self._last_status_emitted_at = self._clock()

    async def __aenter__(self) -> "SilenceWatchdog":
        self._task = asyncio.create_task(self._loop(), name="silence-watchdog")
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _loop(self) -> None:
        """Tick loop. Fires due tiers; sleeps for ``tick_s`` between
        checks so tests can use sub-second thresholds without flakiness.
        """
        try:
            while True:
                await asyncio.sleep(self._tick_s)
                now = self._clock()
                # Suppression: if a non-watchdog status was emitted recently,
                # don't fire on top of it.
                if now - self._last_status_emitted_at < self._suppression_window_s:
                    continue
                silent_for = now - self._last_event_at
                for tier_index, (threshold, message) in enumerate(self._schedule):
                    if tier_index in self._fired_tiers:
                        continue
                    if silent_for < threshold:
                        break  # schedule is sorted — later tiers are still further out
                    await self._safe_emit(message)
                    self._fired_tiers.add(tier_index)
                    self._last_status_emitted_at = now
                    break  # one emission per tick; suppression window paces the next tier
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — defensive: keep loop alive on bugs in emit
            logger.exception("[SilenceWatchdog] unexpected error in tick loop")

    async def _safe_emit(self, message: str) -> None:
        """Emit a status, swallowing exceptions so a transient SSE
        failure doesn't kill the watchdog (the next tier still has a
        chance to fire if the channel recovers).
        """
        try:
            await self._emit_status(StreamStatus(message=message))
        except Exception:  # noqa: BLE001
            logger.warning(
                "[SilenceWatchdog] emit_status raised; continuing", exc_info=True
            )


async def wrap_stream_with_heartbeat(
    stream: AsyncGenerator[StreamBaseResponse, None],
    *,
    schedule: list[tuple[float, str]] | None = None,
    suppression_window_s: float = DEFAULT_SUPPRESSION_WINDOW_S,
    tick_s: float = DEFAULT_TICK_S,
) -> AsyncGenerator[StreamBaseResponse, None]:
    """Wrap a stream of ``StreamBaseResponse`` events with a silence
    watchdog so periodic ``StreamStatus`` events fire during long gaps.

    Driver-emitted ``StreamStatus`` events arm the suppression window so
    the watchdog doesn't talk over them. All other driver events bump the
    silence timer. Watchdog emissions are interleaved into the output
    stream via an internal queue so consumers see a single merged stream.
    """
    out_queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
    sentinel = object()

    async def _emit_from_watchdog(status: StreamStatus) -> None:
        await out_queue.put(("watchdog", status))

    async def _drain_driver() -> None:
        try:
            async for event in stream:
                await out_queue.put(("driver", event))
        finally:
            # Forward close to the underlying stream so its own cleanup
            # (e.g. SDK stream lock release) runs deterministically — the
            # ``async for`` above does not call ``aclose()`` on cancel /
            # GeneratorExit, leaving the lock held until GC. Mirrors the
            # explicit aclose pattern in ``executor/processor.py``.
            with contextlib.suppress(Exception):
                await stream.aclose()
            await out_queue.put(("driver", sentinel))

    drain_task = asyncio.create_task(_drain_driver(), name="stream-heartbeat-drain")
    try:
        async with SilenceWatchdog(
            emit_status=_emit_from_watchdog,
            schedule=schedule,
            suppression_window_s=suppression_window_s,
            tick_s=tick_s,
        ) as watchdog:
            while True:
                source, event = await out_queue.get()
                if event is sentinel:
                    break
                if source == "driver":
                    if isinstance(event, StreamStatus):
                        watchdog.note_status_emitted()
                    else:
                        watchdog.bump()
                yield event
    finally:
        drain_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drain_task
