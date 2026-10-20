"""do-not-merge probe (#15739): watch each worker event loop for stalls.

Every loop a turn runs on gets a heartbeat task (every 50 ms). A watchdog
thread notices a heartbeat older than 300 ms, records the stall, and once it
passes a second logs the loop thread's Python stack, so a stall shows what
held the loop. ``stalls_since`` reports the calling thread's stalls since a
turn started, for a per-turn status.
"""

import asyncio
import contextvars
import logging
import sys
import threading
import time
import traceback

logger = logging.getLogger(__name__)

TURN_T0: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "perf_loopwatch_turn_t0", default=None
)

_BEAT_S = 0.05
_STALL_S = 0.3
_STACK_AFTER_S = 1.0


class _Watched:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self.beat = time.monotonic()
        self.stall_start: float | None = None
        self.stack_logged = False
        self.stalls: list[tuple[float, float]] = []


_watched: dict[int, _Watched] = {}
_lock = threading.Lock()
_watchdog: threading.Thread | None = None


def ensure_watch() -> None:
    """Start the heartbeat on the running loop and the watchdog, once each."""
    global _watchdog
    loop = asyncio.get_running_loop()
    tid = threading.get_ident()
    with _lock:
        current = _watched.get(tid)
        if current is not None and current.loop is loop:
            return
        _watched[tid] = _Watched(loop)
        if _watchdog is None:
            _watchdog = threading.Thread(
                target=_watch, name="perf-loopwatch", daemon=True
            )
            _watchdog.start()
    loop.create_task(_beat(tid, loop), name="perf-loopwatch-beat")


def stalls_since(t0: float | None) -> tuple[int, float, float]:
    """This thread's loop stalls that began at or after ``t0``: count, max ms,
    total ms (a stall still running counts up to now)."""
    entry = _watched.get(threading.get_ident())
    if entry is None or t0 is None:
        return 0, 0.0, 0.0
    durations = [d for start, d in list(entry.stalls) if start >= t0]
    if entry.stall_start is not None and entry.stall_start >= t0:
        durations.append(time.monotonic() - entry.stall_start)
    if not durations:
        return 0, 0.0, 0.0
    return len(durations), max(durations) * 1000, sum(durations) * 1000


async def _beat(tid: int, loop: asyncio.AbstractEventLoop) -> None:
    while True:
        entry = _watched.get(tid)
        if entry is None or entry.loop is not loop:
            return
        now = time.monotonic()
        if entry.stall_start is not None:
            duration = now - entry.stall_start
            entry.stalls.append((entry.stall_start, duration))
            del entry.stalls[:-500]
            if duration >= _STACK_AFTER_S:
                logger.warning(
                    "[perf-loopwatch] loop on thread %s stalled %.0f ms (ended)",
                    tid,
                    duration * 1000,
                )
            entry.stall_start = None
            entry.stack_logged = False
        entry.beat = now
        await asyncio.sleep(_BEAT_S)


def _watch() -> None:
    while True:
        time.sleep(_BEAT_S)
        now = time.monotonic()
        for tid, entry in list(_watched.items()):
            if now - entry.beat <= _STALL_S:
                continue
            if entry.stall_start is None:
                entry.stall_start = entry.beat
            if entry.stack_logged or now - entry.stall_start < _STACK_AFTER_S:
                continue
            entry.stack_logged = True
            frame = sys._current_frames().get(tid)
            stack = "".join(traceback.format_stack(frame)[-16:]) if frame else "?"
            logger.warning(
                "[perf-loopwatch] loop on thread %s stalled %.0f ms so far, at:\n%s",
                tid,
                (now - entry.stall_start) * 1000,
                stack,
            )
