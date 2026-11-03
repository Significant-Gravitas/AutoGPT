"""macOS input-capture producer — the real per-OS `CaptureSource` for darwin.

This is the concrete implementation of the seam documented in `capture.py`:
a `CaptureSource` whose `steps()` yields one `TrajectoryStep` per observed
user action, sourced from a Quartz `CGEventTap`. Pass an instance as the
`input_events` of a `ScreenshotActionFloor`; no other layer changes.

Permissions (TCC): a session-level `CGEventTap` requires the host process to
hold **Input Monitoring** (and, to read window *titles*, **Screen Recording**)
in System Settings → Privacy & Security. If the grant is missing,
`CGEventTapCreate` returns `None` — `start()` surfaces that as
`tap_created == False` so the caller can route to the consent flow rather than
silently capturing nothing. This is exactly the gate WORKFLOW_RECORDING.md §9
relies on.

Listen-only: the tap is created with `kCGEventTapOptionListenOnly`, so it
observes without consuming or modifying the user's input.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import AsyncIterator

from ..protocol import TrajectoryStep
from .capture import CaptureSource

logger = logging.getLogger(__name__)

# Quartz keycode → coarse action mapping. A real field-level producer would
# buffer keystrokes into a single `fill` per focused field; for the reference
# producer we map at the event grain (mouse-down → click, key-down → fill).
_MOUSE_DOWN_TYPES: set[int] = set()
_KEY_DOWN_TYPE: int | None = None


class MacInputCaptureSource(CaptureSource):
    """Observes real mouse-down / key-down events via a Quartz CGEventTap.

    Runs the tap on a dedicated thread with its own CFRunLoop; the callback
    enqueues lightweight event dicts that `steps()` drains and converts to
    `TrajectoryStep`s on the asyncio side. Cancellation-safe: closing the async
    generator stops the runloop and joins the thread.
    """

    def __init__(self, *, poll_interval: float = 0.2) -> None:
        self._poll = poll_interval
        self._q: queue.Queue[dict] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._runloop = None  # CFRunLoopRef captured on the tap thread
        self._tap = None
        self._stop = threading.Event()
        self._seq = 0
        self.tap_created: bool = False

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Create the tap + spin the runloop thread. Returns whether the tap
        was created (False ⇒ TCC Input-Monitoring grant missing)."""
        import Quartz

        mask = (
            Quartz.CGEventMaskBit(Quartz.kCGEventLeftMouseDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventRightMouseDown)
            | Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown)
        )
        global _MOUSE_DOWN_TYPES, _KEY_DOWN_TYPE
        _MOUSE_DOWN_TYPES = {
            Quartz.kCGEventLeftMouseDown,
            Quartz.kCGEventRightMouseDown,
        }
        _KEY_DOWN_TYPE = Quartz.kCGEventKeyDown

        tap = Quartz.CGEventTapCreate(
            Quartz.kCGSessionEventTap,
            Quartz.kCGHeadInsertEventTap,
            Quartz.kCGEventTapOptionListenOnly,
            mask,
            self._on_event,
            None,
        )
        if tap is None:
            # No Input Monitoring permission — the documented TCC gate.
            self.tap_created = False
            logger.warning(
                "macOS capture: CGEventTapCreate returned None — Input "
                "Monitoring permission not granted to this process."
            )
            return False

        self._tap = tap
        self.tap_created = True
        self._thread = threading.Thread(target=self._run_tap, daemon=True)
        self._thread.start()
        return True

    def _run_tap(self) -> None:
        import Quartz

        self._runloop = Quartz.CFRunLoopGetCurrent()
        source = Quartz.CFMachPortCreateRunLoopSource(None, self._tap, 0)
        Quartz.CFRunLoopAddSource(self._runloop, source, Quartz.kCFRunLoopCommonModes)
        Quartz.CGEventTapEnable(self._tap, True)
        if self._stop.is_set():
            return
        Quartz.CFRunLoopRun()  # blocks until CFRunLoopStop

    def _on_event(self, proxy, etype, event, refcon):
        """CGEventTap callback (runloop thread). Enqueue a plain dict; convert
        to a TrajectoryStep on the async side. Always return the event
        (listen-only passthrough)."""
        import Quartz

        try:
            loc = Quartz.CGEventGetLocation(event)
            record = {
                "etype": int(etype),
                "x": int(loc.x),
                "y": int(loc.y),
                "ts": time.time(),
            }
            if _KEY_DOWN_TYPE is not None and int(etype) == _KEY_DOWN_TYPE:
                record["keycode"] = Quartz.CGEventGetIntegerValueField(
                    event, Quartz.kCGKeyboardEventKeycode
                )
            self._q.put_nowait(record)
        except Exception:
            logger.debug("macOS capture: event marshal failed", exc_info=True)
        return event

    def stop(self) -> None:
        self._stop.set()
        if self._runloop is not None:
            import Quartz

            Quartz.CFRunLoopStop(self._runloop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ── frontmost app / window (best-effort) ───────────────────────────────

    @staticmethod
    def _frontmost_app() -> str:
        try:
            import AppKit

            app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
            return str(app.localizedName()) if app else ""
        except Exception:
            return ""

    @staticmethod
    def _frontmost_window(app_name: str) -> str:
        """Best-effort window title. kCGWindowName requires Screen Recording
        on recent macOS; degrade to "" without it (replay still works off the
        floor screenshot + a11y)."""
        try:
            import Quartz

            info = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements,
                Quartz.kCGNullWindowID,
            )
            for w in info or []:
                if w.get("kCGWindowOwnerName") == app_name and w.get("kCGWindowLayer", 1) == 0:
                    return str(w.get("kCGWindowName", "") or "")
        except Exception:
            pass
        return ""

    def _to_step(self, rec: dict) -> TrajectoryStep:
        self._seq += 1
        if rec["etype"] in _MOUSE_DOWN_TYPES:
            action = "click"
        elif _KEY_DOWN_TYPE is not None and rec["etype"] == _KEY_DOWN_TYPE:
            action = "fill"
        else:
            action = "click"
        app = self._frontmost_app()
        return TrajectoryStep(
            seq=self._seq,
            ts=rec["ts"],
            actor="human",
            action=action,  # type: ignore[arg-type]
            screenshot_ref="",  # floor fills this; never inline bytes
            cursor=(rec["x"], rec["y"]),
            active_app=app,
            active_window=self._frontmost_window(app),
        )

    # ── the seam ───────────────────────────────────────────────────────────

    async def steps(self) -> AsyncIterator[TrajectoryStep]:
        import asyncio

        if not self.tap_created and self._tap is None:
            # Caller didn't start(), or TCC blocked it. Yield nothing.
            return
        try:
            while not self._stop.is_set():
                try:
                    rec = await asyncio.to_thread(self._q.get, True, self._poll)
                except queue.Empty:
                    continue
                yield self._to_step(rec)
        finally:
            self.stop()
