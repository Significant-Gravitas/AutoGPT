"""
Capture sources — the OS-specific seam, behind one interface.

`CaptureSource` is THE seam. Everything downstream (RecordingSession, the
handler, the wire) consumes `TrajectoryStep`s from an async iterator and never
knows whether they came from a real OS input hook, a screenshot floor, or a
scripted test list.

    ┌─────────────────────┐        steps()        ┌──────────────────┐
    │  CaptureSource (ABC) │ ───────────────────▶ │ RecordingSession │
    └─────────────────────┘  AsyncIterator[Step]  └──────────────────┘
            ▲
            ├── MockCaptureSource    (scripted list — tests + dev)
            ├── ScreenshotActionFloor (universal floor; wraps screenshot backend)
            └── A11yEnricher          (wraps a floor source, attaches AX enrichment)

WHERE NATIVE OS-HOOK PRODUCERS PLUG IN
--------------------------------------
The genuinely OS-specific part observes the user's *input* (the click/keys
that mark "a step happened"). The production macOS producer is
`MacInputCaptureSource`; Windows and Linux producers have not landed. Each is
itself a `CaptureSource`:

    `ScreenshotActionFloor(input_events=<real OS input-hook CaptureSource>)`

The floor consumes a stream of *input-event* steps from its injected
`input_events` source and, for each, snapshots the pre-action frame + cursor +
active app/window to fill the floor fields. To add another OS, write a
`CaptureSource` whose `steps()` yields one step per observed user action
(Windows `SetWindowsHookEx`, Linux X11 `XRecord`) and pass it as
`input_events`. No other layer changes. The `MockCaptureSource` remains the
reference test shape.

Browser DOM enrichment is NOT built here: it arrives from the platform/extension
side (the claude-in-chrome channel) as already-`dom`-kind steps on the wire;
the shim just accepts them.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from ..config import ShimConfig
from ..protocol import StepEnrichment, TrajectoryStep

logger = logging.getLogger(__name__)


class CaptureSource(ABC):
    """A source of TrajectorySteps. The single seam between OS-specific capture
    and the OS-agnostic recording pipeline."""

    @abstractmethod
    def steps(self) -> AsyncIterator[TrajectoryStep]:
        """Yield steps until the source is exhausted or stopped.

        Implementations should be cancellation-safe: when the consuming task is
        cancelled (STOP_RECORDING), the async generator is closed and any OS
        hooks should be torn down in a `finally`.
        """


class MockCaptureSource(CaptureSource):
    """Replays a scripted list of steps. For tests + dev.

    Also the reference shape for a real OS input-hook producer: a producer just
    needs to yield one TrajectoryStep per observed user action.
    """

    def __init__(self, scripted: list[TrajectoryStep]) -> None:
        self._scripted = list(scripted)

    async def steps(self) -> AsyncIterator[TrajectoryStep]:
        for step in self._scripted:
            yield step


class ScreenshotActionFloor(CaptureSource):
    """The universal floor (§4): snapshot the pre-action frame + cursor +
    active app/window for each observed user action.

    Wraps the existing computer-use screenshot backend. Since we cannot observe
    real user input in this environment, the floor's input-event source is
    itself an injected `CaptureSource` (`input_events`) — that is the per-OS
    seam. For each input-event step the producer yields, the floor:

      * mints a screenshot stub id (the real frame is buffered out-of-band; the
        wire carries only the stub ref, never inline bytes — §1);
      * fills `cursor` from the backend's live cursor position when the
        producer left it at the origin;
      * leaves enrichment as kind:none (the floor never enriches — that's the
        A11yEnricher / browser channel's job).

    The screenshot backend call is best-effort: if it raises (no display, no
    permission), the step still flows with its producer-supplied fields and a
    placeholder stub, because the floor degrading is better than dropping a
    user action.
    """

    def __init__(
        self,
        *,
        input_events: CaptureSource,
        config: ShimConfig,
        backend: object | None = None,
    ) -> None:
        self._input_events = input_events
        self._config = config
        # `backend` is a ComputerUseBackend; typed as object to avoid importing
        # the heavy backend module at construction. Lazily resolved.
        self._backend = backend

    def _get_backend(self) -> object | None:
        if self._backend is None:
            try:
                from ..computer_use import get_backend

                self._backend = get_backend(self._config)
            except Exception:
                logger.debug("floor: could not build computer-use backend", exc_info=True)
                self._backend = None
        return self._backend

    def _snapshot_cursor(self, fallback: tuple[int, int]) -> tuple[int, int]:
        """Best-effort live cursor read; falls back to the producer's value."""
        backend = self._get_backend()
        if backend is None:
            return fallback
        try:
            x, y, _mon = backend.cursor_position()  # type: ignore[attr-defined]
            return (int(x), int(y))
        except Exception:
            logger.debug("floor: cursor_position failed; using producer value", exc_info=True)
            return fallback

    async def steps(self) -> AsyncIterator[TrajectoryStep]:
        async for event in self._input_events.steps():
            # The producer supplies action + active_app/window. The floor owns
            # screenshot_ref and may refine cursor from the live backend.
            cursor = event.cursor
            if cursor == (0, 0):
                cursor = self._snapshot_cursor(cursor)
            # TODO(os-native): capture the real pre-action frame here and buffer
            # it under this stub id. We never inline bytes on the wire (§1).
            screenshot_ref = event.screenshot_ref or f"stub_{event.seq}"
            yield event.model_copy(update={"cursor": cursor, "screenshot_ref": screenshot_ref})


class A11yEnricher(CaptureSource):
    """Desktop a11y enrichment (§4): attach role/label/ax_path to floor steps
    where the accessibility tree resolves them; kind:none when it doesn't.

    Wraps another `CaptureSource` (the floor). For each step it asks an injected
    `resolver` for the AX element under the step's cursor. When the resolver
    returns nothing — Electron, canvas/WebGL, Wayland, games, or no permission —
    the step keeps `enrichment.kind: none` and replay falls back to visual
    grounding (§7). That is lower fidelity, NOT a failure.

    `resolver` is the per-OS seam (macOS AXObserver / Windows UI Automation /
    Linux AT-SPI). It is a callable taking (cursor, active_app, active_window)
    and returning a dict with any of {role, label, ax_path}, or None. The
    default resolver returns None for everything, so without a real per-OS
    resolver every step is kind:none — the documented degraded baseline.
    """

    def __init__(
        self,
        *,
        floor: CaptureSource,
        resolver: object | None = None,
    ) -> None:
        self._floor = floor
        self._resolver = resolver

    def _resolve(self, step: TrajectoryStep) -> dict | None:
        """Call the per-OS resolver; None on absence or any failure."""
        if self._resolver is None:
            return None
        try:
            return self._resolver(  # type: ignore[operator]
                cursor=step.cursor,
                active_app=step.active_app,
                active_window=step.active_window,
            )
        except Exception:
            logger.debug("a11y: resolver raised; treating as no enrichment", exc_info=True)
            return None

    async def steps(self) -> AsyncIterator[TrajectoryStep]:
        async for step in self._floor.steps():
            # Respect enrichment that a richer channel (browser DOM) already
            # attached — don't downgrade a dom step to ax/none.
            if step.enrichment.kind != "none":
                yield step
                continue
            resolved = self._resolve(step)
            if not resolved:
                yield step  # keeps kind:none
                continue
            yield step.model_copy(
                update={
                    "enrichment": StepEnrichment(
                        kind="ax",
                        ax_path=resolved.get("ax_path"),
                        role=resolved.get("role"),
                        label=resolved.get("label"),
                    )
                }
            )


__all__ = [
    "A11yEnricher",
    "CaptureSource",
    "MockCaptureSource",
    "ScreenshotActionFloor",
]
