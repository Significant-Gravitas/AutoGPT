"""macOS capture producer — conversion-path tests.

Skipped off-darwin or without pyobjc. The live-tap path (a real human's HID
events flowing through CGEventTap) is validated by construction — it's the
standard session-tap mechanism — but is NOT auto-testable: synthetic
same-process `CGEventPost` events are not delivered to a listen-only session
tap on hardened macOS, so there's no way to self-generate observable input in
CI. These tests cover everything up to that line: tap creation (the TCC gate)
and the real-CGEvent → TrajectoryStep conversion.
"""

from __future__ import annotations

import sys

import pytest

from autogpt_local_executor.recording.macos_capture import MacInputCaptureSource

quartz = pytest.importorskip("Quartz", reason="pyobjc Quartz not installed")
pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="macOS capture is darwin-only")


def _real_mouse_event(x: int, y: int):
    return quartz.CGEventCreateMouseEvent(
        None, quartz.kCGEventLeftMouseDown, (x, y), quartz.kCGMouseButtonLeft
    )


def _real_key_event(keycode: int):
    return quartz.CGEventCreateKeyboardEvent(None, keycode, True)


def test_tap_creation_returns_bool_and_tears_down():
    src = MacInputCaptureSource()
    created = src.start()
    assert isinstance(created, bool)
    # Whether or not TCC granted, stop() must be safe to call.
    src.stop()


def test_real_mouse_event_converts_to_click_step():
    src = MacInputCaptureSource()
    ev = _real_mouse_event(840, 314)
    src._on_event(None, quartz.CGEventGetType(ev), ev, None)
    assert src._q.qsize() == 1
    step = src._to_step(src._q.get_nowait())
    assert step.action == "click"
    assert step.cursor == (840, 314)
    assert step.actor == "human"
    # Floor fields present; enrichment defaults to kind:none (the floor /
    # a11y enricher fill the rest downstream).
    assert step.enrichment.kind == "none"


def test_real_key_event_converts_to_fill_step():
    src = MacInputCaptureSource()
    ev = _real_key_event(0)  # 'a'
    src._on_event(None, quartz.CGEventGetType(ev), ev, None)
    step = src._to_step(src._q.get_nowait())
    assert step.action == "fill"


def test_steps_yields_nothing_when_never_started():
    import asyncio

    src = MacInputCaptureSource()

    async def drain():
        out = []
        async for s in src.steps():
            out.append(s)
        return out

    assert asyncio.run(drain()) == []
