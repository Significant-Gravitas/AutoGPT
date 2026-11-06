"""
Tests for MacInputCaptureSource's tap thread, using a fake Quartz module.

These run on every OS. The fake stands in for pyobjc and models the CFRunLoop
behaviour the stop() race depends on: CFRunLoopStop stops a run loop only if
it is running at that moment, and does nothing otherwise.
"""

from __future__ import annotations

import sys
import threading
import types
from collections.abc import Callable, Iterator

import pytest

from autogpt_local_executor.recording import macos_capture
from autogpt_local_executor.recording.macos_capture import MacInputCaptureSource

# Generous bound for cross-thread handshakes, so a slow runner never flakes.
_WAIT = 10.0


class _FakeRunLoop:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.wake = threading.Event()


class _FakeQuartz(types.ModuleType):
    """Just enough of pyobjc's Quartz for the CGEventTap thread."""

    kCGEventLeftMouseDown = 1
    kCGEventRightMouseDown = 3
    kCGEventKeyDown = 10
    kCGSessionEventTap = 1
    kCGHeadInsertEventTap = 0
    kCGEventTapOptionListenOnly = 1
    kCFRunLoopCommonModes = "kCFRunLoopCommonModes"
    kCFRunLoopDefaultMode = "kCFRunLoopDefaultMode"
    kCFRunLoopRunFinished = 1
    kCFRunLoopRunStopped = 2
    kCFRunLoopRunTimedOut = 3
    kCFRunLoopRunHandledSource = 4

    def __init__(self) -> None:
        super().__init__("Quartz")
        self.loop = _FakeRunLoop()
        self.tap_enabled = False
        self.tap_invalidated = False
        self.no_sources = False
        self.released = False
        self.loop_running = threading.Event()
        self.stop_called = threading.Event()
        self.before_first_run: Callable[[], None] | None = None

    def CGEventMaskBit(self, event_type: int) -> int:
        return 1 << event_type

    def CGEventTapCreate(self, *_args: object) -> object:
        self.tap_enabled = True  # a new tap starts enabled
        return "tap"

    def CGEventTapEnable(self, _tap: object, enable: bool) -> None:
        self.tap_enabled = enable

    def CFMachPortInvalidate(self, _port: object) -> None:
        self.tap_invalidated = True

    def CFMachPortCreateRunLoopSource(self, *_args: object) -> object:
        return "source"

    def CFRunLoopGetCurrent(self) -> _FakeRunLoop:
        return self.loop

    def CFRunLoopAddSource(self, *_args: object) -> None:
        pass

    def CFRunLoopRemoveSource(self, *_args: object) -> None:
        pass

    def CFRunLoopStop(self, loop: _FakeRunLoop) -> None:
        with loop.lock:
            if loop.running:
                loop.wake.set()
        self.stop_called.set()

    def CFRunLoopRun(self) -> None:
        self._run(None)

    def CFRunLoopRunInMode(self, _mode: object, seconds: float, _return_after: bool) -> int:
        return self._run(seconds)

    def _run(self, seconds: float | None) -> int:
        hook, self.before_first_run = self.before_first_run, None
        if hook is not None:
            hook()
        if self.no_sources:
            return self.kCFRunLoopRunFinished
        with self.loop.lock:
            if self.released:
                return self.kCFRunLoopRunStopped
            self.loop.running = True
        self.loop_running.set()
        stopped = self.loop.wake.wait(seconds)
        with self.loop.lock:
            self.loop.running = False
            self.loop.wake.clear()
        return self.kCFRunLoopRunStopped if stopped else self.kCFRunLoopRunTimedOut

    def release(self) -> None:
        """End any run loop the code under test left running, so its thread exits."""
        with self.loop.lock:
            self.released = True
            self.loop.wake.set()


@pytest.fixture
def quartz(monkeypatch: pytest.MonkeyPatch) -> _FakeQuartz:
    fake = _FakeQuartz()
    monkeypatch.setitem(sys.modules, "Quartz", fake)
    # start() rebinds these module globals; put them back for other tests.
    monkeypatch.setattr(macos_capture, "_MOUSE_DOWN_TYPES", macos_capture._MOUSE_DOWN_TYPES)
    monkeypatch.setattr(macos_capture, "_KEY_DOWN_TYPE", macos_capture._KEY_DOWN_TYPE)
    return fake


@pytest.fixture
def source(quartz: _FakeQuartz) -> Iterator[MacInputCaptureSource]:
    src = MacInputCaptureSource()
    yield src
    src._stop.set()
    quartz.release()
    if src._thread is not None:
        src._thread.join(_WAIT)


def _assert_tap_ended(src: MacInputCaptureSource, quartz: _FakeQuartz) -> None:
    assert src._thread is not None
    src._thread.join(_WAIT)
    assert not src._thread.is_alive(), "the tap thread kept running after stop()"
    assert quartz.tap_enabled is False, "the event tap was left enabled"
    assert quartz.tap_invalidated is True, "the event tap's port was never invalidated"


def test_stop_just_before_the_run_loop_starts_still_ends_the_tap(
    quartz: _FakeQuartz, source: MacInputCaptureSource
) -> None:
    entered = threading.Event()

    def stop_lands_first() -> None:
        # Hold the tap thread after it last checked the stop flag, until stop()
        # has called CFRunLoopStop on a run loop that is not running yet.
        entered.set()
        quartz.stop_called.wait(_WAIT)

    quartz.before_first_run = stop_lands_first
    assert source.start() is True
    assert entered.wait(_WAIT)

    source.stop()

    _assert_tap_ended(source, quartz)


def test_stop_while_the_run_loop_runs_tears_the_tap_down(
    quartz: _FakeQuartz, source: MacInputCaptureSource
) -> None:
    assert source.start() is True
    assert quartz.loop_running.wait(_WAIT)

    source.stop()

    _assert_tap_ended(source, quartz)


def test_tap_thread_ends_when_its_run_loop_has_nothing_left_to_run(
    quartz: _FakeQuartz, source: MacInputCaptureSource
) -> None:
    quartz.no_sources = True

    assert source.start() is True

    _assert_tap_ended(source, quartz)


def test_the_quartz_names_the_tap_thread_uses_exist() -> None:
    """The tap thread only runs on a Mac with Input Monitoring, so check the
    pyobjc names it calls where pyobjc is installed (macOS CI)."""
    quartz = pytest.importorskip("Quartz")
    for name in (
        "CFMachPortCreateRunLoopSource",
        "CFMachPortInvalidate",
        "CFRunLoopAddSource",
        "CFRunLoopGetCurrent",
        "CFRunLoopRunInMode",
        "CFRunLoopStop",
        "CGEventTapEnable",
        "kCFRunLoopCommonModes",
        "kCFRunLoopDefaultMode",
        "kCFRunLoopRunFinished",
    ):
        assert hasattr(quartz, name), name
