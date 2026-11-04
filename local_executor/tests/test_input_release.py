"""Held keys and mouse buttons are let go of when an input action ends,
whatever ended it, on every backend."""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from autogpt_local_executor.computer_use.backends.linux import LinuxBackend
from autogpt_local_executor.computer_use.backends.macos import MacOSBackend
from autogpt_local_executor.computer_use.backends.windows import WindowsBackend
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.protocol import DisplayMonitor

_DISPLAYS = [
    DisplayMonitor(
        index=0,
        primary=True,
        physical_size=(1920, 1080),
        logical_size=(1920, 1080),
        scale=1.0,
        origin=(0, 0),
    )
]


class FailSafeException(Exception):
    pass


def _cornered_pyautogui(*, fires_on_key_down: int | None = None) -> types.SimpleNamespace:
    """A pyautogui whose cursor sits in a fail-safe corner. Like the real
    one, every call raises FailSafeException while FAILSAFE is on, except
    the key presses before the `fires_on_key_down`th, so an action can get
    part way. It records what is pressed and let go of."""
    state: dict[str, Any] = {"downs": 0}
    fake = types.SimpleNamespace(FAILSAFE=True, FailSafeException=FailSafeException)
    fake.held_keys = []
    fake.held_buttons = []
    fake.released = []

    def check() -> None:
        if fake.FAILSAFE:
            raise FailSafeException("fail-safe triggered from a corner")

    def key_down(key: str, **_kwargs) -> None:
        state["downs"] += 1
        if fires_on_key_down is None or state["downs"] >= fires_on_key_down:
            check()
        fake.held_keys.append(key)

    def key_up(key: str, **_kwargs) -> None:
        check()
        fake.released.append(key)
        fake.held_keys.remove(key)

    def mouse_down(*_args, button: str = "left", **_kwargs) -> None:
        fake.held_buttons.append(button)

    def mouse_up(*_args, button: str = "left", **_kwargs) -> None:
        check()
        fake.released.append(f"mouse:{button}")
        fake.held_buttons.remove(button)

    def move_to(*_args, **_kwargs) -> None:
        # The cursor reaches the corner part way through a drag.
        if fake.held_buttons:
            check()

    fake.keyDown = key_down
    fake.keyUp = key_up
    fake.mouseDown = mouse_down
    fake.mouseUp = mouse_up
    fake.moveTo = move_to
    fake.click = lambda *_a, **_k: check()
    fake.rightClick = fake.doubleClick = fake.middleClick = fake.tripleClick = fake.click
    return fake


def _backend(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = ShimConfig(allowed_root=tmp_path, audit_log_path=tmp_path / "audit.log")
    backend = {"windows": WindowsBackend, "linux": LinuxBackend, "macos": MacOSBackend}[name](
        config
    )
    monkeypatch.setattr(backend, "display_info", lambda: _DISPLAYS)
    if name == "macos":
        monkeypatch.setattr(backend, "_ax_trusted", lambda: True)
    if name == "linux":
        monkeypatch.setattr(backend, "_wayland", False)
    return backend


_BACKENDS = pytest.mark.parametrize("name", ["windows", "linux", "macos"])


@_BACKENDS
@pytest.mark.parametrize(
    "action",
    [
        {"action": "left_click", "coordinate": (50, 50), "modifiers": ["shift", "ctrl"]},
        {"action": "hold_key", "key": "shift+ctrl+a", "duration_ms": 0},
        {"action": "drag", "path": [(10, 10), (60, 60)], "button": "left"},
    ],
    ids=["click-with-modifiers", "hold-key", "drag"],
)
def test_the_fail_safe_never_leaves_a_key_or_button_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, action: dict
) -> None:
    """The user moves the mouse into a corner to stop the automation, and
    pyautogui's fail-safe then raises from every call, the releases in the
    actions' `finally` blocks included, so a modifier or the mouse button
    stayed held down."""
    # The presses go through; the cursor reaches the corner after them.
    pyautogui = _cornered_pyautogui(fires_on_key_down=99)
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    backend = _backend(name, tmp_path, monkeypatch)

    # A click or drag stops at the fail-safe; a hold has nothing left to do.
    with contextlib.suppress(FailSafeException):
        backend.input_action(**action)

    assert pyautogui.held_keys == []
    assert pyautogui.held_buttons == []
    assert pyautogui.released
    assert pyautogui.FAILSAFE is True


@_BACKENDS
def test_keys_pressed_before_the_fail_safe_fired_are_let_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """The modifiers were pressed before the `try`, so when the fail-safe
    fired on the second press, the first was never let go of."""
    pyautogui = _cornered_pyautogui(fires_on_key_down=2)
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    backend = _backend(name, tmp_path, monkeypatch)

    with pytest.raises(FailSafeException):
        backend.input_action("left_click", coordinate=(50, 50), modifiers=["shift", "ctrl"])

    assert pyautogui.held_keys == []
    assert pyautogui.released == ["shift"]


@_BACKENDS
def test_scroll_holds_the_requested_modifiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """The scroll branch never read `modifiers`, so Shift+scroll (sideways
    in most apps) went out as a plain scroll and was acknowledged."""
    from unittest.mock import MagicMock, call

    pyautogui = MagicMock()
    pyautogui.FAILSAFE = True
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    backend = _backend(name, tmp_path, monkeypatch)
    if name == "windows":
        monkeypatch.setattr(backend, "_turn_wheel", pyautogui.scroll)

    backend.input_action(
        "scroll", coordinate=(50, 50), modifiers=["shift"], scroll_direction="down", scroll_amount=2
    )

    pressed = [c for c in pyautogui.mock_calls if c[0] in ("keyDown", "keyUp", "scroll")]
    assert [c[0] for c in pressed] == ["keyDown", "scroll", "keyUp"]
    assert pressed[0] == call.keyDown("shift") and pressed[2] == call.keyUp("shift")
