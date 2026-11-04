"""macOS backend tests.

Covers the macOS-specific code paths in
autogpt_local_executor/computer_use/backends/macos.py without ever
touching Quartz / AppKit / pyobjc for real. Tests run on any OS because
we monkeypatch every imported native call.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from autogpt_local_executor.computer_use.backends._common import PASTE_THRESHOLD_CHARS
from autogpt_local_executor.computer_use.backends.macos import (
    CONCEALED_PB_TYPES,
    MacOSBackend,
)
from autogpt_local_executor.computer_use.errors import (
    ClipboardConcealedError,
    ClipboardWriteError,
    FeatureNotSupportedError,
    InputOutOfBoundsError,
    PermissionPendingError,
    WindowStaleError,
)
from autogpt_local_executor.computer_use.window_registry import WindowFingerprint
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.protocol import DisplayMonitor

# ── Test helpers ───────────────────────────────────────────────────────


def _config(tmp_path: Path, **overrides) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "_audit.log",
        enable_computer_use=True,
        platform_url="http://localhost:9999",
        machine_id="test-machine",
        **overrides,
    )


@pytest.fixture
def displays_1080p() -> list[DisplayMonitor]:
    return [
        DisplayMonitor(
            index=0,
            primary=True,
            physical_size=(1920, 1080),
            logical_size=(1920, 1080),
            scale=1.0,
            origin=(0, 0),
        )
    ]


@pytest.fixture
def displays_dual() -> list[DisplayMonitor]:
    return [
        DisplayMonitor(
            index=0,
            primary=True,
            physical_size=(1920, 1080),
            logical_size=(1920, 1080),
            scale=1.0,
            origin=(0, 0),
        ),
        DisplayMonitor(
            index=1,
            primary=False,
            physical_size=(2560, 1440),
            logical_size=(2560, 1440),
            scale=1.0,
            origin=(1920, 0),
        ),
    ]


# ── Capability advertisement ───────────────────────────────────────────


def test_coarse_features_default_no_clipboard(tmp_path: Path) -> None:
    b = MacOSBackend(_config(tmp_path))
    feats = b.coarse_features()
    assert "screenshot" in feats
    assert "input" in feats
    assert "clipboard" not in feats


def test_coarse_features_with_clipboard_flag(tmp_path: Path) -> None:
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    feats = b.coarse_features()
    assert "clipboard" in feats


def test_features_lists_screenshot_region(tmp_path: Path) -> None:
    b = MacOSBackend(_config(tmp_path))
    assert "screenshot.region" in b.features()


# ── Bounds (Q1) ────────────────────────────────────────────────────────


def test_input_action_raises_on_out_of_bounds(tmp_path, displays_1080p, monkeypatch) -> None:
    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    with pytest.raises(InputOutOfBoundsError) as ei:
        b.input_action("left_click", coordinate=(10000, 10000))
    assert ei.value.details["requested_coordinate"] == [10000, 10000]
    assert ei.value.details["displays"] == [{"index": 0, "origin": [0, 0], "size": [1920, 1080]}]


def test_input_action_in_bounds_on_secondary_display(tmp_path, displays_dual, monkeypatch) -> None:
    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: displays_dual)
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    # Stub pyautogui via sys.modules.
    fake_pg = MagicMock()
    fake_pg.FAILSAFE = True
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)
    b.input_action("left_click", coordinate=(2000, 500))  # in monitor 1
    fake_pg.click.assert_called_once_with(2000, 500)


def test_drag_path_checks_each_point(tmp_path, displays_1080p, monkeypatch) -> None:
    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    fake_pg = MagicMock()
    fake_pg.FAILSAFE = True
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)
    with pytest.raises(InputOutOfBoundsError):
        b.input_action("drag", path=[(10, 10), (100, 100), (10000, 10000)], button="left")


# ── AX gate (Q5) ────────────────────────────────────────────────────────


def test_input_action_raises_permission_pending_when_ax_denied(
    tmp_path, displays_1080p, monkeypatch
) -> None:
    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    monkeypatch.setattr(b, "_ax_trusted", lambda: False)
    with pytest.raises(PermissionPendingError) as ei:
        b.input_action("left_click", coordinate=(100, 100))
    assert ei.value.details["permission"] == "accessibility"
    assert ei.value.details["platform"] == "darwin"


def test_wait_action_does_not_check_ax(tmp_path, monkeypatch) -> None:
    """`wait` is the one INPUT_ACTION that shouldn't need AX (it's just
    a sleep)."""
    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(b, "_ax_trusted", lambda: False)
    slept: list[float] = []
    monkeypatch.setattr(time, "sleep", slept.append)
    b.input_action("wait", duration_ms=20)
    assert slept == [0.02]


# ── Clipboard (Q3) ─────────────────────────────────────────────────────


def test_clipboard_disabled_returns_feature_not_supported(tmp_path: Path) -> None:
    b = MacOSBackend(_config(tmp_path))  # clipboard off by default
    with pytest.raises(FeatureNotSupportedError):
        b.clipboard_read()
    with pytest.raises(FeatureNotSupportedError):
        b.clipboard_write(content="x")


def test_clipboard_writeback_only_round_trip(tmp_path, monkeypatch) -> None:
    """Default (--enable-clipboard only): we write, then we read, we get
    our own content back."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    # Simulate NSPasteboard via the read/write hooks.
    state = {"content": None, "seq": 0, "types": []}

    def _write(content: str) -> int:
        state["content"] = content
        state["seq"] += 1
        return state["seq"]

    def _read():
        return state["content"], state["seq"]

    monkeypatch.setattr(b, "_write_pasteboard_text", _write)
    monkeypatch.setattr(b, "_read_pasteboard_text", _read)
    monkeypatch.setattr(b, "_read_pasteboard_types", lambda: state["types"])
    b.clipboard_write(content="hello")
    res = b.clipboard_read()
    assert res.content == "hello"
    assert res.size_bytes == 5


def test_clipboard_writeback_only_refuses_foreign_content(tmp_path, monkeypatch) -> None:
    """Default: user copies something else → CLIPBOARD_CONCEALED."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_read_pasteboard_text", lambda: ("user_text", 99))
    monkeypatch.setattr(b, "_read_pasteboard_types", lambda: [])
    with pytest.raises(ClipboardConcealedError) as ei:
        b.clipboard_read()
    # Either writeback_only (we never wrote) or writeback_overwritten
    # (we wrote but they overwrote). Both are valid per the spec.
    assert ei.value.details["reason"] in ("writeback_only", "writeback_overwritten")


def test_clipboard_read_foreign_returns_user_content(tmp_path, monkeypatch) -> None:
    """With --enable-clipboard-read-foreign, foreign content is returned
    (subject to the concealed-type check)."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_read_pasteboard_text", lambda: ("user_text", 99))
    monkeypatch.setattr(b, "_read_pasteboard_types", lambda: [])
    res = b.clipboard_read()
    assert res.content == "user_text"


def test_clipboard_read_refuses_concealed_type(tmp_path, monkeypatch) -> None:
    """ConcealedType marker on the pasteboard → refuse even with
    --enable-clipboard-read-foreign."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_read_pasteboard_text", lambda: ("secret", 1))
    marker = next(iter(CONCEALED_PB_TYPES))
    monkeypatch.setattr(b, "_read_pasteboard_types", lambda: [marker])
    with pytest.raises(ClipboardConcealedError) as ei:
        b.clipboard_read()
    assert ei.value.details["reason"] == "concealed_type"
    assert ei.value.details["marker"] == marker


def test_clipboard_writeback_window_expires(tmp_path, monkeypatch) -> None:
    """Q3: read after the 30 s window → CLIPBOARD_CONCEALED."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    # Force a tiny window so the test runs fast.
    b.clipboard_writeback = b.clipboard_writeback.__class__(window_seconds=0.01)
    state = {"content": None, "seq": 0}

    def _write(content: str) -> int:
        state["content"] = content
        state["seq"] += 1
        return state["seq"]

    monkeypatch.setattr(b, "_write_pasteboard_text", _write)
    monkeypatch.setattr(b, "_read_pasteboard_text", lambda: (state["content"], state["seq"]))
    monkeypatch.setattr(b, "_read_pasteboard_types", lambda: [])
    b.clipboard_write(content="hi")
    time.sleep(0.05)
    with pytest.raises(ClipboardConcealedError) as ei:
        b.clipboard_read()
    assert ei.value.details["reason"] == "writeback_only"


# ── Paste threshold + restore (Q4) ────────────────────────────────────


def test_paste_falls_through_under_threshold(tmp_path, monkeypatch, displays_1080p) -> None:
    """Strings < 200 chars get per-key typed even when paste=True."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    fake_pg = MagicMock()
    fake_pg.FAILSAFE = True
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)
    b.input_action("type", text="short", paste=True)
    fake_pg.write.assert_called_once_with("short", interval=0.02)
    fake_pg.hotkey.assert_not_called()


def test_paste_uses_hotkey_above_threshold(tmp_path, monkeypatch, displays_1080p) -> None:
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    monkeypatch.setattr(b, "_write_pasteboard_text", lambda c: 1)
    monkeypatch.setattr(b, "_read_pasteboard_text", lambda: (None, 1))
    fake_pg = MagicMock()
    fake_pg.FAILSAFE = True
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)
    big = "x" * 250
    b.input_action("type", text=big, paste=True)
    fake_pg.hotkey.assert_called_once_with("command", "v")
    fake_pg.write.assert_not_called()


def test_preserve_clipboard_restores_when_changecount_advanced_by_one(
    tmp_path, monkeypatch, displays_1080p
) -> None:
    """Q4: with preserve_clipboard=True, restore happens when the only
    advance to changeCount is our own paste write (seq + 1)."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    state = {"content": "PREVIOUS", "seq": 10}

    def _write(content: str) -> int:
        state["content"] = content
        state["seq"] += 1
        return state["seq"]

    monkeypatch.setattr(b, "_write_pasteboard_text", _write)
    monkeypatch.setattr(b, "_read_pasteboard_text", lambda: (state["content"], state["seq"]))
    monkeypatch.setattr(b, "_read_pasteboard_types", lambda: ["public.utf8-plain-text"])
    fake_pg = MagicMock()
    fake_pg.FAILSAFE = True
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)

    big = "x" * 250
    b.input_action("type", text=big, paste=True, preserve_clipboard=True)
    # After the paste sequence: snapshot at seq=10, then we write
    # (seq=11), then we restore to PREVIOUS (seq=12). The clipboard
    # content should be PREVIOUS again.
    fake_pg.hotkey.assert_called_once_with("command", "v")
    assert state["content"] == "PREVIOUS"


def test_preserve_clipboard_skips_restore_when_user_copied_during_paste(
    tmp_path, monkeypatch, displays_1080p
) -> None:
    """Q4: if changeCount advanced past `prev_seq + 1` between snapshot
    and restore, a third party wrote — skip restore."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)

    # Simulate: snapshot returns ("PREVIOUS", 10). _write_pasteboard_text
    # bumps seq by 1 (our paste). Then the second read returns (current,
    # current_seq) where current_seq=12 (user copied during paste, bumping
    # it past 11).
    reads: list[tuple[str | None, int]] = [
        ("PREVIOUS", 10),  # initial snapshot
        ("PREVIOUS", 10),  # the snapshot's check that nothing changed
        ("USER_COPIED", 12),  # second read after paste
    ]

    def _read():
        return reads.pop(0)

    write_count = {"n": 0}

    def _write(content: str) -> int:
        write_count["n"] += 1
        # +1 from the paste — that's the only write we should see.
        return 11

    monkeypatch.setattr(b, "_write_pasteboard_text", _write)
    monkeypatch.setattr(b, "_read_pasteboard_text", _read)
    monkeypatch.setattr(b, "_read_pasteboard_types", lambda: ["public.utf8-plain-text"])
    fake_pg = MagicMock()
    fake_pg.FAILSAFE = True
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)

    big = "x" * 250
    b.input_action("type", text=big, paste=True, preserve_clipboard=True)
    # Exactly one write — the paste. NOT the restore.
    assert write_count["n"] == 1


# ── Window registry (Q2) ───────────────────────────────────────────────


def test_window_focus_with_stale_handle_raises(tmp_path, monkeypatch) -> None:
    b = MacOSBackend(_config(tmp_path))
    fp = WindowFingerprint(pid=999, class_name="A", creation_timestamp=None)
    wid = b.window_registry.mint(424242, fp, extra={"bounds": (0, 0, 100, 100)})
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    monkeypatch.setattr(b, "_live_fingerprint", lambda _: None)
    with pytest.raises(WindowStaleError) as ei:
        b.window_focus(wid)
    assert ei.value.details["window_id"] == wid


def test_window_list_clears_registry_each_call(tmp_path, monkeypatch) -> None:
    """Repeat list calls should not leak window_ids."""
    b = MacOSBackend(_config(tmp_path))
    fp = WindowFingerprint(pid=1, class_name="X", creation_timestamp=None)
    b.window_registry.mint(1, fp)
    b.window_registry.mint(2, fp)
    assert len(b.window_registry) == 2

    # Mock Quartz to return one entry.
    fake_qz = MagicMock()
    fake_qz.CGWindowListCopyWindowInfo.return_value = [
        {
            "kCGWindowNumber": 555,
            "kCGWindowOwnerPID": 100,
            "kCGWindowOwnerName": "Term",
            "kCGWindowName": "Title",
            "kCGWindowBounds": {"X": 0, "Y": 0, "Width": 100, "Height": 100},
        }
    ]
    fake_qz.kCGNullWindowID = 0
    fake_qz.kCGWindowListExcludeDesktopElements = 0
    fake_qz.kCGWindowListOptionOnScreenOnly = 0
    monkeypatch.setitem(sys.modules, "Quartz", fake_qz)
    monkeypatch.setattr(b, "_focused_pid", lambda: 100)
    monkeypatch.setattr(b, "_bundle_for_pid", lambda _: None)
    out = b.window_list()
    assert len(out) == 1
    # And the registry now has exactly one entry (the new one).
    assert len(b.window_registry) == 1


# ── Permissions check ─────────────────────────────────────────────────


def test_permissions_check_uses_ax_and_screen_capture(tmp_path, monkeypatch) -> None:
    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    monkeypatch.setattr(b, "_screen_capture_allowed", lambda: False)
    out = b.permissions_check(["accessibility", "screen_recording", "input_monitoring"])
    assert out == {
        "accessibility": "granted",
        "screen_recording": "denied",
        "input_monitoring": "unknown",
    }


def test_permissions_check_unknown_keys_are_not_applicable(tmp_path, monkeypatch) -> None:
    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    monkeypatch.setattr(b, "_screen_capture_allowed", lambda: True)
    out = b.permissions_check(["camera", "microphone"])
    assert out == {"camera": "not_applicable", "microphone": "not_applicable"}


# ── Paste with preserve_clipboard (Q3/Q4) ───────────────────────────────


def _fake_pasteboard(
    b: MacOSBackend, monkeypatch, *, content: str, types=("public.utf8-plain-text",)
) -> dict:
    state: dict = {"content": content, "seq": 1, "types": list(types), "writes": []}

    def _write(text: str) -> int:
        state["content"] = text
        state["seq"] += 1
        state["writes"].append(text)
        return state["seq"]

    monkeypatch.setattr(b, "_write_pasteboard_text", _write)
    monkeypatch.setattr(b, "_read_pasteboard_text", lambda: (state["content"], state["seq"]))
    monkeypatch.setattr(b, "_read_pasteboard_types", lambda: state["types"])
    return state


def test_a_clipboard_restored_after_a_paste_stays_unreadable(tmp_path, monkeypatch) -> None:
    """The user's clipboard goes back after the paste, and it is still theirs:
    a writeback-only read must not return it."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    monkeypatch.setitem(sys.modules, "pyautogui", MagicMock())
    clipboard = _fake_pasteboard(b, monkeypatch, content="user secret")

    b.input_action("type", text="x" * PASTE_THRESHOLD_CHARS, paste=True, preserve_clipboard=True)

    assert clipboard["content"] == "user secret"
    with pytest.raises(ClipboardConcealedError):
        b.clipboard_read()


def test_a_concealed_clipboard_is_left_alone_and_the_text_typed(tmp_path, monkeypatch) -> None:
    """Restoring would drop the concealed marker, so the clipboard is never
    read or replaced."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_pasteboard(
        b, monkeypatch, content="password", types=["org.nspasteboard.ConcealedType"]
    )
    text = "x" * PASTE_THRESHOLD_CHARS

    b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    assert clipboard["writes"] == []
    pyautogui.write.assert_called_once_with(text, interval=0.02)
    pyautogui.hotkey.assert_not_called()


# ── Window focus through Accessibility (WINDOW_FOCUS) ──────────────────


def _focusable(b: MacOSBackend, monkeypatch, *, cg_frame, cg_title, ax_windows) -> dict:
    """One CoreGraphics window to focus, and the app's Accessibility windows
    as (frame, title) pairs."""
    fp = WindowFingerprint(pid=4242, class_name="Editor", creation_timestamp=None)
    window_id = b.window_registry.mint(7, fp, extra={"bounds": (0, 0, 10, 10)})
    app = MagicMock()
    appkit = MagicMock()
    appkit.NSRunningApplication.runningApplicationWithProcessIdentifier_.return_value = app
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    monkeypatch.setattr(b, "_live_fingerprint", lambda _handle: fp)
    monkeypatch.setattr(b, "_cg_window_frame", lambda _handle: (cg_frame, cg_title))
    monkeypatch.setattr(b, "_ax_windows", lambda pid: list(range(len(ax_windows))))
    monkeypatch.setattr(b, "_ax_frame", lambda window: ax_windows[window][0])
    monkeypatch.setattr(b, "_ax_title", lambda window: ax_windows[window][1])
    raised: list[int] = []
    monkeypatch.setattr(b, "_ax_raise", lambda window: raised.append(window) or True)
    return {"window_id": window_id, "app": app, "raised": raised}


def test_window_focus_raises_the_chosen_window_not_just_its_app(tmp_path, monkeypatch) -> None:
    b = MacOSBackend(_config(tmp_path))
    focus = _focusable(
        b,
        monkeypatch,
        cg_frame=(400, 300, 800, 600),
        cg_title="",
        ax_windows=[((0, 25, 800, 600), "Notes"), ((400, 300, 800, 600), "Report")],
    )

    b.window_focus(focus["window_id"])

    assert focus["raised"] == [1]
    focus["app"].activateWithOptions_.assert_called_once()


def test_window_focus_tells_same_frame_windows_apart_by_title(tmp_path, monkeypatch) -> None:
    b = MacOSBackend(_config(tmp_path))
    focus = _focusable(
        b,
        monkeypatch,
        cg_frame=(0, 25, 800, 600),
        cg_title="Draft 2",
        ax_windows=[((0, 25, 800, 600), "Draft 1"), ((0, 25, 800, 600), "Draft 2")],
    )

    b.window_focus(focus["window_id"])

    assert focus["raised"] == [1]


def test_window_focus_reports_a_window_it_cannot_single_out(tmp_path, monkeypatch) -> None:
    """Acknowledging would let the next keystrokes land in another window."""
    b = MacOSBackend(_config(tmp_path))
    focus = _focusable(
        b,
        monkeypatch,
        cg_frame=(0, 25, 800, 600),
        cg_title="",
        ax_windows=[((0, 25, 800, 600), "Draft 1"), ((0, 25, 800, 600), "Draft 2")],
    )

    with pytest.raises(WindowStaleError):
        b.window_focus(focus["window_id"])

    assert focus["raised"] == []
    focus["app"].activateWithOptions_.assert_not_called()


def test_the_accessibility_names_used_for_focus_exist() -> None:
    """The focus path's pyobjc calls only run on macOS; check their names there."""
    services = pytest.importorskip("ApplicationServices")
    for name in (
        "AXUIElementCopyAttributeValue",
        "AXUIElementCreateApplication",
        "AXUIElementPerformAction",
        "AXUIElementSetAttributeValue",
        "AXValueGetValue",
        "kAXErrorSuccess",
        "kAXMainAttribute",
        "kAXPositionAttribute",
        "kAXRaiseAction",
        "kAXSizeAttribute",
        "kAXTitleAttribute",
        "kAXValueCGPointType",
        "kAXValueCGSizeType",
        "kAXWindowsAttribute",
    ):
        assert hasattr(services, name), name


def test_a_failed_pasteboard_write_is_an_error_and_never_pastes(tmp_path, monkeypatch) -> None:
    """Pasting after a failed write would paste whatever the user had copied."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    monkeypatch.setitem(sys.modules, "AppKit", MagicMock())
    pasteboard = MagicMock()
    pasteboard.setString_forType_.return_value = False
    monkeypatch.setattr(b, "_pasteboard", lambda: pasteboard)

    with pytest.raises(ClipboardWriteError):
        b.clipboard_write(content="hello")
    with pytest.raises(ClipboardWriteError):
        b.input_action("type", text="x" * PASTE_THRESHOLD_CHARS, paste=True)

    pyautogui.hotkey.assert_not_called()


def test_input_bounds_are_in_points_on_a_retina_display(tmp_path, monkeypatch) -> None:
    """Quartz gives origins and logical sizes in points, and pixel counts twice
    as large; pyautogui clicks in points, so the bounds must be in points too."""
    retina = [
        DisplayMonitor(
            index=0,
            primary=True,
            physical_size=(2880, 1800),
            logical_size=(1440, 900),
            scale=2.0,
            origin=(0, 0),
        )
    ]
    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: retina)
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    fake_pg = MagicMock()
    fake_pg.FAILSAFE = True
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)

    b.input_action("left_click", coordinate=(1400, 880))
    with pytest.raises(InputOutOfBoundsError) as ei:
        b.input_action("left_click", coordinate=(2000, 1000))

    fake_pg.click.assert_called_once_with(1400, 880)
    assert ei.value.details["displays"] == [{"index": 0, "origin": [0, 0], "size": [1440, 900]}]


# ── App launch by bundle ID ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("args", "activate", "expected"),
    [
        (None, True, ["open", "-b", "com.apple.TextEdit"]),
        (
            ["/Users/a/notes.txt"],
            False,
            ["open", "-b", "com.apple.TextEdit", "-g", "--args", "/Users/a/notes.txt"],
        ),
    ],
)
def test_bundle_launch_passes_args_and_activation(
    tmp_path, monkeypatch, args, activate, expected
) -> None:
    import subprocess

    from autogpt_local_executor.computer_use.backends import macos as macos_backend

    b = MacOSBackend(_config(tmp_path))
    commands: list[list[str]] = []

    def run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(macos_backend.subprocess, "run", run)
    monkeypatch.setattr(b, "_pid_for_bundle", lambda _bundle: 4242)

    pid = b.app_launch(bundle_id="com.apple.TextEdit", args=args, activate=activate)

    assert pid == 4242
    assert commands == [expected]


def test_bundle_launch_reports_an_app_open_cannot_find(tmp_path, monkeypatch) -> None:
    import subprocess

    from autogpt_local_executor.computer_use.backends import macos as macos_backend

    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(
        macos_backend.subprocess,
        "run",
        lambda command, **_kwargs: subprocess.CompletedProcess(
            command, 1, stdout=b"", stderr=b"Unable to find application named 'x'"
        ),
    )

    with pytest.raises(FeatureNotSupportedError, match="Unable to find application"):
        b.app_launch(bundle_id="com.example.missing")


# ── Window list: include_minimized ───────────────────────────────────────


@pytest.mark.parametrize(
    ("include_minimized", "include_offscreen", "on_screen_only"),
    [(False, False, True), (True, False, False), (False, True, False)],
)
def test_window_list_asks_for_off_screen_windows_when_minimized_ones_are_wanted(
    tmp_path, monkeypatch, include_minimized, include_offscreen, on_screen_only
) -> None:
    """Minimized windows are off screen, so on-screen-only listing never has them."""
    fake_qz = MagicMock()
    fake_qz.kCGWindowListExcludeDesktopElements = 0x10
    fake_qz.kCGWindowListOptionOnScreenOnly = 0x1
    fake_qz.CGWindowListCopyWindowInfo.return_value = []
    monkeypatch.setitem(sys.modules, "Quartz", fake_qz)
    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(b, "_focused_pid", lambda: None)

    b.window_list(include_minimized=include_minimized, include_offscreen=include_offscreen)

    options = fake_qz.CGWindowListCopyWindowInfo.call_args.args[0]
    assert bool(options & 0x1) is on_screen_only


@pytest.mark.parametrize(
    "types",
    [None, ["public.png"], ["public.utf8-plain-text", "public.rtf"]],
    ids=["unreadable", "image", "rich-text"],
)
def test_preserving_a_clipboard_a_text_restore_cannot_rebuild_types_instead(
    tmp_path, monkeypatch, types
) -> None:
    """The restore puts back plain text only, so pasting over an image, rich
    text, or a clipboard whose types can't be read would lose it."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_pasteboard(b, monkeypatch, content="kept")
    monkeypatch.setattr(b, "_read_pasteboard_types", lambda: types)
    text = "x" * PASTE_THRESHOLD_CHARS

    b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    assert clipboard["writes"] == []
    pyautogui.write.assert_called_once_with(text, interval=0.02)


def test_a_clipboard_read_refuses_when_its_types_cannot_be_read(tmp_path, monkeypatch) -> None:
    """Unreadable types could be hiding a concealed marker."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_read_pasteboard_text", lambda: ("password", 3))
    monkeypatch.setattr(b, "_read_pasteboard_types", lambda: None)

    with pytest.raises(ClipboardConcealedError):
        b.clipboard_read()


def test_window_focus_fails_when_macos_will_not_activate_the_app(tmp_path, monkeypatch) -> None:
    b = MacOSBackend(_config(tmp_path))
    focus = _focusable(
        b,
        monkeypatch,
        cg_frame=(0, 25, 800, 600),
        cg_title="",
        ax_windows=[((0, 25, 800, 600), "Notes")],
    )
    focus["app"].activateWithOptions_.return_value = False

    with pytest.raises(WindowStaleError):
        b.window_focus(focus["window_id"])


# ── Retina geometry ──────────────────────────────────────────────────────


def _retina_quartz() -> MagicMock:
    """One 1440x900-point display backed by 2880x1800 pixels. Like the real
    API, CGDisplayPixelsWide reports the mode's size in points."""
    quartz = MagicMock()
    quartz.CGGetActiveDisplayList.return_value = (0, [1], 1)
    quartz.CGMainDisplayID.return_value = 1
    quartz.CGDisplayBounds.return_value = SimpleNamespace(
        origin=SimpleNamespace(x=0, y=0), size=SimpleNamespace(width=1440, height=900)
    )
    quartz.CGDisplayPixelsWide.return_value = 1440
    quartz.CGDisplayPixelsHigh.return_value = 900
    quartz.CGDisplayModeGetPixelWidth.return_value = 2880
    quartz.CGDisplayModeGetPixelHeight.return_value = 1800
    return quartz


def test_display_info_reads_the_backing_pixels_of_a_retina_display(tmp_path, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "Quartz", _retina_quartz())

    [display] = MacOSBackend(_config(tmp_path)).display_info()

    assert display.logical_size == (1440, 900)
    assert display.physical_size == (2880, 1800)
    assert display.scale == 2.0


def test_a_window_capture_reports_its_scale_and_size_in_points(tmp_path, monkeypatch) -> None:
    """CGWindowListCreateImage returns backing pixels, twice the window's
    800x600-point bounds on a Retina display. It must leave out the window's
    shadow, or the image covers more than the bounds that `origin`,
    `logical_size` and `display_scale` describe."""
    from autogpt_local_executor.computer_use.backends import macos as macos_backend

    quartz = MagicMock()
    # No live frame: the bounds the window was listed with are used.
    quartz.CGWindowListCopyWindowInfo.return_value = []
    monkeypatch.setitem(sys.modules, "Quartz", quartz)
    monkeypatch.setitem(sys.modules, "Cocoa", MagicMock())
    pil = MagicMock()
    pil.Image.open.return_value = SimpleNamespace(width=1600, height=1200)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setattr(macos_backend, "image_to_base64", lambda *_a, **_k: (b"", "image/png"))
    b = MacOSBackend(_config(tmp_path))
    fp = WindowFingerprint(pid=7, class_name="Notes", creation_timestamp=None)
    window_id = b.window_registry.mint(42, fp, extra={"bounds": (100, 50, 900, 650)})
    monkeypatch.setattr(b, "_live_fingerprint", lambda _handle: fp)

    result = b._screenshot_window(window_id, quality=75, format="png", include_cursor=False)

    assert result.origin == (100, 50)
    assert result.logical_size == (800, 600)
    assert result.display_scale == 2.0
    image_option = quartz.CGWindowListCreateImage.call_args.args[3]
    assert image_option is quartz.kCGWindowImageBoundsIgnoreFraming
    pil.ImageDraw.Draw.return_value.polygon.assert_not_called()


def test_a_window_capture_draws_the_cursor_when_asked(tmp_path, monkeypatch) -> None:
    """The cursor is in points; the window image is in backing pixels."""
    from autogpt_local_executor.computer_use.backends import macos as macos_backend

    quartz = MagicMock()
    quartz.NSEvent.mouseLocation.return_value = SimpleNamespace(x=300, y=900 - 150)
    quartz.CGWindowListCopyWindowInfo.return_value = []
    monkeypatch.setitem(sys.modules, "Quartz", quartz)
    monkeypatch.setitem(sys.modules, "Cocoa", MagicMock())
    pil = MagicMock()
    pil.Image.open.return_value = SimpleNamespace(width=1600, height=1200)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setattr(macos_backend, "image_to_base64", lambda *_a, **_k: (b"", "image/png"))
    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(b, "_primary_display_size", lambda: (1440, 900))
    fp = WindowFingerprint(pid=7, class_name="Notes", creation_timestamp=None)
    window_id = b.window_registry.mint(42, fp, extra={"bounds": (100, 50, 900, 650)})
    monkeypatch.setattr(b, "_live_fingerprint", lambda _handle: fp)

    b._screenshot_window(window_id, quality=75, format="png", include_cursor=True)

    [arrow] = pil.ImageDraw.Draw.return_value.polygon.call_args.args
    assert arrow[0] == ((300 - 100) * 2, (150 - 50) * 2)


_SIDE_BY_SIDE_RETINA = [
    DisplayMonitor(
        index=0,
        primary=True,
        physical_size=(2880, 1800),
        logical_size=(1440, 900),
        scale=2.0,
        origin=(0, 0),
    ),
    DisplayMonitor(
        index=1,
        physical_size=(2880, 1800),
        logical_size=(1440, 900),
        scale=2.0,
        origin=(1440, 0),
    ),
]


class _ScaledScreenCapture:
    """mss.mss() over the two displays above, in points, returning images
    `factor` times larger: mss versions differ on whether a Retina capture
    comes back at the backing resolution."""

    monitors = [
        {"left": 0, "top": 0, "width": 2880, "height": 900},
        {"left": 0, "top": 0, "width": 1440, "height": 900},
        {"left": 1440, "top": 0, "width": 1440, "height": 900},
    ]

    def __init__(self, factor: int) -> None:
        self.factor = factor

    def __enter__(self) -> _ScaledScreenCapture:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def grab(self, rect: dict[str, int]) -> SimpleNamespace:
        size = (rect["width"] * self.factor, rect["height"] * self.factor)
        return SimpleNamespace(size=size, bgra=b"")


class _RecordingDraw:
    def __init__(self) -> None:
        self.polygons: list[list[tuple[float, float]]] = []

    def polygon(self, points, fill=None) -> None:
        self.polygons.append(list(points))


def _scaled_capture_backend(
    tmp_path, monkeypatch, factor: int
) -> tuple[MacOSBackend, _RecordingDraw]:
    from autogpt_local_executor.computer_use.backends import macos as macos_backend

    draw = _RecordingDraw()
    pil = SimpleNamespace(
        Image=SimpleNamespace(
            frombytes=lambda _mode, size, *_a: SimpleNamespace(width=size[0], height=size[1])
        ),
        ImageDraw=SimpleNamespace(Draw=lambda _img: draw),
    )
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setitem(
        sys.modules, "mss", SimpleNamespace(mss=lambda: _ScaledScreenCapture(factor))
    )
    monkeypatch.setattr(macos_backend, "image_to_base64", lambda *_a, **_k: (b"", "image/png"))
    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: _SIDE_BY_SIDE_RETINA)
    monkeypatch.setattr(b, "_primary_display_size", lambda: (1440, 900))
    return b, draw


def _cursor_at(monkeypatch, x: float, y_from_top: float) -> None:
    """NSEvent.mouseLocation is in points with a bottom-left origin."""
    location = SimpleNamespace(x=x, y=900 - y_from_top)
    quartz = SimpleNamespace(NSEvent=SimpleNamespace(mouseLocation=lambda: location))
    monkeypatch.setitem(sys.modules, "Quartz", quartz)


@pytest.mark.parametrize("factor", [1, 2])
@pytest.mark.parametrize(
    "region",
    [None, (1500, 100, 1900, 400)],
    ids=["display", "region"],
)
def test_a_display_capture_reports_the_scale_its_image_came_back_at(
    tmp_path, monkeypatch, factor, region
) -> None:
    """Quartz says 2.0 either way; only the image says which one mss produced,
    and agents divide image points by display_scale before clicking."""
    b, _draw = _scaled_capture_backend(tmp_path, monkeypatch, factor)

    result = b.screenshot(monitor=1, region=region, include_cursor=False)

    assert result.display_scale == float(factor)


@pytest.mark.parametrize("factor", [1, 2])
def test_the_cursor_is_drawn_where_it_is_in_the_image(tmp_path, monkeypatch, factor) -> None:
    """The cursor's location is in points, the image may be in backing pixels."""
    b, draw = _scaled_capture_backend(tmp_path, monkeypatch, factor)
    _cursor_at(monkeypatch, 1600, 150)

    b.screenshot(monitor=1, region=(1500, 100, 1900, 400), include_cursor=True)

    [arrow] = draw.polygons
    assert arrow[0] == (100 * factor, 50 * factor)
    assert arrow[3] == (100 * factor + 4 * factor, 50 * factor + 12 * factor)


def test_cursor_position_finds_the_display_in_points_on_retina(tmp_path, monkeypatch) -> None:
    """In backing pixels display 0 would reach x=2880 and swallow display 1."""
    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: _SIDE_BY_SIDE_RETINA)
    monkeypatch.setattr(b, "_primary_display_size", lambda: (1440, 900))
    _cursor_at(monkeypatch, 2000, 100)

    assert b.cursor_position() == (2000, 100, 1)


def test_the_quartz_display_mode_names_exist() -> None:
    """display_info's Retina path and the cursor lookups only run on a Mac;
    check the names they import there."""
    quartz = pytest.importorskip("Quartz")
    for name in (
        "CGDisplayCopyDisplayMode",
        "CGDisplayModeGetPixelHeight",
        "CGDisplayModeGetPixelWidth",
        "NSEvent",
        "kCGWindowImageBoundsIgnoreFraming",
    ):
        assert hasattr(quartz, name), name


@pytest.mark.parametrize(
    ("content", "types", "read"),
    [
        (None, (), None),
        ("user text", ("public.utf8-plain-text",), (None, None)),
    ],
    ids=["empty", "unreadable"],
)
def test_a_clipboard_with_no_text_to_put_back_is_left_alone_and_the_text_typed(
    tmp_path, monkeypatch, content, types, read
) -> None:
    """With nothing to restore, a paste would either leave the pasted text
    for other apps or clear it before a slow app has pasted; and a clipboard
    whose text can't be read would be lost. Typing leaves it as it was."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_pasteboard(b, monkeypatch, content=content, types=types)
    if read is not None:
        monkeypatch.setattr(b, "_read_pasteboard_text", lambda: read)
    text = "x" * PASTE_THRESHOLD_CHARS

    b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    assert clipboard["writes"] == []
    assert clipboard["content"] == content
    pyautogui.write.assert_called_once_with(text, interval=0.02)
    pyautogui.hotkey.assert_not_called()


def test_the_clipboard_is_not_restored_over_a_copy_counted_as_ours(tmp_path, monkeypatch) -> None:
    """A copy that lands between our write and our read of the change count
    is counted as ours; the pasteboard no longer holding our text gives it away."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    monkeypatch.setitem(sys.modules, "pyautogui", MagicMock())
    clipboard = _fake_pasteboard(b, monkeypatch, content="user text")

    def write_then_someone_copies(text: str) -> int:
        clipboard["writes"].append(text)
        clipboard["content"] = "copied in the gap"
        clipboard["seq"] += 2
        return clipboard["seq"]

    monkeypatch.setattr(b, "_write_pasteboard_text", write_then_someone_copies)

    b.input_action("type", text="x" * PASTE_THRESHOLD_CHARS, paste=True, preserve_clipboard=True)

    assert clipboard["writes"] == ["x" * PASTE_THRESHOLD_CHARS]
    assert clipboard["content"] == "copied in the gap"


def test_the_clipboard_is_restored_when_the_paste_hotkey_raises(tmp_path, monkeypatch) -> None:
    """pyautogui's fail-safe raises before sending any key, after the text is
    already on the pasteboard."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    pyautogui = MagicMock()
    pyautogui.hotkey.side_effect = RuntimeError("fail-safe")
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_pasteboard(b, monkeypatch, content="user text")

    with pytest.raises(RuntimeError, match="fail-safe"):
        b.input_action(
            "type", text="x" * PASTE_THRESHOLD_CHARS, paste=True, preserve_clipboard=True
        )

    assert clipboard["content"] == "user text"


def _changing_pasteboard(b, monkeypatch, settles_after: int) -> dict:
    """A pasteboard another app writes to between the first reads."""
    state = {"reads": 0, "seq": 1, "content": "first"}

    def read() -> tuple[str, int]:
        state["reads"] += 1
        if state["reads"] <= settles_after:
            state["seq"] += 1
            state["content"] = f"copy {state['seq']}"
        return state["content"], state["seq"]

    monkeypatch.setattr(b, "_read_pasteboard_text", read)
    monkeypatch.setattr(b, "_read_pasteboard_types", lambda: ["public.utf8-plain-text"])
    return state


def test_a_clipboard_read_refuses_a_pasteboard_that_keeps_changing(tmp_path, monkeypatch) -> None:
    """The text and the types came from separate queries, so a concealed
    value swapped in between them passed the marker check."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    _changing_pasteboard(b, monkeypatch, settles_after=100)

    with pytest.raises(ClipboardConcealedError):
        b.clipboard_read()


def test_a_clipboard_read_accepts_a_pasteboard_once_it_settles(tmp_path, monkeypatch) -> None:
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    state = _changing_pasteboard(b, monkeypatch, settles_after=1)

    assert b.clipboard_read().content == state["content"]


def test_a_window_capture_uses_the_live_window_frame_and_its_display(tmp_path, monkeypatch) -> None:
    """Bounds saved at window_list time go stale when the window moves, and
    every window reported display 0."""
    from autogpt_local_executor.computer_use.backends import macos as macos_backend

    quartz = MagicMock()
    quartz.CGWindowListCopyWindowInfo.return_value = [
        {"kCGWindowBounds": {"X": 1600, "Y": 100, "Width": 400, "Height": 300}}
    ]
    monkeypatch.setitem(sys.modules, "Quartz", quartz)
    monkeypatch.setitem(sys.modules, "Cocoa", MagicMock())
    pil = MagicMock()
    pil.Image.open.return_value = SimpleNamespace(width=800, height=600)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    monkeypatch.setattr(macos_backend, "image_to_base64", lambda *_a, **_k: (b"", "image/png"))
    b = MacOSBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: _SIDE_BY_SIDE_RETINA)
    fp = WindowFingerprint(pid=7, class_name="Notes", creation_timestamp=None)
    window_id = b.window_registry.mint(42, fp, extra={"bounds": (0, 0, 800, 600), "monitor": 0})
    monkeypatch.setattr(b, "_live_fingerprint", lambda _handle: fp)

    result = b._screenshot_window(window_id, quality=75, format="png", include_cursor=False)

    assert result.origin == (1600, 100)
    assert result.logical_size == (400, 300)
    assert result.display_scale == 2.0
    assert (result.monitor, result.display_id) == (1, 1)


def test_typing_refuses_text_it_cannot_type_instead_of_dropping_it(tmp_path, monkeypatch) -> None:
    """pyautogui types only ASCII; anything else was skipped without a word."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_ax_trusted", lambda: True, raising=False)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)

    with pytest.raises(FeatureNotSupportedError, match="paste"):
        b.input_action("type", text="Grüße, 世界")

    pyautogui.write.assert_not_called()


def test_short_text_that_cannot_be_typed_is_pasted_when_asked(tmp_path, monkeypatch) -> None:
    """Under the 200-character threshold the text was always typed, so
    "café" could be neither typed nor pasted."""
    b = MacOSBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_ax_trusted", lambda: True)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_pasteboard(b, monkeypatch, content="user text")

    b.input_action("type", text="café", paste=True)

    assert clipboard["writes"] == ["café"]
    pyautogui.hotkey.assert_called_once_with("command", "v")
    pyautogui.write.assert_not_called()
