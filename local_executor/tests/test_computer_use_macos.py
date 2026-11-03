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
from unittest.mock import MagicMock

import pytest

from autogpt_local_executor.computer_use.backends.macos import (
    CONCEALED_PB_TYPES,
    MacOSBackend,
)
from autogpt_local_executor.computer_use.errors import (
    ClipboardConcealedError,
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
    fake_pg = MagicMock()
    fake_pg.FAILSAFE = True
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)

    big = "x" * 250
    b.input_action("type", text=big, paste=True, preserve_clipboard=True)
    # After the paste sequence: snapshot at seq=10, then we write
    # (seq=11), then we restore to PREVIOUS (seq=12). The clipboard
    # content should be PREVIOUS again.
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
