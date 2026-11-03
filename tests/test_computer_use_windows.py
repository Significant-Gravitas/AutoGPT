"""Windows backend tests — all native APIs mocked so they run anywhere."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autogpt_local_executor.computer_use.backends._common import PASTE_THRESHOLD_CHARS
from autogpt_local_executor.computer_use.backends.windows import (
    CF_PRIVATEFIRST,
    WindowsBackend,
)
from autogpt_local_executor.computer_use.errors import (
    ClipboardConcealedError,
    ClipboardWriteError,
    FeatureNotSupportedError,
    InputOutOfBoundsError,
    WindowStaleError,
)
from autogpt_local_executor.computer_use.window_registry import WindowFingerprint
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.protocol import DisplayMonitor


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


# ── Capabilities ───────────────────────────────────────────────────────


def test_coarse_features_default(tmp_path: Path) -> None:
    b = WindowsBackend(_config(tmp_path))
    f = b.coarse_features()
    for need in ("screenshot", "input", "windows", "apps", "permissions"):
        assert need in f
    assert "clipboard" not in f


def test_coarse_features_with_clipboard(tmp_path: Path) -> None:
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    assert "clipboard" in b.coarse_features()


# ── Bounds (Q1) ────────────────────────────────────────────────────────


def test_input_action_raises_on_out_of_bounds(tmp_path, displays_1080p, monkeypatch) -> None:
    b = WindowsBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    with pytest.raises(InputOutOfBoundsError):
        b.input_action("left_click", coordinate=(10000, 10000))


def test_input_action_in_bounds_calls_pyautogui(tmp_path, displays_1080p, monkeypatch) -> None:
    b = WindowsBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    fake_pg = MagicMock()
    fake_pg.FAILSAFE = True
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)
    b.input_action("left_click", coordinate=(100, 100))
    fake_pg.click.assert_called_once_with(100, 100)


# ── Window registry (Q2) ───────────────────────────────────────────────


def test_window_focus_with_stale_handle_raises(tmp_path, monkeypatch) -> None:
    b = WindowsBackend(_config(tmp_path))
    fp = WindowFingerprint(pid=99, class_name="A", creation_timestamp=None)
    wid = b.window_registry.mint(0x12345, fp, extra={"bounds": (0, 0, 100, 100)})

    fake_win32gui = MagicMock()
    monkeypatch.setitem(sys.modules, "win32gui", fake_win32gui)
    # _live_fingerprint returns None ⇒ window stale.
    monkeypatch.setattr(b, "_live_fingerprint", lambda _: None)
    with pytest.raises(WindowStaleError):
        b.window_focus(wid)


# ── Clipboard (Q3) ─────────────────────────────────────────────────────


def test_clipboard_disabled_returns_feature_not_supported(tmp_path) -> None:
    b = WindowsBackend(_config(tmp_path))
    with pytest.raises(FeatureNotSupportedError):
        b.clipboard_read()


def test_clipboard_cf_private_marker_raises_concealed(tmp_path, monkeypatch) -> None:
    """Q3: CF_PRIVATE format on the clipboard → CLIPBOARD_CONCEALED with
    marker=CF_PRIVATE."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_has_private_format", lambda: True)
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: ("secret", 1))
    with pytest.raises(ClipboardConcealedError) as ei:
        b.clipboard_read()
    assert ei.value.details["marker"] == "CF_PRIVATE"


def test_clipboard_writeback_round_trip(tmp_path, monkeypatch) -> None:
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    state = {"content": None, "seq": 0}

    def _write(content: str) -> int:
        state["content"] = content
        state["seq"] += 1
        return state["seq"]

    monkeypatch.setattr(b, "_write_clipboard_text", _write)
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: (state["content"], state["seq"]))
    monkeypatch.setattr(b, "_has_private_format", lambda: False)
    b.clipboard_write(content="hello")
    res = b.clipboard_read()
    assert res.content == "hello"


def test_clipboard_writeback_only_refuses_foreign(tmp_path, monkeypatch) -> None:
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: ("user_text", 99))
    monkeypatch.setattr(b, "_has_private_format", lambda: False)
    with pytest.raises(ClipboardConcealedError):
        b.clipboard_read()


# ── CF_PRIVATE detection (sanity check on the enum range) ──────────────


def test_cf_private_first_constant() -> None:
    assert CF_PRIVATEFIRST == 0x0200


# ── Permissions ────────────────────────────────────────────────────────


def test_permissions_check_reports_elevation(tmp_path, monkeypatch) -> None:
    b = WindowsBackend(_config(tmp_path))
    monkeypatch.setattr(b, "_is_elevated", lambda: True)
    out = b.permissions_check(["accessibility", "screen_recording"])
    assert out == {"accessibility": "granted", "screen_recording": "granted"}


def test_permissions_check_unknown_when_not_elevated(tmp_path, monkeypatch) -> None:
    b = WindowsBackend(_config(tmp_path))
    monkeypatch.setattr(b, "_is_elevated", lambda: False)
    out = b.permissions_check(["accessibility"])
    assert out == {"accessibility": "unknown"}


# ── App launch ─────────────────────────────────────────────────────────


def test_app_launch_requires_executable_path(tmp_path) -> None:
    b = WindowsBackend(_config(tmp_path))
    with pytest.raises(FeatureNotSupportedError):
        b.app_launch(bundle_id="com.example.app")


# ── Paste with preserve_clipboard (Q3/Q4) ───────────────────────────────


def _fake_clipboard(b: WindowsBackend, monkeypatch, *, content: str, private=False) -> dict:
    state: dict = {"content": content, "seq": 1, "writes": []}

    def _write(text: str) -> int:
        state["content"] = text
        state["seq"] += 1
        state["writes"].append(text)
        return state["seq"]

    monkeypatch.setattr(b, "_write_clipboard_text", _write)
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: (state["content"], state["seq"]))
    # CF_UNICODETEXT (13), plus a CF_PRIVATE format (0x0200) when asked.
    monkeypatch.setattr(b, "_clipboard_formats", lambda: {13, 0x0200} if private else {13})
    return state


def test_a_clipboard_restored_after_a_paste_stays_unreadable(tmp_path, monkeypatch) -> None:
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setitem(sys.modules, "pyautogui", MagicMock())
    clipboard = _fake_clipboard(b, monkeypatch, content="user secret")

    b.input_action("type", text="x" * PASTE_THRESHOLD_CHARS, paste=True, preserve_clipboard=True)

    assert clipboard["content"] == "user secret"
    with pytest.raises(ClipboardConcealedError):
        b.clipboard_read()


def test_a_cf_private_clipboard_is_left_alone_and_the_text_typed(tmp_path, monkeypatch) -> None:
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_clipboard(b, monkeypatch, content="password", private=True)
    text = "x" * PASTE_THRESHOLD_CHARS

    b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    assert clipboard["writes"] == []
    pyautogui.write.assert_called_once_with(text, interval=0.02)
    pyautogui.hotkey.assert_not_called()


def test_a_failed_clipboard_write_is_an_error_and_never_pastes(tmp_path, monkeypatch) -> None:
    """Another window holding the clipboard makes OpenClipboard fail; pasting
    then would paste whatever the user had copied."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = MagicMock()
    clipboard.OpenClipboard.side_effect = OSError("Access is denied.")
    monkeypatch.setitem(sys.modules, "win32clipboard", clipboard)
    monkeypatch.setitem(sys.modules, "win32con", MagicMock())

    with pytest.raises(ClipboardWriteError):
        b.clipboard_write(content="hello")
    with pytest.raises(ClipboardWriteError):
        b.input_action("type", text="x" * PASTE_THRESHOLD_CHARS, paste=True)

    pyautogui.hotkey.assert_not_called()


def test_a_window_printwindow_cannot_capture_is_an_error(tmp_path, monkeypatch) -> None:
    """PrintWindow returns 0 when the window did not render; the bitmap is
    then blank, so it must not be sent as a screenshot."""
    import ctypes

    b = WindowsBackend(_config(tmp_path))
    fp = WindowFingerprint(pid=7, class_name="Notepad", creation_timestamp=None)
    window_id = b.window_registry.mint(0x1234, fp, extra={"monitor": 0})
    monkeypatch.setattr(b, "_live_fingerprint", lambda _hwnd: fp)
    win32gui = MagicMock()
    win32gui.IsWindow.return_value = True
    win32gui.GetWindowRect.return_value = (0, 0, 100, 50)
    monkeypatch.setitem(sys.modules, "win32gui", win32gui)
    monkeypatch.setitem(sys.modules, "win32ui", MagicMock())
    monkeypatch.setitem(sys.modules, "PIL", MagicMock())
    windll = MagicMock()
    windll.user32.PrintWindow.return_value = 0
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)

    with pytest.raises(FeatureNotSupportedError, match="could not capture"):
        b._screenshot_window(window_id, quality=75, format="jpeg")

    win32gui.ReleaseDC.assert_called_once()


@pytest.mark.parametrize("formats", [None, {8}], ids=["unreadable", "image"])
def test_preserving_a_clipboard_a_text_restore_cannot_rebuild_types_instead(
    tmp_path, monkeypatch, formats
) -> None:
    """The restore puts back text only, so pasting over an image (CF_DIB) or a
    clipboard whose formats can't be read would lose it."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_clipboard(b, monkeypatch, content="kept")
    monkeypatch.setattr(b, "_clipboard_formats", lambda: formats)
    text = "x" * PASTE_THRESHOLD_CHARS

    b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    assert clipboard["writes"] == []
    pyautogui.write.assert_called_once_with(text, interval=0.02)


def test_a_clipboard_read_refuses_when_its_formats_cannot_be_read(tmp_path, monkeypatch) -> None:
    """Unreadable formats could be hiding a CF_PRIVATE one."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: ("password", 3))
    monkeypatch.setattr(b, "_clipboard_formats", lambda: None)

    with pytest.raises(ClipboardConcealedError):
        b.clipboard_read()


@pytest.mark.parametrize(("bump", "restored"), [(2, True), (5, True)])
def test_the_clipboard_is_restored_however_far_our_write_moves_the_sequence(
    tmp_path, monkeypatch, bump: int, restored: bool
) -> None:
    """EmptyClipboard and SetClipboardData each advance the sequence number,
    so one write of ours moves it by more than one."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setitem(sys.modules, "pyautogui", MagicMock())
    clipboard = _fake_clipboard(b, monkeypatch, content="user text")

    def write(text: str) -> int:
        clipboard["content"] = text
        clipboard["seq"] += bump
        clipboard["writes"].append(text)
        return clipboard["seq"]

    monkeypatch.setattr(b, "_write_clipboard_text", write)

    b.input_action("type", text="x" * PASTE_THRESHOLD_CHARS, paste=True, preserve_clipboard=True)

    assert (clipboard["content"] == "user text") is restored


def test_the_clipboard_is_not_restored_over_a_copy_made_during_the_paste(
    tmp_path, monkeypatch
) -> None:
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_clipboard(b, monkeypatch, content="user text")

    def user_copies(*_keys: str) -> None:
        clipboard["content"] = "copied during the paste"
        clipboard["seq"] += 1

    pyautogui.hotkey.side_effect = user_copies

    b.input_action("type", text="x" * PASTE_THRESHOLD_CHARS, paste=True, preserve_clipboard=True)

    assert clipboard["content"] == "copied during the paste"
