"""Windows backend tests — all native APIs mocked so they run anywhere."""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

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
    monkeypatch.setattr(b, "_clipboard_formats", lambda: {13, 0x0200})
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
    monkeypatch.setattr(b, "_clipboard_formats", lambda: {13})
    b.clipboard_write(content="hello")
    res = b.clipboard_read()
    assert res.content == "hello"


def test_clipboard_writeback_only_refuses_foreign(tmp_path, monkeypatch) -> None:
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: ("user_text", 99))
    monkeypatch.setattr(b, "_clipboard_formats", lambda: {13})
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
    monkeypatch.setattr(
        b,
        "_clipboard_formats",
        lambda: set() if state["content"] is None else ({13, 0x0200} if private else {13}),
    )
    return state


def test_a_clipboard_restored_after_a_paste_stays_unreadable(tmp_path, monkeypatch) -> None:
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_clipboard(b, monkeypatch, content="user secret")
    text = "x" * PASTE_THRESHOLD_CHARS

    b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    pyautogui.hotkey.assert_called_once_with("ctrl", "v")
    assert clipboard["writes"] == [text, "user secret"]
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


@pytest.mark.parametrize("fails", [False, True], ids=["writes", "write-fails"])
def test_a_clipboard_write_opens_the_clipboard_with_a_window_of_its_own(
    tmp_path, monkeypatch, fails: bool
) -> None:
    """Opened with no window (NULL), EmptyClipboard leaves the clipboard with
    no owner, and Microsoft documents that SetClipboardData then fails: every
    write and paste would clear the clipboard and then fail."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    clipboard = MagicMock()
    if fails:
        clipboard.SetClipboardData.side_effect = OSError("SetClipboardData failed")
    win32gui = MagicMock()
    win32gui.CreateWindowEx.return_value = 0x5150
    calls = MagicMock()
    calls.attach_mock(win32gui.CreateWindowEx, "create")
    calls.attach_mock(clipboard.OpenClipboard, "open")
    calls.attach_mock(clipboard.EmptyClipboard, "empty")
    calls.attach_mock(clipboard.SetClipboardData, "set")
    calls.attach_mock(clipboard.CloseClipboard, "close")
    calls.attach_mock(win32gui.DestroyWindow, "destroy")
    monkeypatch.setitem(sys.modules, "win32clipboard", clipboard)
    monkeypatch.setitem(sys.modules, "win32con", MagicMock(CF_UNICODETEXT=13, HWND_MESSAGE=-3))
    monkeypatch.setitem(sys.modules, "win32gui", win32gui)

    if fails:
        with pytest.raises(ClipboardWriteError):
            b.clipboard_write(content="hello")
    else:
        b.clipboard_write(content="hello")

    assert [name for name, _args, _kwargs in calls.mock_calls] == [
        "create",
        "open",
        "empty",
        "set",
        "close",
        "destroy",
    ]
    clipboard.OpenClipboard.assert_called_once_with(0x5150)
    # A message-only window: never shown, and never among the listed windows.
    assert win32gui.CreateWindowEx.call_args.args[8] == -3
    win32gui.DestroyWindow.assert_called_once_with(0x5150)


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


def _capturable_window(tmp_path, monkeypatch, *, rect=(0, 0, 100, 50)) -> SimpleNamespace:
    """A responding window that _screenshot_window captures, with every native
    call faked; a test changes one fake to make a step fail."""
    import ctypes

    from autogpt_local_executor.computer_use.backends import windows as windows_backend

    b = WindowsBackend(_config(tmp_path))
    fp = WindowFingerprint(pid=7, class_name="Notepad", creation_timestamp=None)
    window_id = b.window_registry.mint(0x1234, fp, extra={"monitor": 0})
    monkeypatch.setattr(b, "_live_fingerprint", lambda _hwnd: fp)
    width, height = rect[2] - rect[0], rect[3] - rect[1]
    win32gui = MagicMock()
    win32gui.IsWindow.return_value = True
    win32gui.GetWindowRect.return_value = rect
    monkeypatch.setitem(sys.modules, "win32gui", win32gui)
    win32ui = MagicMock()
    win32ui.CreateBitmap.return_value.GetInfo.return_value = {"bmWidth": width, "bmHeight": height}
    monkeypatch.setitem(sys.modules, "win32ui", win32ui)
    monkeypatch.setitem(sys.modules, "mss", MagicMock())
    pil = MagicMock()
    pil.Image.frombuffer.return_value = SimpleNamespace(width=width, height=height)
    monkeypatch.setitem(sys.modules, "PIL", pil)
    windll = MagicMock()
    windll.user32.IsHungAppWindow.return_value = 0
    windll.user32.PrintWindow.return_value = 1
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)
    monkeypatch.setattr(windows_backend, "image_to_base64", lambda *_a, **_k: (b"", "image/png"))
    return SimpleNamespace(
        backend=b,
        window_id=window_id,
        hwnd=0x1234,
        win32gui=win32gui,
        win32ui=win32ui,
        user32=windll.user32,
    )


def test_a_window_printwindow_cannot_capture_is_an_error(tmp_path, monkeypatch) -> None:
    """PrintWindow returns 0 when the window did not render; the bitmap is
    then blank, so it must not be sent as a screenshot. Everything is still
    freed, the bitmap only after the DC it is selected into: GDI does not
    delete a bitmap that a DC still holds. The window's DC came from
    GetWindowDC, so ReleaseDC alone frees it; DeleteDC is only for DCs
    made with CreateDC or CreateCompatibleDC."""
    window = _capturable_window(tmp_path, monkeypatch)
    window.user32.PrintWindow.return_value = 0
    mfc_dc = window.win32ui.CreateDCFromHandle.return_value
    save_dc = mfc_dc.CreateCompatibleDC.return_value
    freed = MagicMock()
    freed.attach_mock(save_dc.SelectObject, "select")
    freed.attach_mock(save_dc.DeleteDC, "delete_save_dc")
    freed.attach_mock(mfc_dc.DeleteDC, "delete_mfc_dc")
    freed.attach_mock(window.win32gui.DeleteObject, "delete_object")
    freed.attach_mock(window.win32gui.ReleaseDC, "release_dc")

    with pytest.raises(FeatureNotSupportedError, match="could not capture"):
        window.backend._screenshot_window(window.window_id, quality=75, format="jpeg")

    bitmap = window.win32ui.CreateBitmap.return_value.GetHandle.return_value
    hwnd_dc = window.win32gui.GetWindowDC.return_value
    # The DC's own bitmap goes back in before anything is freed.
    assert freed.mock_calls == [
        call.select(window.win32ui.CreateBitmap.return_value),
        call.select(save_dc.SelectObject.return_value),
        call.delete_save_dc(),
        call.delete_object(bitmap),
        call.release_dc(window.hwnd, hwnd_dc),
    ]


def test_a_release_that_fails_still_frees_the_rest(tmp_path, monkeypatch) -> None:
    """A release that raised skipped every release after it, leaking the
    bitmap and the window's DC."""
    window = _capturable_window(tmp_path, monkeypatch)
    save_dc = window.win32ui.CreateDCFromHandle.return_value.CreateCompatibleDC.return_value
    save_dc.DeleteDC.side_effect = RuntimeError("DeleteDC failed")

    window.backend._screenshot_window(window.window_id, quality=75, format="jpeg")

    bitmap = window.win32ui.CreateBitmap.return_value.GetHandle.return_value
    window.win32gui.DeleteObject.assert_called_once_with(bitmap)
    window.win32gui.ReleaseDC.assert_called_once_with(
        window.hwnd, window.win32gui.GetWindowDC.return_value
    )


def test_a_window_that_is_not_responding_is_not_captured(tmp_path, monkeypatch) -> None:
    """PrintWindow asks the window to paint itself and waits for it, so a hung
    window held the capture's thread, and its request slot, for good."""
    window = _capturable_window(tmp_path, monkeypatch)
    window.user32.IsHungAppWindow.return_value = 1

    with pytest.raises(FeatureNotSupportedError, match="not responding"):
        window.backend._screenshot_window(window.window_id, quality=75, format="jpeg")

    window.user32.IsHungAppWindow.assert_called_once_with(window.hwnd)
    window.user32.PrintWindow.assert_not_called()
    window.win32gui.GetWindowDC.assert_not_called()


@pytest.mark.parametrize(
    "formats", [None, {8}, {13, 8}], ids=["unreadable", "image", "text-and-image"]
)
def test_preserving_a_clipboard_a_text_restore_cannot_rebuild_types_instead(
    tmp_path, monkeypatch, formats
) -> None:
    """The restore puts back text only, so pasting over an image (CF_DIB), even
    one with text beside it, or a clipboard whose formats can't be read would
    lose it."""
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
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_clipboard(b, monkeypatch, content="user text")

    def write(text: str) -> int:
        clipboard["content"] = text
        clipboard["seq"] += bump
        clipboard["writes"].append(text)
        return clipboard["seq"]

    monkeypatch.setattr(b, "_write_clipboard_text", write)
    text = "x" * PASTE_THRESHOLD_CHARS

    b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    pyautogui.hotkey.assert_called_once_with("ctrl", "v")
    assert clipboard["writes"][:1] == [text]
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
    text = "x" * PASTE_THRESHOLD_CHARS

    b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    pyautogui.hotkey.assert_called_once_with("ctrl", "v")
    assert clipboard["writes"] == [text]
    assert clipboard["content"] == "copied during the paste"


@pytest.mark.parametrize(
    ("content", "formats", "read"),
    [(None, set(), None), ("user text", {13}, (None, None))],
    ids=["empty", "unreadable"],
)
def test_a_clipboard_with_no_text_to_put_back_is_left_alone_and_the_text_typed(
    tmp_path, monkeypatch, content, formats, read
) -> None:
    """With nothing to restore, a paste would either leave the pasted text
    for other apps or clear it before a slow app has pasted; and a clipboard
    whose text can't be read would be lost. Typing leaves it as it was."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_clipboard(b, monkeypatch, content=content)
    monkeypatch.setattr(b, "_clipboard_formats", lambda: formats)
    if read is not None:
        monkeypatch.setattr(b, "_read_clipboard_text", lambda: read)
    text = "x" * PASTE_THRESHOLD_CHARS

    b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    assert clipboard["writes"] == []
    assert clipboard["content"] == content
    pyautogui.write.assert_called_once_with(text, interval=0.02)
    pyautogui.hotkey.assert_not_called()


def test_the_clipboard_is_not_restored_over_a_copy_counted_as_ours(tmp_path, monkeypatch) -> None:
    """The sequence number is read after CloseClipboard, so a copy landing in
    that gap is counted as ours; the clipboard no longer holding our text
    gives it away."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_clipboard(b, monkeypatch, content="user text")

    def write_then_someone_copies(text: str) -> int:
        clipboard["writes"].append(text)
        clipboard["content"] = "copied in the gap"
        clipboard["seq"] += 3
        return clipboard["seq"]

    monkeypatch.setattr(b, "_write_clipboard_text", write_then_someone_copies)

    b.input_action("type", text="x" * PASTE_THRESHOLD_CHARS, paste=True, preserve_clipboard=True)

    pyautogui.hotkey.assert_called_once_with("ctrl", "v")
    assert clipboard["writes"] == ["x" * PASTE_THRESHOLD_CHARS]
    assert clipboard["content"] == "copied in the gap"


@pytest.mark.parametrize("include_cursor", [True, False])
def test_a_window_capture_draws_the_cursor_only_when_asked(
    tmp_path, monkeypatch, include_cursor
) -> None:
    window = _capturable_window(tmp_path, monkeypatch, rect=(40, 30, 140, 80))
    drawn: list[tuple[int, int]] = []
    monkeypatch.setattr(
        window.backend, "_composite_cursor", lambda _img, origin: drawn.append(origin)
    )

    window.backend.screenshot(window_id=window.window_id, include_cursor=include_cursor)

    assert drawn == ([(40, 30)] if include_cursor else [])


def test_the_clipboard_is_restored_when_the_paste_hotkey_raises(tmp_path, monkeypatch) -> None:
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    pyautogui = MagicMock()
    pyautogui.hotkey.side_effect = RuntimeError("fail-safe")
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_clipboard(b, monkeypatch, content="user text")
    text = "x" * PASTE_THRESHOLD_CHARS

    with pytest.raises(RuntimeError, match="fail-safe"):
        b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    pyautogui.hotkey.assert_called_once_with("ctrl", "v")
    assert clipboard["writes"] == [text, "user text"]
    assert clipboard["content"] == "user text"


def test_a_window_capture_releases_what_it_acquired_when_a_step_fails(
    tmp_path, monkeypatch
) -> None:
    """The DCs were acquired outside the try, so a failing bitmap leaked them."""
    window = _capturable_window(tmp_path, monkeypatch)
    bitmap = window.win32ui.CreateBitmap.return_value
    bitmap.CreateCompatibleBitmap.side_effect = RuntimeError("no GDI")

    with pytest.raises(RuntimeError, match="no GDI"):
        window.backend._screenshot_window(window.window_id, quality=75, format="jpeg")

    mfc_dc = window.win32ui.CreateDCFromHandle.return_value
    mfc_dc.CreateCompatibleDC.return_value.DeleteDC.assert_called_once()
    # The window's DC goes back through ReleaseDC alone; mfc_dc only wraps it.
    mfc_dc.DeleteDC.assert_not_called()
    window.win32gui.ReleaseDC.assert_called_once_with(
        window.hwnd, window.win32gui.GetWindowDC.return_value
    )
    window.win32gui.DeleteObject.assert_not_called()


def test_a_clipboard_read_refuses_a_private_format_that_lands_between_reads(
    tmp_path, monkeypatch
) -> None:
    """The CF_PRIVATE check and the text read were separate clipboard
    sessions, so a password-manager copy between them was returned."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    state = {"seq": 1, "content": "public", "private": False, "reads": 0}

    def read() -> tuple[str, int]:
        state["reads"] += 1
        if state["reads"] == 1:
            # The password manager copies right after this first read.
            state["seq"] += 2
            state["content"], state["private"] = "secret", True
        return state["content"], state["seq"]

    monkeypatch.setattr(b, "_read_clipboard_text", read)
    monkeypatch.setattr(b, "_clipboard_formats", lambda: {13, 0x0200} if state["private"] else {13})

    with pytest.raises(ClipboardConcealedError):
        b.clipboard_read()


def test_typing_refuses_text_it_cannot_type_instead_of_dropping_it(tmp_path, monkeypatch) -> None:
    """pyautogui types only ASCII; anything else was skipped without a word."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_ax_trusted", lambda: True, raising=False)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)

    with pytest.raises(FeatureNotSupportedError, match="paste"):
        b.input_action("type", text="Grüße, 世界")

    pyautogui.write.assert_not_called()


@pytest.mark.parametrize("restore_fails", [False, True], ids=["restored", "restore-fails"])
def test_a_paste_whose_write_fails_puts_the_users_text_back(
    tmp_path, monkeypatch, caplog, restore_fails
) -> None:
    """EmptyClipboard runs before SetClipboardData, so a failed write left the
    clipboard empty and the user's text gone. A restore that fails too is
    logged, and the write's own error is the one reported."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_clipboard(b, monkeypatch, content="user text")

    def write(text: str) -> int:
        clipboard["writes"].append(text)
        # EmptyClipboard goes through, then SetClipboardData fails: for the
        # paste, and for the restore too when asked.
        clipboard["content"] = None
        clipboard["seq"] += 1
        if len(clipboard["writes"]) == 1 or restore_fails:
            raise ClipboardWriteError(
                f"SetClipboardData failed on write {len(clipboard['writes'])}"
            )
        clipboard["content"] = text
        clipboard["seq"] += 1
        return clipboard["seq"]

    monkeypatch.setattr(b, "_write_clipboard_text", write)
    caplog.set_level(logging.WARNING, logger=WindowsBackend.__module__)
    text = "x" * PASTE_THRESHOLD_CHARS

    with pytest.raises(ClipboardWriteError, match="write 1"):
        b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    assert clipboard["writes"] == [text, "user text"]
    assert clipboard["content"] == (None if restore_fails else "user text")
    pyautogui.hotkey.assert_not_called()
    warning = (
        WindowsBackend.__module__,
        logging.WARNING,
        "Could not restore the clipboard after a failed write",
    )
    assert (warning in caplog.record_tuples) is restore_fails


def test_a_clipboard_read_that_cannot_be_read_consistently_says_so(tmp_path, monkeypatch) -> None:
    """A read that fails disagrees with one that works, so readings that never
    match are no proof the clipboard kept changing."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    reads = itertools.cycle([(None, None), ("user text", 4)])
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: next(reads))
    monkeypatch.setattr(b, "_clipboard_formats", lambda: {13})

    with pytest.raises(ClipboardConcealedError, match="could not be read consistently"):
        b.clipboard_read()


def test_a_clipboard_whose_reads_keep_failing_is_refused(tmp_path, monkeypatch) -> None:
    """Two failed reads looked alike, so a foreign read returned "" for a
    clipboard it never read."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: (None, None))
    monkeypatch.setattr(b, "_clipboard_formats", lambda: {13})

    with pytest.raises(ClipboardConcealedError):
        b.clipboard_read()


# ── Scrolling, key names and the window list ───────────────────────────


@pytest.mark.parametrize(
    ("direction", "flag", "delta"),
    [("down", 0x0800, -360), ("up", 0x0800, 360), ("left", 0x1000, -360), ("right", 0x1000, 360)],
)
def test_a_scroll_turns_the_wheel_whole_notches_at_the_point(
    tmp_path, displays_1080p, monkeypatch, direction: str, flag: int, delta: int
) -> None:
    """pyautogui sends its amount as the raw wheel delta, where a notch is
    120, so 3 scrolled 3/120 of a notch. Its hscroll turns the vertical wheel
    on Windows, and it sends the event wherever the cursor already is."""
    import ctypes

    b = WindowsBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    windll = MagicMock()
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)
    events = MagicMock()
    events.attach_mock(pyautogui.moveTo, "move")
    events.attach_mock(windll.user32.mouse_event, "wheel")

    b.input_action("scroll", coordinate=(100, 200), scroll_direction=direction, scroll_amount=3)

    assert events.mock_calls == [call.move(100, 200), call.wheel(flag, 0, 0, delta, 0)]
    pyautogui.scroll.assert_not_called()
    pyautogui.hscroll.assert_not_called()


def test_a_scroll_without_a_point_turns_the_wheel_where_the_cursor_is(
    tmp_path, monkeypatch
) -> None:
    """Without a point, a left or right scroll turned the vertical wheel."""
    import ctypes

    b = WindowsBackend(_config(tmp_path))
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    windll = MagicMock()
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)

    b.input_action("scroll", scroll_direction="left", scroll_amount=2)

    pyautogui.moveTo.assert_not_called()
    windll.user32.mouse_event.assert_called_once_with(0x1000, 0, 0, -240, 0)


@pytest.mark.parametrize("name", ["super", "meta", "Meta"])
def test_super_and_meta_press_the_windows_key(tmp_path, monkeypatch, name: str) -> None:
    """COMPUTER_USE.md makes meta an alias for super, the Windows key here.
    Chords passed both names through, and pyautogui has no `meta`: it skips a
    name it doesn't know without a word."""
    b = WindowsBackend(_config(tmp_path))
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)

    b.input_action("key", key=f"{name}+r")
    b.input_action("hold_key", key=f"{name}+r", duration_ms=0)

    pyautogui.hotkey.assert_called_once_with("win", "r")
    assert pyautogui.keyDown.call_args_list == [call("win"), call("r")]


def test_window_list_leaves_out_minimized_windows_unless_asked_and_never_hidden_ones(
    tmp_path, monkeypatch
) -> None:
    """A minimized window keeps WS_VISIBLE, so it was listed by default, and
    include_minimized=True let hidden and untitled helper windows in."""
    windows = {
        1: {"visible": True, "title": "Editor", "iconic": False},
        2: {"visible": True, "title": "Minimized", "iconic": True},
        3: {"visible": False, "title": "Hidden", "iconic": False},
        4: {"visible": True, "title": "", "iconic": False},
    }
    win32gui = MagicMock()
    win32gui.EnumWindows.side_effect = lambda enum, extra: [enum(h, extra) for h in windows]
    win32gui.IsWindowVisible.side_effect = lambda h: windows[h]["visible"]
    win32gui.GetWindowText.side_effect = lambda h: windows[h]["title"]
    win32gui.IsIconic.side_effect = lambda h: windows[h]["iconic"]
    win32gui.GetClassName.return_value = "Window"
    win32gui.GetWindowRect.return_value = (0, 0, 100, 100)
    win32gui.GetForegroundWindow.return_value = 1
    win32process = MagicMock()
    win32process.GetWindowThreadProcessId.return_value = (1, 4242)
    monkeypatch.setitem(sys.modules, "win32gui", win32gui)
    monkeypatch.setitem(sys.modules, "win32process", win32process)
    b = WindowsBackend(_config(tmp_path))

    assert [w.title for w in b.window_list()] == ["Editor"]
    assert [w.title for w in b.window_list(include_minimized=True)] == ["Editor", "Minimized"]


def test_a_failed_write_leaves_a_copy_made_since_alone(tmp_path, monkeypatch) -> None:
    """The recovery write put the user's old text back without looking, so a
    copy made after the failed write was replaced with it."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True))
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    clipboard = _fake_clipboard(b, monkeypatch, content="user text")
    text = "x" * PASTE_THRESHOLD_CHARS

    def write(content: str) -> int:
        clipboard["writes"].append(content)
        if len(clipboard["writes"]) == 1:
            # SetClipboardData fails after EmptyClipboard, and another app
            # copies before the shim looks again.
            clipboard["content"] = "copied meanwhile"
            clipboard["seq"] += 2
            raise ClipboardWriteError("SetClipboardData failed")
        clipboard["content"] = content
        clipboard["seq"] += 1
        return clipboard["seq"]

    monkeypatch.setattr(b, "_write_clipboard_text", write)

    with pytest.raises(ClipboardWriteError):
        b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    assert clipboard["writes"] == [text]
    assert clipboard["content"] == "copied meanwhile"


@pytest.mark.parametrize(
    "name", ["ExcludeClipboardContentFromMonitorProcessing", "Clipboard Viewer Ignore"]
)
def test_a_copy_kept_from_clipboard_monitors_is_concealed(tmp_path, monkeypatch, name) -> None:
    """Password managers mark a copy with a registered format, not CF_PRIVATE,
    so a foreign read returned the password."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_clipboard_formats", lambda: {13, 0xC0F1})
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: ("hunter2", 5))
    win32clipboard = MagicMock()
    win32clipboard.GetClipboardFormatName.side_effect = lambda fmt: {0xC0F1: name}[fmt]
    monkeypatch.setitem(sys.modules, "win32clipboard", win32clipboard)

    with pytest.raises(ClipboardConcealedError) as ei:
        b.clipboard_read()

    assert ei.value.details["marker"] == name


def test_a_copy_with_another_registered_format_is_read(tmp_path, monkeypatch) -> None:
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_clipboard_formats", lambda: {13, 0xC0F2})
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: ("<b>hi</b>", 5))
    win32clipboard = MagicMock()
    win32clipboard.GetClipboardFormatName.side_effect = lambda fmt: {0xC0F2: "HTML Format"}[fmt]
    monkeypatch.setitem(sys.modules, "win32clipboard", win32clipboard)

    assert b.clipboard_read().content == "<b>hi</b>"


def test_a_clipboard_held_open_for_a_moment_is_read_once_it_is_free(tmp_path, monkeypatch) -> None:
    """Another app holding the clipboard open for a few milliseconds failed all
    three back-to-back reads, and the copy was reported as concealed."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    slept: list[float] = []
    monkeypatch.setattr("time.sleep", slept.append)
    # Held open until the shim has waited a moment.
    monkeypatch.setattr(
        b, "_read_clipboard_text", lambda: ("user text", 7) if slept else (None, None)
    )
    monkeypatch.setattr(b, "_clipboard_formats", lambda: {13})

    assert b.clipboard_read().content == "user text"
    assert slept


def test_an_unknown_key_name_is_refused_before_any_key_is_pressed(tmp_path, monkeypatch) -> None:
    """pyautogui skips a name it doesn't know, so ctrl+bogus pressed only
    Ctrl, and was acknowledged."""
    b = WindowsBackend(_config(tmp_path))
    pyautogui = MagicMock()
    pyautogui.KEY_NAMES = ["ctrl", "s", "win", "r"]
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)

    with pytest.raises(FeatureNotSupportedError, match="bogus"):
        b.input_action("key", key="ctrl+bogus")
    with pytest.raises(FeatureNotSupportedError, match="bogus"):
        b.input_action("hold_key", key="ctrl+bogus", duration_ms=0)
    pyautogui.hotkey.assert_not_called()
    pyautogui.keyDown.assert_not_called()

    b.input_action("key", key="ctrl+S")
    pyautogui.hotkey.assert_called_once_with("ctrl", "s")


def test_a_formats_read_that_fails_for_a_moment_is_read_again(tmp_path, monkeypatch) -> None:
    """Only disagreeing text or sequence reads were retried. A formats read
    that failed while another app held the clipboard counted as unreadable,
    and the copy was reported as concealed."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    formats = iter([None, {13}])
    monkeypatch.setattr(b, "_clipboard_formats", lambda: next(formats))
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: ("user text", 7))

    assert b.clipboard_read().content == "user text"


@pytest.mark.parametrize(
    "name", ["clipboard viewer ignore", "EXCLUDECLIPBOARDCONTENTFROMMONITORPROCESSING"]
)
def test_a_concealment_marker_counts_whatever_its_case(tmp_path, monkeypatch, name) -> None:
    """Windows matches format names without regard to case and returns the
    spelling the name was first registered with, so an exact match missed a
    marker another app had registered in lowercase."""
    b = WindowsBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_clipboard_formats", lambda: {13, 0xC0F1})
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: ("hunter2", 5))
    win32clipboard = MagicMock()
    win32clipboard.GetClipboardFormatName.side_effect = lambda fmt: {0xC0F1: name}[fmt]
    monkeypatch.setitem(sys.modules, "win32clipboard", win32clipboard)

    with pytest.raises(ClipboardConcealedError):
        b.clipboard_read()


@pytest.mark.parametrize("key", ["ctrl++s", "ctrl+", "+s", "ctrl+ +s"])
def test_a_chord_with_an_empty_part_is_refused(tmp_path, monkeypatch, key: str) -> None:
    """Empty parts were dropped, so ctrl++s pressed ctrl+s and was
    acknowledged."""
    b = WindowsBackend(_config(tmp_path))
    pyautogui = MagicMock()
    pyautogui.KEY_NAMES = ["ctrl", "s"]
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)

    with pytest.raises(FeatureNotSupportedError, match="empty"):
        b.input_action("key", key=key)
    pyautogui.hotkey.assert_not_called()


def test_an_app_lists_its_executable_from_a_limited_handle(tmp_path, monkeypatch) -> None:
    """GetModuleFileNameEx needs PROCESS_VM_READ as well, which the handle
    didn't have, so every app was listed without its executable path."""
    import ctypes

    windll = MagicMock()
    windll.kernel32.OpenProcess.return_value = 0x1234

    def query(handle, flags, buffer, size):
        buffer.value = "C:\\Program Files\\Editor\\editor.exe"
        return 1

    windll.kernel32.QueryFullProcessImageNameW.side_effect = query
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)
    b = WindowsBackend(_config(tmp_path))

    assert b._exe_for_pid(4242) == "C:\\Program Files\\Editor\\editor.exe"
    windll.kernel32.OpenProcess.assert_called_once_with(0x1000, False, 4242)
    windll.kernel32.CloseHandle.assert_called_once_with(0x1234)


@pytest.mark.parametrize("minimized", [True, False])
def test_focusing_a_minimized_window_restores_it_first(tmp_path, monkeypatch, minimized) -> None:
    """SetForegroundWindow on a minimized window leaves it minimized."""
    b = WindowsBackend(_config(tmp_path))
    fp = WindowFingerprint(pid=99, class_name="A", creation_timestamp=None)
    wid = b.window_registry.mint(0x12345, fp, extra={"bounds": (0, 0, 100, 100)})
    monkeypatch.setattr(b, "_live_fingerprint", lambda _: fp)
    win32gui = MagicMock()
    win32gui.IsIconic.return_value = minimized
    calls = MagicMock()
    calls.attach_mock(win32gui.ShowWindow, "show")
    calls.attach_mock(win32gui.SetForegroundWindow, "foreground")
    monkeypatch.setitem(sys.modules, "win32gui", win32gui)

    b.window_focus(wid)

    expected = [call.show(0x12345, 9)] if minimized else []
    assert calls.mock_calls == [*expected, call.foreground(0x12345)]


def test_window_list_refuses_an_app_filter_it_cannot_apply(tmp_path, monkeypatch) -> None:
    """Windows have no bundle ID here, so app_bundle_id was ignored and every
    window came back, from unrelated apps too. The refusal comes first, so
    window IDs from an earlier listing stay valid."""
    b = WindowsBackend(_config(tmp_path))
    fp = WindowFingerprint(pid=7, class_name="Notepad", creation_timestamp=None)
    earlier = b.window_registry.mint(0x1234, fp, extra={"monitor": 0})
    monkeypatch.setitem(sys.modules, "win32gui", MagicMock())
    monkeypatch.setitem(sys.modules, "win32process", MagicMock())

    with pytest.raises(FeatureNotSupportedError, match="app_bundle_id"):
        b.window_list(app_bundle_id="com.apple.Safari")

    assert b.window_registry.lookup(earlier) is not None
