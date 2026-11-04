"""Linux backend tests — Wayland, X11, wmctrl, xclip all mocked."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autogpt_local_executor.computer_use.backends import linux as linux_backend
from autogpt_local_executor.computer_use.backends._common import PASTE_THRESHOLD_CHARS
from autogpt_local_executor.computer_use.backends.linux import LinuxBackend
from autogpt_local_executor.computer_use.errors import (
    ClipboardConcealedError,
    ClipboardWriteError,
    FeatureNotSupportedError,
    InputOutOfBoundsError,
)
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


@pytest.fixture(autouse=True)
def _x11_session(monkeypatch):
    """Default tests to a 'fake X11' env to avoid Wayland short-circuits."""
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(linux_backend, "_is_wayland", lambda: False)


_LIST_TARGETS = ["xclip", "-selection", "clipboard", "-t", "TARGETS", "-o"]
# What `xclip -t TARGETS -o` lists for text copied in a GTK app.
_GTK_TEXT_TARGETS = [
    "TIMESTAMP",
    "TARGETS",
    "MULTIPLE",
    "SAVE_TARGETS",
    "UTF8_STRING",
    "COMPOUND_TEXT",
    "TEXT",
    "STRING",
    "text/plain;charset=utf-8",
    "text/plain",
]


def _clipboard_targets(monkeypatch, targets: list[str] | None) -> None:
    """Fake `xclip -t TARGETS -o`. With None it fails, as it does when the
    clipboard's owner lists no targets."""

    def check_output(command, **_kwargs):
        assert command == _LIST_TARGETS
        if targets is None:
            raise linux_backend.subprocess.CalledProcessError(1, command)
        return "".join(f"{target}\n" for target in targets).encode()

    monkeypatch.setattr(linux_backend.subprocess, "check_output", check_output)


def _xdg_applications(monkeypatch, tmp_path: Path) -> Path:
    """Point the XDG data directories into tmp_path and return the user's
    applications directory."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "share"))
    monkeypatch.setenv("XDG_DATA_DIRS", str(tmp_path / "system"))
    applications = tmp_path / "share" / "applications"
    applications.mkdir(parents=True)
    return applications


# ── Capabilities ───────────────────────────────────────────────────────


def test_x11_advertises_input_apps(tmp_path) -> None:
    b = LinuxBackend(_config(tmp_path))
    f = b.coarse_features()
    assert "screenshot" in f
    assert "input" in f
    assert "apps" in f


def test_wayland_advertises_only_screenshot(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setattr(linux_backend, "_is_wayland", lambda: True)
    b = LinuxBackend(_config(tmp_path))
    assert b.coarse_features() == ["screenshot"]


# ── Wayland input is blocked (FEATURE_NOT_SUPPORTED) ───────────────────


def test_wayland_input_returns_feature_not_supported(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(linux_backend, "_is_wayland", lambda: True)
    b = LinuxBackend(_config(tmp_path))
    with pytest.raises(FeatureNotSupportedError) as ei:
        b.input_action("left_click", coordinate=(100, 100))
    assert "Wayland" in ei.value.details["reason"]


def test_x11_input_works(tmp_path, displays_1080p, monkeypatch) -> None:
    b = LinuxBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    fake_pg = MagicMock()
    fake_pg.FAILSAFE = True
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)
    b.input_action("left_click", coordinate=(100, 100))
    fake_pg.click.assert_called_once_with(100, 100)


def test_x11_input_out_of_bounds(tmp_path, displays_1080p, monkeypatch) -> None:
    b = LinuxBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    with pytest.raises(InputOutOfBoundsError):
        b.input_action("left_click", coordinate=(10000, 10000))


def test_triple_click_emits_three_clicks(tmp_path, displays_1080p, monkeypatch) -> None:
    """X11 has no native triple — we synthesize."""
    b = LinuxBackend(_config(tmp_path))
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    fake_pg = MagicMock()
    fake_pg.FAILSAFE = True
    monkeypatch.setitem(sys.modules, "pyautogui", fake_pg)
    b.input_action("triple_click", coordinate=(50, 50))
    assert fake_pg.click.call_count == 3


# ── Window list (wmctrl gating) ────────────────────────────────────────


def test_window_list_without_wmctrl(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: None)
    b = LinuxBackend(_config(tmp_path))
    with pytest.raises(FeatureNotSupportedError) as ei:
        b.window_list()
    assert "wmctrl" in ei.value.details["reason"]


def test_window_list_with_wmctrl_parses_output(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path))

    sample = (
        "0x01000003  0 12345 100 100 800 600 host  Terminal\n"
        "0x02000004  0 67890 900 100 600 400 host  Firefox\n"
    )
    monkeypatch.setattr(linux_backend.subprocess, "check_output", lambda *a, **kw: sample)
    out = b.window_list()
    assert len(out) == 2
    assert out[0].pid == 12345
    assert out[0].window_id.startswith("win_")
    assert out[1].title == "Firefox"


# ── Clipboard (Q3) ─────────────────────────────────────────────────────


def test_clipboard_disabled_returns_feature_not_supported(tmp_path) -> None:
    b = LinuxBackend(_config(tmp_path))
    with pytest.raises(FeatureNotSupportedError):
        b.clipboard_read()


def test_clipboard_writeback_round_trip(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True))
    state = {"content": None}

    def _write(content: str) -> int | None:
        state["content"] = content
        return None  # Linux CLI tools don't expose a seq number

    monkeypatch.setattr(b, "_write_clipboard_text", _write)
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: (state["content"], None))
    b.clipboard_write(content="hello")
    res = b.clipboard_read()
    assert res.content == "hello"


def test_clipboard_writeback_only_refuses_foreign(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: ("user_text", None))
    with pytest.raises(ClipboardConcealedError):
        b.clipboard_read()


def test_clipboard_read_foreign_returns_content(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: ("user_text", None))
    res = b.clipboard_read()
    assert res.content == "user_text"


# ── Permissions ────────────────────────────────────────────────────────


def test_permissions_check_x11_grants_tcc_keys(tmp_path) -> None:
    b = LinuxBackend(_config(tmp_path))
    out = b.permissions_check(["accessibility", "screen_recording"])
    assert out == {"accessibility": "granted", "screen_recording": "granted"}


def test_permissions_check_wayland_denies_tcc_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(linux_backend, "_is_wayland", lambda: True)
    b = LinuxBackend(_config(tmp_path))
    out = b.permissions_check(["accessibility", "screen_recording"])
    assert out == {"accessibility": "denied", "screen_recording": "denied"}


# ── Paste with preserve_clipboard (Q3/Q4) ───────────────────────────────


def test_a_clipboard_restored_after_a_paste_stays_unreadable(tmp_path, monkeypatch) -> None:
    """The user's clipboard goes back after the paste, and it is still theirs:
    a writeback-only read must not return it."""
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_wayland", False)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    state = {"content": "user secret"}

    def _write(text: str) -> None:
        state["content"] = text

    monkeypatch.setattr(b, "_write_clipboard_text", _write)
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: (state["content"], None))
    _clipboard_targets(monkeypatch, ["TARGETS", "UTF8_STRING"])

    b.input_action("type", text="x" * PASTE_THRESHOLD_CHARS, paste=True, preserve_clipboard=True)

    pyautogui.hotkey.assert_called_once_with("ctrl", "v")
    assert state["content"] == "user secret"
    with pytest.raises(ClipboardConcealedError):
        b.clipboard_read()


def test_a_clipboard_with_no_text_to_put_back_is_left_alone_and_the_text_typed(
    tmp_path, monkeypatch
) -> None:
    """xclip can't read an empty clipboard, an image, or anything else without
    a text target. A paste would leave the pasted text there in its place."""
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_wayland", False)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    writes: list[str] = []
    monkeypatch.setattr(b, "_write_clipboard_text", writes.append)
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: (None, None))
    text = "x" * PASTE_THRESHOLD_CHARS

    b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    assert writes == []
    pyautogui.write.assert_called_once_with(text, interval=0.02)
    pyautogui.hotkey.assert_not_called()


@pytest.mark.parametrize(
    "targets",
    [_GTK_TEXT_TARGETS, ["UTF8_STRING", "text/plain;charset=UTF-8"]],
    ids=["gtk", "charset-in-capitals"],
)
def test_a_preserved_paste_goes_ahead_when_the_clipboard_holds_only_text(
    tmp_path, monkeypatch, targets
) -> None:
    """GTK lists COMPOUND_TEXT and the selection's own targets beside the
    text ones; restoring the text rebuilds all of them. A MIME type's case
    doesn't matter."""
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_wayland", False)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    state = {"content": "user text"}
    monkeypatch.setattr(b, "_write_clipboard_text", lambda text: state.update(content=text))
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: (state["content"], None))
    _clipboard_targets(monkeypatch, targets)

    b.input_action("type", text="x" * PASTE_THRESHOLD_CHARS, paste=True, preserve_clipboard=True)

    pyautogui.hotkey.assert_called_once_with("ctrl", "v")
    pyautogui.write.assert_not_called()
    assert state["content"] == "user text"


@pytest.mark.parametrize(
    "targets",
    [
        [*_GTK_TEXT_TARGETS, "text/html"],
        ["TIMESTAMP", "TARGETS", "image/png", "UTF8_STRING"],
        None,
    ],
    ids=["rich-text", "image", "unlisted"],
)
def test_a_preserved_paste_types_over_a_clipboard_holding_more_than_text(
    tmp_path, monkeypatch, targets: list[str] | None
) -> None:
    """The restore puts back text only, so the HTML or image beside it was
    lost. Targets that can't be listed count as more than text."""
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_wayland", False)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    writes: list[str] = []
    monkeypatch.setattr(b, "_write_clipboard_text", writes.append)
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: ("user text", None))
    _clipboard_targets(monkeypatch, targets)
    text = "x" * PASTE_THRESHOLD_CHARS

    b.input_action("type", text=text, paste=True, preserve_clipboard=True)

    assert writes == []
    pyautogui.write.assert_called_once_with(text, interval=0.02)
    pyautogui.hotkey.assert_not_called()


@pytest.mark.parametrize(
    ("wayland", "expected"),
    [(False, _LIST_TARGETS), (True, ["wl-paste", "--list-types"])],
    ids=["x11", "wayland"],
)
def test_the_clipboard_targets_come_from_xclip_on_x11_and_wl_paste_on_wayland(
    tmp_path, monkeypatch, wayland: bool, expected: list[str]
) -> None:
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_wayland", wayland)
    commands: list[list[str]] = []

    def check_output(command, **_kwargs):
        commands.append(command)
        return b"text/plain\ntext/html\n"

    monkeypatch.setattr(linux_backend.subprocess, "check_output", check_output)

    assert b._clipboard_holds_only_text() is False
    assert commands == [expected]


def test_a_failed_clipboard_write_is_an_error_and_never_pastes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_wayland", False)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)

    def fail(command, **_kwargs):
        raise linux_backend.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(linux_backend.subprocess, "run", fail)

    with pytest.raises(ClipboardWriteError):
        b.clipboard_write(content="hello")
    with pytest.raises(ClipboardWriteError):
        b.input_action("type", text="x" * PASTE_THRESHOLD_CHARS, paste=True)

    pyautogui.hotkey.assert_not_called()


@pytest.mark.parametrize("helper", ["gtk-launch", "dex"])
def test_desktop_files_launch_with_the_argument_each_helper_takes(
    tmp_path, monkeypatch, helper: str
) -> None:
    """gtk-launch takes a desktop file ID; dex takes the file's path."""
    applications = _xdg_applications(monkeypatch, tmp_path)
    folder = applications if helper == "gtk-launch" else tmp_path / "opt" / "apps"
    desktop = _desktop_file(folder, "org.gnome.Calculator.desktop")
    monkeypatch.setattr(
        linux_backend.shutil, "which", lambda name: f"/usr/bin/{name}" if name == helper else None
    )
    popen = MagicMock()
    popen.return_value.pid = 4321
    popen.return_value.wait.return_value = 0
    monkeypatch.setattr(linux_backend.subprocess, "Popen", popen)
    b = LinuxBackend(_config(tmp_path))

    b.app_launch(executable_path=str(desktop))

    argument = "org.gnome.Calculator" if helper == "gtk-launch" else str(desktop)
    assert popen.call_args.args[0] == [helper, argument]


def test_a_desktop_file_refuses_launch_arguments(tmp_path, monkeypatch) -> None:
    """The file's Exec line sets the app's arguments, so `args` were dropped
    without a word."""
    monkeypatch.setattr(linux_backend.shutil, "which", lambda name: f"/usr/bin/{name}")
    popen = MagicMock()
    monkeypatch.setattr(linux_backend.subprocess, "Popen", popen)
    b = LinuxBackend(_config(tmp_path))

    with pytest.raises(FeatureNotSupportedError, match="args"):
        b.app_launch(
            executable_path="/usr/share/applications/firefox.desktop", args=["--private-window"]
        )

    popen.assert_not_called()


@pytest.mark.parametrize("in_applications", [True, False], ids=["in-applications", "elsewhere"])
def test_a_desktop_file_outside_the_application_directories_launches_with_dex(
    tmp_path, monkeypatch, in_applications: bool
) -> None:
    """gtk-launch looks a desktop file ID up in the XDG application
    directories, so it can't find a file anywhere else; dex runs the file."""
    applications = _xdg_applications(monkeypatch, tmp_path)
    folder = applications if in_applications else tmp_path / "Desktop"
    desktop_file = _desktop_file(folder)
    monkeypatch.setattr(linux_backend.shutil, "which", lambda name: f"/usr/bin/{name}")
    popen = MagicMock()
    popen.return_value.pid = 4321
    popen.return_value.wait.return_value = 0
    monkeypatch.setattr(linux_backend.subprocess, "Popen", popen)
    b = LinuxBackend(_config(tmp_path))

    assert b.app_launch(executable_path=str(desktop_file)) is None
    assert popen.call_args.args[0] == (
        ["gtk-launch", "calculator"] if in_applications else ["dex", str(desktop_file)]
    )


def test_a_desktop_file_gtk_launch_cannot_find_is_refused(tmp_path, monkeypatch) -> None:
    """Without dex, a file outside the application directories went to
    gtk-launch by its ID, which finds another file of that name, or none."""
    _xdg_applications(monkeypatch, tmp_path)
    _desktop_file(tmp_path / "Desktop")
    monkeypatch.setattr(
        linux_backend.shutil,
        "which",
        lambda name: "/usr/bin/gtk-launch" if name == "gtk-launch" else None,
    )
    popen = MagicMock()
    popen.return_value.pid = 4321
    monkeypatch.setattr(linux_backend.subprocess, "Popen", popen)
    b = LinuxBackend(_config(tmp_path))

    with pytest.raises(FeatureNotSupportedError, match="dex"):
        b.app_launch(executable_path=str(tmp_path / "Desktop" / "calculator.desktop"))

    popen.assert_not_called()


def test_a_gtk_launch_still_starting_the_app_counts_as_launched(tmp_path, monkeypatch) -> None:
    """Only a failing exit status says the launch failed."""
    applications = _xdg_applications(monkeypatch, tmp_path)
    monkeypatch.setattr(linux_backend.shutil, "which", lambda name: f"/usr/bin/{name}")
    popen = MagicMock()
    popen.return_value.pid = 4321
    popen.return_value.wait.side_effect = linux_backend.subprocess.TimeoutExpired("gtk-launch", 10)
    monkeypatch.setattr(linux_backend.subprocess, "Popen", popen)
    b = LinuxBackend(_config(tmp_path))

    assert b.app_launch(executable_path=str(_desktop_file(applications))) is None


def test_the_clipboard_is_restored_when_the_paste_hotkey_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_wayland", False)
    pyautogui = MagicMock()
    pyautogui.hotkey.side_effect = RuntimeError("fail-safe")
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    state = {"content": "user text"}
    monkeypatch.setattr(b, "_write_clipboard_text", lambda text: state.update(content=text))
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: (state["content"], None))
    _clipboard_targets(monkeypatch, ["TARGETS", "UTF8_STRING"])

    with pytest.raises(RuntimeError, match="fail-safe"):
        b.input_action(
            "type", text="x" * PASTE_THRESHOLD_CHARS, paste=True, preserve_clipboard=True
        )

    pyautogui.hotkey.assert_called_once_with("ctrl", "v")
    assert state["content"] == "user text"


def test_the_linux_restore_leaves_a_copy_made_during_the_paste(tmp_path, monkeypatch) -> None:
    """The restore wrote the snapshot back without looking, over a copy
    the user made in the 50 ms before it."""
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_wayland", False)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    state = {"content": "user text"}
    monkeypatch.setattr(b, "_write_clipboard_text", lambda text: state.update(content=text))
    monkeypatch.setattr(b, "_read_clipboard_text", lambda: (state["content"], None))
    _clipboard_targets(monkeypatch, ["TARGETS", "UTF8_STRING"])
    pyautogui.hotkey.side_effect = lambda *_keys: state.update(content="copied during the paste")

    b.input_action("type", text="x" * PASTE_THRESHOLD_CHARS, paste=True, preserve_clipboard=True)

    pyautogui.hotkey.assert_called_once_with("ctrl", "v")
    assert state["content"] == "copied during the paste"


def test_typing_refuses_text_it_cannot_type_instead_of_dropping_it(tmp_path, monkeypatch) -> None:
    """pyautogui types only ASCII; anything else was skipped without a word."""
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_wayland", False)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)

    with pytest.raises(FeatureNotSupportedError, match="paste"):
        b.input_action("type", text="Grüße, 世界")

    pyautogui.write.assert_not_called()


def test_x11_without_xclip_types_rather_than_pasting_through_wl_copy(tmp_path, monkeypatch) -> None:
    """wl-copy counted as a clipboard tool on X11, where writes go through
    xclip only, so the paste failed instead of typing the text."""
    monkeypatch.setattr(
        linux_backend.shutil, "which", lambda n: "/usr/bin/wl-copy" if n == "wl-copy" else None
    )
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True))
    monkeypatch.setattr(b, "_wayland", False)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)

    b.input_action("type", text="x" * PASTE_THRESHOLD_CHARS, paste=True)

    pyautogui.write.assert_called_once()
    pyautogui.hotkey.assert_not_called()
    assert "clipboard" not in b.coarse_features()
    assert "clipboard.write" not in b.features()


@pytest.mark.parametrize("name", ["super", "meta"])
def test_super_and_meta_press_the_super_key_on_x11(tmp_path, monkeypatch, name: str) -> None:
    """pyautogui's X11 keys call the Super key `win`; chords passed `super`
    and `meta` through, and it skips a name it doesn't know without a word."""
    b = LinuxBackend(_config(tmp_path))
    monkeypatch.setattr(b, "_wayland", False)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)

    b.input_action("key", key=f"{name}+l")

    pyautogui.hotkey.assert_called_once_with("win", "l")


@pytest.mark.parametrize("at_point", [False, True], ids=["at-cursor", "at-point"])
def test_a_left_or_right_scroll_scrolls_sideways_on_x11(
    tmp_path, displays_1080p, monkeypatch, at_point: bool
) -> None:
    """Every scroll went to pyautogui.scroll, so left and right scrolled up
    and down instead."""
    b = LinuxBackend(_config(tmp_path))
    monkeypatch.setattr(b, "_wayland", False)
    monkeypatch.setattr(b, "display_info", lambda: displays_1080p)
    pyautogui = MagicMock()
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)
    point = {"coordinate": (100, 200)} if at_point else {}

    b.input_action("scroll", scroll_direction="left", scroll_amount=2, **point)

    expected = {"x": 100, "y": 200} if at_point else {}
    pyautogui.hscroll.assert_called_once_with(-2, **expected)
    pyautogui.scroll.assert_not_called()


def test_an_unknown_key_name_is_refused_on_x11(tmp_path, monkeypatch) -> None:
    b = LinuxBackend(_config(tmp_path))
    monkeypatch.setattr(b, "_wayland", False)
    pyautogui = MagicMock()
    pyautogui.KEY_NAMES = ["ctrl", "l", "win"]
    monkeypatch.setitem(sys.modules, "pyautogui", pyautogui)

    with pytest.raises(FeatureNotSupportedError, match="bogus"):
        b.input_action("key", key="super+bogus")
    pyautogui.hotkey.assert_not_called()


def _desktop_file(folder: Path, name: str = "calculator.desktop", exec_line: str = "calc") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(f"[Desktop Entry]\nType=Application\nExec={exec_line}\n")
    return path


def _helpers(monkeypatch, *names: str) -> MagicMock:
    monkeypatch.setattr(
        linux_backend.shutil, "which", lambda name: f"/usr/bin/{name}" if name in names else None
    )
    popen = MagicMock()
    popen.return_value.pid = 4321
    popen.return_value.wait.return_value = 0
    monkeypatch.setattr(linux_backend.subprocess, "Popen", popen)
    return popen


@pytest.mark.parametrize("has_dex", [True, False], ids=["with-dex", "gtk-launch-only"])
def test_a_desktop_file_another_entry_shadows_is_not_started_through_gtk_launch(
    tmp_path, monkeypatch, has_dex: bool
) -> None:
    """gtk-launch starts the first file with that desktop ID in the XDG
    directories, so a same-named entry in the user's directory started
    instead of the system file asked for, and the launch was acknowledged."""
    user_applications = _xdg_applications(monkeypatch, tmp_path)
    wanted = _desktop_file(tmp_path / "system" / "applications", "editor.desktop", "wanted")
    _desktop_file(user_applications, "editor.desktop", "someone-else")
    popen = _helpers(monkeypatch, "gtk-launch", *(["dex"] if has_dex else []))
    b = LinuxBackend(_config(tmp_path))

    if has_dex:
        b.app_launch(executable_path=str(wanted))
        assert popen.call_args.args[0] == ["dex", str(wanted)]
    else:
        with pytest.raises(FeatureNotSupportedError, match="dex"):
            b.app_launch(executable_path=str(wanted))
        popen.assert_not_called()


@pytest.mark.parametrize("has_dex", [True, False], ids=["with-dex", "gtk-launch-only"])
def test_a_desktop_file_in_an_applications_subdirectory_launches_by_its_desktop_id(
    tmp_path, monkeypatch, has_dex: bool
) -> None:
    """A file in a subdirectory of an applications directory has its path
    below that directory as its ID, with "/" turned into "-", so
    applications/kde4/kcalc.desktop is kde4-kcalc. Only the file's name was
    looked up, so without dex it was refused, though gtk-launch starts it."""
    applications = _xdg_applications(monkeypatch, tmp_path)
    desktop = _desktop_file(applications / "kde4", "kcalc.desktop")
    popen = _helpers(monkeypatch, "gtk-launch", *(["dex"] if has_dex else []))
    b = LinuxBackend(_config(tmp_path))

    assert b.app_launch(executable_path=str(desktop)) is None
    assert popen.call_args.args[0] == ["gtk-launch", "kde4-kcalc"]


@pytest.mark.parametrize(
    "other",
    ["share/applications/kde4-kcalc.desktop", "system/applications/kde4-kcalc.desktop"],
    ids=["earlier-directory", "same-directory"],
)
def test_a_nested_desktop_file_whose_id_leads_elsewhere_launches_with_dex(
    tmp_path, monkeypatch, other: str
) -> None:
    """gtk-launch starts the first file with the ID: one in an earlier
    directory, or, in the same directory, whichever it lists first."""
    _xdg_applications(monkeypatch, tmp_path)
    wanted = _desktop_file(tmp_path / "system" / "applications" / "kde4", "kcalc.desktop")
    elsewhere = tmp_path / other
    _desktop_file(elsewhere.parent, elsewhere.name, "someone-else")
    popen = _helpers(monkeypatch, "gtk-launch", "dex")
    b = LinuxBackend(_config(tmp_path))

    b.app_launch(executable_path=str(wanted))

    assert popen.call_args.args[0] == ["dex", str(wanted)]


@pytest.mark.parametrize("helper", ["gtk-launch", "dex"])
def test_a_desktop_launch_reports_no_pid(tmp_path, monkeypatch, helper: str) -> None:
    """The pid was the helper's, which exits once the app has started: it
    named a process already gone, and one the OS could reuse."""
    applications = _xdg_applications(monkeypatch, tmp_path)
    folder = applications if helper == "gtk-launch" else tmp_path / "Desktop"
    desktop = _desktop_file(folder)
    popen = _helpers(monkeypatch, helper)
    b = LinuxBackend(_config(tmp_path))

    assert b.app_launch(executable_path=str(desktop)) is None
    assert popen.call_args.args[0][0] == helper


@pytest.mark.parametrize("helper", ["gtk-launch", "dex"])
def test_a_desktop_launch_whose_helper_fails_is_an_error(tmp_path, monkeypatch, helper) -> None:
    """dex's exit status went unchecked, so a file it couldn't start was
    acknowledged as launched."""
    applications = _xdg_applications(monkeypatch, tmp_path)
    folder = applications if helper == "gtk-launch" else tmp_path / "Desktop"
    desktop = _desktop_file(folder)
    popen = _helpers(monkeypatch, helper)
    popen.return_value.wait.return_value = 2
    b = LinuxBackend(_config(tmp_path))

    with pytest.raises(FeatureNotSupportedError, match=helper):
        b.app_launch(executable_path=str(desktop))


@pytest.mark.parametrize("helpers", [("gtk-launch", "dex"), ("dex",)], ids=["both", "dex"])
def test_a_desktop_file_that_does_not_exist_is_refused_before_a_helper_starts(
    tmp_path, monkeypatch, helpers
) -> None:
    """dex started for a path with no file behind it, and the launch was
    acknowledged."""
    _xdg_applications(monkeypatch, tmp_path)
    popen = _helpers(monkeypatch, *helpers)
    b = LinuxBackend(_config(tmp_path))

    with pytest.raises(FeatureNotSupportedError, match="no .desktop file"):
        b.app_launch(executable_path=str(tmp_path / "Desktop" / "missing.desktop"))
    popen.assert_not_called()


@pytest.mark.parametrize(("try_exec", "launched"), [("/no/such/app", False), (None, True)])
def test_a_desktop_file_whose_try_exec_is_missing_is_refused(
    tmp_path, monkeypatch, try_exec, launched: bool
) -> None:
    """A file whose TryExec program isn't installed can't start, but dex can
    still exit 0 for it, so the launch was acknowledged."""
    import sys as _sys

    _xdg_applications(monkeypatch, tmp_path)
    desktop = tmp_path / "Desktop" / "editor.desktop"
    desktop.parent.mkdir(parents=True)
    program = try_exec or _sys.executable
    desktop.write_text(f"[Desktop Entry]\nType=Application\nTryExec={program}\nExec=editor\n")
    popen = _helpers(monkeypatch, "dex")
    b = LinuxBackend(_config(tmp_path))

    if launched:
        assert b.app_launch(executable_path=str(desktop)) is None
        assert popen.call_args.args[0] == ["dex", str(desktop)]
    else:
        with pytest.raises(FeatureNotSupportedError, match="TryExec"):
            b.app_launch(executable_path=str(desktop))
        popen.assert_not_called()


def _clipboard_read_fails(monkeypatch, said: bytes) -> list[list[str]]:
    """Make every clipboard tool exit 1 after printing `said`."""
    commands: list[list[str]] = []

    def check_output(command, **kwargs):
        commands.append(command)
        assert kwargs.get("stderr") == linux_backend.subprocess.PIPE
        raise linux_backend.subprocess.CalledProcessError(1, command, stderr=said)

    monkeypatch.setattr(linux_backend.subprocess, "check_output", check_output)
    return commands


@pytest.mark.parametrize("wayland", [False, True], ids=["x11", "wayland"])
def test_a_foreign_clipboard_read_that_fails_is_an_error(
    tmp_path, monkeypatch, wayland: bool
) -> None:
    """A failed xclip or wl-paste read answered as an empty clipboard, so the
    platform couldn't tell a failed read from an empty clipboard."""
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_wayland", wayland)
    _clipboard_read_fails(monkeypatch, b"Error: Can't open display: :0\n")

    with pytest.raises(ClipboardConcealedError, match="could not be read") as raised:
        b.clipboard_read()
    assert raised.value.details == {"reason": "concealed_type", "marker": "unknown"}


@pytest.mark.parametrize(
    ("wayland", "said"),
    [
        (False, b"xclip: Error: There is no owner for the CLIPBOARD selection\n"),
        (False, b"xclip: Error: firefox cannot convert CLIPBOARD selection to target 'STRING'\n"),
        (False, b"Error: target STRING not available\n"),
        (True, b"Nothing is copied\n"),
        (True, b'Clipboard content is not available as requested type "text"\n'),
    ],
    ids=["x11-empty", "x11-no-text", "old-xclip-no-text", "wayland-empty", "wayland-no-text"],
)
def test_a_foreign_read_of_a_clipboard_with_no_text_is_empty(
    tmp_path, monkeypatch, wayland: bool, said: bytes
) -> None:
    """The tools fail when the clipboard is empty or holds no text, which
    reads as empty, not as a failed read."""
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_wayland", wayland)
    _clipboard_read_fails(monkeypatch, said)

    assert b.clipboard_read().content == ""


def test_wl_paste_reads_the_clipboard_as_text(tmp_path, monkeypatch) -> None:
    """Without a type, wl-paste pastes whatever the clipboard holds, so an
    image came back as its bytes, decoded as text."""
    monkeypatch.setattr(linux_backend.shutil, "which", lambda n: "/usr/bin/" + n)
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_wayland", True)
    commands = _clipboard_read_fails(monkeypatch, b"Nothing is copied\n")

    b.clipboard_read()

    assert commands == [["wl-paste", "--no-newline", "--type", "text"]]


def test_a_clipboard_read_with_no_tool_that_reads_is_refused(tmp_path, monkeypatch) -> None:
    """On Wayland with wl-copy alone there is nothing to read with, and a
    foreign read answered with an empty clipboard instead of refusing."""
    monkeypatch.setattr(
        linux_backend.shutil, "which", lambda n: "/usr/bin/wl-copy" if n == "wl-copy" else None
    )
    b = LinuxBackend(_config(tmp_path, enable_clipboard=True, enable_clipboard_read_foreign=True))
    monkeypatch.setattr(b, "_wayland", True)

    assert "clipboard.read" not in b.features()
    with pytest.raises(FeatureNotSupportedError, match="wl-paste"):
        b.clipboard_read()
