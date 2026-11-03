"""Linux backend tests — Wayland, X11, wmctrl, xclip all mocked."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autogpt_local_executor.computer_use.backends import linux as linux_backend
from autogpt_local_executor.computer_use.backends.linux import LinuxBackend
from autogpt_local_executor.computer_use.errors import (
    ClipboardConcealedError,
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
