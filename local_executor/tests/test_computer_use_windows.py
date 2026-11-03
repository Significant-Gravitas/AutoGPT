"""Windows backend tests — all native APIs mocked so they run anywhere."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autogpt_local_executor.computer_use.backends.windows import (
    CF_PRIVATEFIRST,
    WindowsBackend,
)
from autogpt_local_executor.computer_use.errors import (
    ClipboardConcealedError,
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
