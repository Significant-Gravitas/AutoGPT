"""Tests for `autogpt-shim doctor` and the daemon TCC preflight."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from autogpt_local_executor import doctor
from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.config import ShimConfig


def _config(tmp_path: Path, **overrides) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "_audit.log",
        platform_url="http://localhost:9999",
        machine_id="test-machine",
        **overrides,
    )


def test_doctor_returns_zero_when_computer_use_not_requested(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """No computer-use requested → blocking permissions don't block."""
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "darwin")
    monkeypatch.setattr(doctor, "_ax_trusted_with_prompt", lambda: False)
    monkeypatch.setattr(doctor, "_screen_recording_granted", lambda: False)
    monkeypatch.setattr(
        "autogpt_local_executor.computer_use.get_backend",
        lambda _: MagicMock(display_info=lambda: []),
    )
    config = _config(tmp_path, enable_computer_use=False)
    rc = doctor.run_doctor(config)
    assert rc == 0


def test_control_doctor_checks_browser_without_creating_legacy_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_root = tmp_path / "must-not-be-created"
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setattr(
        "autogpt_local_executor.computer_use.get_backend",
        lambda _: MagicMock(display_info=lambda: []),
    )
    config = _config(tmp_path, session_id=None)
    config.allowed_root = missing_root

    assert doctor.run_doctor(config) == 0
    assert not missing_root.exists()


def test_doctor_returns_78_when_ax_denied_and_computer_use_requested(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "darwin")
    monkeypatch.setattr(doctor, "_ax_trusted_with_prompt", lambda: False)
    monkeypatch.setattr(doctor, "_screen_recording_granted", lambda: True)
    monkeypatch.setattr(
        "autogpt_local_executor.computer_use.get_backend",
        lambda _: MagicMock(display_info=lambda: []),
    )
    config = _config(tmp_path, enable_computer_use=True)
    rc = doctor.run_doctor(config)
    assert rc == doctor.EX_CONFIG


def test_doctor_linux_wayland_warns_when_computer_use_on(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.setattr(
        "autogpt_local_executor.computer_use.get_backend",
        lambda _: MagicMock(display_info=lambda: []),
    )
    config = _config(tmp_path, enable_computer_use=True)
    rc = doctor.run_doctor(config)
    assert rc == doctor.EX_CONFIG


def test_doctor_linux_x11_ok_when_computer_use_on(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(
        "autogpt_local_executor.computer_use.get_backend",
        lambda _: MagicMock(display_info=lambda: []),
    )
    config = _config(tmp_path, enable_computer_use=True)
    rc = doctor.run_doctor(config)
    assert rc == 0


def test_doctor_wsl2_blocks_when_computer_use_on(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "wsl2")
    monkeypatch.setattr(
        "autogpt_local_executor.computer_use.get_backend",
        lambda _: MagicMock(display_info=lambda: []),
    )
    config = _config(tmp_path, enable_computer_use=True)
    rc = doctor.run_doctor(config)
    assert rc == doctor.EX_CONFIG


# ── Daemon TCC preflight (C2) ───────────────────────────────────────────


def _daemon(config: ShimConfig):
    from autogpt_local_executor.daemon import ShimDaemon

    audit = AuditWriter(path=config.audit_log_path, audit_key=b"a" * 32)
    return ShimDaemon(config=config, token_store=MagicMock(), audit=audit)


@pytest.mark.asyncio
async def test_daemon_preflight_passes_when_computer_use_off(tmp_path) -> None:
    config = _config(tmp_path, enable_computer_use=False)
    await _daemon(config)._preflight_or_raise()


async def test_daemon_preflight_raises_when_ax_denied_on_macos(tmp_path, monkeypatch) -> None:
    """Q5: macOS + computer_use requested + AX denied → DaemonPreflightError."""
    from autogpt_local_executor.daemon import DaemonPreflightError

    monkeypatch.setattr(
        "autogpt_local_executor.daemon.platform_info.detect_platform",
        lambda: "darwin",
    )
    import sys

    fake_appsvc = MagicMock()
    fake_appsvc.AXIsProcessTrusted = lambda: False
    monkeypatch.setitem(sys.modules, "ApplicationServices", fake_appsvc)
    config = _config(tmp_path, enable_computer_use=True)
    d = _daemon(config)
    with pytest.raises(DaemonPreflightError):
        await d._preflight_or_raise()
    assert '"op": "DAEMON_PREFLIGHT_FAILED"' in config.audit_log_path.read_text()


async def test_daemon_preflight_passes_when_ax_granted_on_macos(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "autogpt_local_executor.daemon.platform_info.detect_platform",
        lambda: "darwin",
    )
    import sys

    fake_appsvc = MagicMock()
    fake_appsvc.AXIsProcessTrusted = lambda: True
    monkeypatch.setitem(sys.modules, "ApplicationServices", fake_appsvc)
    config = _config(tmp_path, enable_computer_use=True)
    await _daemon(config)._preflight_or_raise()


async def test_daemon_preflight_no_op_on_non_macos(tmp_path, monkeypatch) -> None:
    """Per Q5, the AX gate only applies to macOS — Windows/Linux are
    handled by their own permission models."""
    monkeypatch.setattr(
        "autogpt_local_executor.daemon.platform_info.detect_platform",
        lambda: "linux",
    )
    config = _config(tmp_path, enable_computer_use=True)
    await _daemon(config)._preflight_or_raise()


@pytest.mark.asyncio
async def test_daemon_preflight_fails_closed_when_ax_probe_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from autogpt_local_executor.daemon import DaemonPreflightError

    monkeypatch.setattr(
        "autogpt_local_executor.daemon.platform_info.detect_platform",
        lambda: "darwin",
    )
    monkeypatch.setitem(sys.modules, "ApplicationServices", None)
    config = _config(tmp_path, enable_computer_use=True)

    with pytest.raises(DaemonPreflightError, match="Cannot verify Accessibility"):
        await _daemon(config)._preflight_or_raise()


@pytest.mark.asyncio
async def test_daemon_preflight_fails_closed_when_ax_probe_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from autogpt_local_executor.daemon import DaemonPreflightError

    monkeypatch.setattr(
        "autogpt_local_executor.daemon.platform_info.detect_platform",
        lambda: "darwin",
    )
    fake_appsvc = MagicMock()
    fake_appsvc.AXIsProcessTrusted.side_effect = RuntimeError("TCC unavailable")
    monkeypatch.setitem(sys.modules, "ApplicationServices", fake_appsvc)
    config = _config(tmp_path, enable_computer_use=True)

    with pytest.raises(DaemonPreflightError, match="Could not verify Accessibility"):
        await _daemon(config)._preflight_or_raise()
