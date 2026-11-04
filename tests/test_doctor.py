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
    monkeypatch.setenv("DISPLAY", ":0")
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
    _fake_screen_recording(monkeypatch, granted=True)
    config = _config(tmp_path, enable_computer_use=True)
    d = _daemon(config)
    with pytest.raises(DaemonPreflightError, match="^Accessibility permission not granted"):
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
    _fake_screen_recording(monkeypatch, granted=True)
    config = _config(tmp_path, enable_computer_use=True)
    await _daemon(config)._preflight_or_raise()


def _fake_screen_recording(monkeypatch: pytest.MonkeyPatch, *, granted: bool) -> None:
    import sys

    fake_quartz = MagicMock()
    fake_quartz.CGPreflightScreenCaptureAccess = lambda: granted
    monkeypatch.setitem(sys.modules, "Quartz", fake_quartz)


async def test_daemon_preflight_raises_when_screen_recording_denied_on_macos(
    tmp_path, monkeypatch
) -> None:
    """Without Screen Recording, macOS captures show only the wallpaper and
    menu bar, so computer use can't work; exit 78 as for Accessibility."""
    import json
    import sys

    from autogpt_local_executor.daemon import DaemonPreflightError

    monkeypatch.setattr(
        "autogpt_local_executor.daemon.platform_info.detect_platform",
        lambda: "darwin",
    )
    fake_appsvc = MagicMock()
    fake_appsvc.AXIsProcessTrusted = lambda: True
    monkeypatch.setitem(sys.modules, "ApplicationServices", fake_appsvc)
    _fake_screen_recording(monkeypatch, granted=False)
    config = _config(tmp_path, enable_computer_use=True)

    with pytest.raises(DaemonPreflightError, match="^Screen Recording permission not granted"):
        await _daemon(config)._preflight_or_raise()

    [record] = [json.loads(line) for line in config.audit_log_path.read_text().splitlines()]
    assert record["op"] == "DAEMON_PREFLIGHT_FAILED"
    assert record["details"]["missing_permissions"] == ["screen_recording"]


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
    _fake_screen_recording(monkeypatch, granted=True)
    config = _config(tmp_path, enable_computer_use=True)

    with pytest.raises(DaemonPreflightError, match="Could not verify Accessibility"):
        await _daemon(config)._preflight_or_raise()


def test_doctor_linux_blocks_when_no_display_session_is_detected(tmp_path, monkeypatch) -> None:
    """Without XDG_SESSION_TYPE, DISPLAY or WAYLAND_DISPLAY there is no screen to
    use, which blocks computer use just as Wayland and WSL2 do."""
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    for name in ("XDG_SESSION_TYPE", "DISPLAY", "WAYLAND_DISPLAY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "autogpt_local_executor.computer_use.get_backend",
        lambda _: MagicMock(display_info=lambda: []),
    )

    assert doctor.run_doctor(_config(tmp_path, enable_computer_use=True)) == doctor.EX_CONFIG
    assert doctor.run_doctor(_config(tmp_path, enable_computer_use=False)) == 0


def test_doctor_keychain_warning_states_the_real_requirement(monkeypatch) -> None:
    """OAuth tokens have no file fallback; only the audit key does, and only
    with a passphrase."""
    import sys

    keyring = MagicMock()
    keyring.get_keyring.side_effect = RuntimeError("no backend")
    monkeypatch.setitem(sys.modules, "keyring", keyring)

    result = doctor._check_keychain()

    assert result.marker == "WARN"
    assert "fallback will be used" not in result.detail
    assert "sign-in" in result.detail
    assert "AUTOGPT_SHIM_KEYCHAIN_PASSPHRASE" in result.detail


def test_doctor_linux_reads_display_when_the_session_type_is_unrecognized(
    tmp_path, monkeypatch
) -> None:
    """XDG_SESSION_TYPE=tty with DISPLAY set, as in a remote shell into a
    running X session, still has a usable X11 display."""
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "tty")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(
        "autogpt_local_executor.computer_use.get_backend",
        lambda _: MagicMock(display_info=lambda: []),
    )

    assert doctor.run_doctor(_config(tmp_path, enable_computer_use=True)) == 0


def test_doctor_linux_warns_when_an_x11_session_has_no_display(tmp_path, monkeypatch) -> None:
    """XDG_SESSION_TYPE=x11 with DISPLAY unset was reported as full computer_use."""
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(
        "autogpt_local_executor.computer_use.get_backend",
        lambda _: MagicMock(display_info=lambda: []),
    )

    assert doctor.run_doctor(_config(tmp_path, enable_computer_use=True)) == 78


class _MemoryKeyring:
    def __init__(self, *, keeps: bool = True) -> None:
        self.keeps = keeps
        self.stored: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, name: str, value: str) -> None:
        if self.keeps:
            self.stored[(service, name)] = value

    def get_password(self, service: str, name: str) -> str | None:
        return self.stored.get((service, name))

    def delete_password(self, service: str, name: str) -> None:
        self.stored.pop((service, name), None)


def _keyring_module(backend: object) -> MagicMock:
    module = MagicMock()
    module.get_keyring.return_value = backend
    module.set_password.side_effect = backend.set_password
    module.get_password.side_effect = backend.get_password
    module.delete_password.side_effect = backend.delete_password
    return module


@pytest.mark.parametrize(
    ("keeps", "marker"), [(True, "OK"), (False, "WARN")], ids=["works", "forgets"]
)
def test_doctor_round_trips_a_probe_through_the_keychain(monkeypatch, keeps, marker) -> None:
    """Naming the backend said nothing about whether it stores anything."""
    import sys

    backend = _MemoryKeyring(keeps=keeps)
    monkeypatch.setitem(sys.modules, "keyring", _keyring_module(backend))

    result = doctor._check_keychain()

    assert result.marker == marker
    assert backend.stored == {}


def test_doctor_warns_on_a_keyring_backend_that_can_store_nothing(monkeypatch) -> None:
    import sys

    fail_keyring = type("Keyring", (_MemoryKeyring,), {"__module__": "keyring.backends.fail"})
    monkeypatch.setitem(sys.modules, "keyring", _keyring_module(fail_keyring()))

    result = doctor._check_keychain()

    assert result.marker == "WARN"
    assert "keyring.backends.fail" in result.detail


@pytest.fixture(autouse=True)
def _the_computer_use_modules_are_installed(monkeypatch):
    monkeypatch.setattr(doctor.platform_info, "_module_available", lambda _name: True)


@pytest.fixture(autouse=True)
def _an_x_server_that_answers(monkeypatch):
    """doctor opens the X display as HELLO does; none answers here."""
    monkeypatch.setattr(doctor.platform_info, "_x11_display_opens", lambda: True)


@pytest.fixture(autouse=True)
def _a_keychain_that_works(monkeypatch):
    """doctor round-trips a probe through the keychain; keep the real one,
    which may block on an unlock prompt, out of these tests."""
    import sys

    monkeypatch.setitem(sys.modules, "keyring", _keyring_module(_MemoryKeyring()))


def test_doctor_does_not_hang_on_a_keychain_that_never_answers(monkeypatch) -> None:
    import sys
    import threading

    answer = threading.Event()

    class Blocking(_MemoryKeyring):
        def set_password(self, service: str, name: str, value: str) -> None:
            answer.wait(30)

    monkeypatch.setitem(sys.modules, "keyring", _keyring_module(Blocking()))
    monkeypatch.setattr(doctor, "KEYCHAIN_PROBE_SECONDS", 0.2)
    try:
        result = doctor._check_keychain()
    finally:
        answer.set()

    assert result.marker == "WARN"
    assert "did not answer" in result.detail


def test_doctor_follows_the_backend_when_a_wayland_display_is_set(tmp_path, monkeypatch) -> None:
    """With WAYLAND_DISPLAY set the backend refuses input, but doctor looked at
    XDG_SESSION_TYPE=x11 and reported full computer_use."""
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr(
        "autogpt_local_executor.computer_use.get_backend",
        lambda _: MagicMock(display_info=lambda: []),
    )

    assert doctor.run_doctor(_config(tmp_path, enable_computer_use=True)) == 78


def test_doctor_does_not_hang_finding_a_keyring_backend(monkeypatch) -> None:
    """Finding the backend can block too (the Secret Service check makes
    D-Bus calls), and it ran before the time limit started."""
    import sys
    import threading
    import time

    answer = threading.Event()
    module = _keyring_module(_MemoryKeyring())
    module.get_keyring.side_effect = lambda: answer.wait(30)
    monkeypatch.setitem(sys.modules, "keyring", module)
    monkeypatch.setattr(doctor, "KEYCHAIN_PROBE_SECONDS", 0.2)
    started = time.monotonic()
    try:
        result = doctor._check_keychain()
    finally:
        answer.set()

    assert time.monotonic() - started < 5
    assert result.marker == "WARN"
    assert "did not answer" in result.detail


def test_doctor_warns_when_the_x_server_does_not_answer(tmp_path, monkeypatch) -> None:
    """With DISPLAY naming a server that had gone, doctor reported full
    computer_use while HELLO left it out."""
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", "localhost:10.0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(doctor.platform_info, "_x11_display_opens", lambda: False)
    monkeypatch.setattr(
        "autogpt_local_executor.computer_use.get_backend",
        lambda _: MagicMock(display_info=lambda: []),
    )

    result = next(
        r
        for r in doctor._check_linux(_config(tmp_path), is_wsl2=False)
        if r.label == "display_server"
    )

    assert result.marker == "WARN"
    assert "localhost:10.0" in result.detail
    assert doctor.run_doctor(_config(tmp_path, enable_computer_use=True)) == 78


def test_doctor_warns_when_a_computer_use_module_is_missing(tmp_path, monkeypatch, capsys) -> None:
    """HELLO leaves computer_use out when a module it needs is missing, but
    doctor reported X11 computer_use as available and exited 0."""
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(doctor.platform_info, "_module_available", lambda name: name != "mss")
    monkeypatch.setattr(
        "autogpt_local_executor.computer_use.get_backend",
        lambda _: MagicMock(display_info=lambda: []),
    )

    assert doctor.run_doctor(_config(tmp_path, enable_computer_use=True)) == 78
    out = capsys.readouterr().out
    assert "mss" in out
    assert "full computer_use available" not in out


@pytest.mark.parametrize(
    ("session", "tools", "ok"),
    [
        ("x11", {"wl-copy"}, False),
        ("x11", {"xclip"}, True),
        ("wayland", {"wl-copy"}, True),
        ("wayland", {"xclip"}, True),
        ("wayland", set(), False),
    ],
    ids=["x11-wl-copy-only", "x11-xclip", "wayland-wl-copy", "wayland-xclip", "wayland-none"],
)
def test_doctor_checks_the_clipboard_tool_the_session_uses(
    tmp_path, monkeypatch, session: str, tools: set[str], ok: bool
) -> None:
    """On X11, doctor passed the clipboard check with only wl-copy, which
    can't reach an X11 clipboard; the backend then offered no clipboard."""
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", session)
    monkeypatch.setenv("DISPLAY", ":0")
    if session == "wayland":
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    else:
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(doctor.platform_info, "_x11_display_opens", lambda: True)
    import shutil

    # doctor imports shutil inside the check, so the module's own is patched.
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}" if name in tools else None)
    monkeypatch.setattr(
        "autogpt_local_executor.computer_use.get_backend",
        lambda _: MagicMock(display_info=lambda: []),
    )

    result = next(
        r
        for r in doctor._check_linux(_config(tmp_path, enable_clipboard=True), is_wsl2=False)
        if r.label == "clipboard tools"
    )

    assert (result.marker == "OK") is ok, result
