"""Tests for `autogpt-shim doctor` and the daemon's computer-use preflight."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from autogpt_local_executor import doctor
from autogpt_local_executor.audit import AuditWriter
from autogpt_local_executor.config import ShimConfig
from autogpt_local_executor.daemon import DaemonPreflightError
from autogpt_local_executor.protocol import ErrorCode
from tests.cua_fakes import FAKE, fake_bridge, fake_log

__all__ = ["fake_log"]  # a fixture, used by name


def _config(tmp_path: Path, **overrides) -> ShimConfig:
    return ShimConfig(
        allowed_root=tmp_path,
        audit_log_path=tmp_path / "_audit.log",
        platform_url="http://localhost:9999",
        machine_id="test-machine",
        **overrides,
    )


def test_doctor_returns_zero_when_computer_use_not_requested(tmp_path: Path, capsys) -> None:
    config = _config(tmp_path, enable_computer_use=False)

    assert doctor.run_doctor(config) == 0
    assert "not requested in config" in capsys.readouterr().out


def test_control_doctor_checks_browser_without_creating_legacy_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_root = tmp_path / "must-not-be-created"
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    config = _config(tmp_path, session_id=None)
    config.allowed_root = missing_root

    assert doctor.run_doctor(config) == 0
    assert not missing_root.exists()


# ── Computer use ─────────────────────────────────────────────────────────


def _driver_report(version: str = "0.38.0", probes: list[dict] | None = None) -> str:
    return json.dumps(
        {
            "build": {"version": version},
            "ok": True,
            "probes": probes
            if probes is not None
            else [
                {"label": "binary", "message": f"cua-driver {version}", "status": "ok"},
                {"label": "AT-SPI", "message": "org.a11y.Bus reachable", "status": "ok"},
            ],
        }
    )


class _Commands(list):
    """The cua-driver commands doctor ran, and the report `doctor --json`
    answers with."""

    def __init__(self) -> None:
        super().__init__()
        self.state: dict[str, Any] = {"report": _driver_report()}


@pytest.fixture
def driver_commands(monkeypatch: pytest.MonkeyPatch) -> _Commands:
    ran = _Commands()

    def run(argv, **_kwargs):
        ran.append(list(argv[1:]))
        stdout = ran.state["report"] if argv[1:] == ["doctor", "--json"] else ""
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(doctor.subprocess, "run", run)
    return ran


@pytest.fixture
def a_fake_driver(monkeypatch: pytest.MonkeyPatch, fake_log: Path):
    """doctor's driver probes run the fake, as if on `platform`."""
    settings: dict[str, Any] = {"platform": "linux"}

    def bridge(*, tools, binary):
        return fake_bridge(platform=settings["platform"])

    monkeypatch.setattr(doctor, "CuaDriverBridge", bridge)
    return settings


def _report(out: str) -> dict[str, tuple[str, str]]:
    """doctor's output as {label: (marker, detail)}."""
    lines = {}
    for line in out.splitlines():
        match = re.match(r"\[(\w+)\s*\] (.+?)\s{2,}(.*)$", line)
        if match:
            lines[match.group(2)] = (match.group(1), match.group(3))
    return lines


def _with_driver(tmp_path: Path, plat: str = "linux", **overrides) -> ShimConfig:
    return _config(tmp_path, enable_computer_use=True, cua_driver_path=str(FAKE), **overrides)


def test_doctor_fails_without_cua_driver(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    monkeypatch.setattr(doctor, "find_driver_binary", lambda _path=None: None)

    assert doctor.run_doctor(_config(tmp_path, enable_computer_use=True)) == doctor.EX_CONFIG
    assert "computer-use]" in capsys.readouterr().out


def test_doctor_runs_the_drivers_own_probes_and_starts_it(
    tmp_path, monkeypatch, capsys, driver_commands, a_fake_driver
) -> None:
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    driver_commands.state["report"] = _driver_report(
        probes=[
            {"label": "binary", "message": "cua-driver 0.38.0", "status": "ok"},
            {
                "label": "display server",
                "message": "neither DISPLAY nor WAYLAND_DISPLAY set",
                "detail": "run from a desktop session",
                "status": "warn",
            },
            {"label": "AT-SPI", "message": "org.a11y.Bus reachable", "status": "ok"},
        ]
    )

    rc = doctor.run_doctor(_with_driver(tmp_path))

    lines = _report(capsys.readouterr().out)
    assert rc == 0
    assert driver_commands[0] == ["doctor", "--json"]
    assert lines["cua-driver display server"] == (
        "WARN",
        "neither DISPLAY nor WAYLAND_DISPLAY set (run from a desktop session)",
    )
    assert lines["cua-driver AT-SPI"] == ("OK", "org.a11y.Bus reachable")
    assert "cua-driver binary" not in lines
    assert lines["cua-driver start"] == ("OK", "0.38.0, 38 tools offered")
    assert lines["display"] == ("OK", "1920x1080 (scale 1)")


def test_doctor_fails_on_a_driver_probe_error(
    tmp_path, monkeypatch, driver_commands, a_fake_driver
) -> None:
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    driver_commands.state["report"] = _driver_report(
        probes=[{"label": "AT-SPI", "message": "no session bus", "status": "error"}]
    )

    assert doctor.run_doctor(_with_driver(tmp_path)) == doctor.EX_CONFIG


def test_doctor_fails_on_another_driver_version(
    tmp_path, monkeypatch, capsys, driver_commands, a_fake_driver
) -> None:
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    driver_commands.state["report"] = _driver_report(version="0.40.1")

    assert doctor.run_doctor(_with_driver(tmp_path)) == doctor.EX_CONFIG
    assert "0.40.1" in capsys.readouterr().out


@pytest.mark.skipif(os.name == "nt", reason="POSIX execute bit")
def test_doctor_runs_a_driver_installed_without_its_execute_bit(tmp_path) -> None:
    """pip can install the wheel's binary without its execute bit. The daemon
    sets it before starting the driver; doctor ran the binary first and
    reported Permission denied."""
    binary = tmp_path / "cua-driver"
    binary.write_text(f"#!/bin/sh\necho '{_driver_report(probes=[])}'\n")
    binary.chmod(0o644)

    [check] = doctor._driver_doctor(binary)

    assert (check.marker, check.label) == ("OK", "cua-driver")


@pytest.mark.parametrize(
    "report",
    [
        "[]",
        "null",
        '"0.38.0"',
        '{"build": "0.38.0", "probes": []}',
        '{"build": {"version": "0.38.0"}, "probes": {"label": "AT-SPI"}}',
        '{"build": {"version": "0.38.0"}, "probes": ["AT-SPI"]}',
    ],
    ids=["array", "null", "string", "build-a-string", "probes-an-object", "probe-a-string"],
)
def test_doctor_fails_on_a_driver_report_it_cannot_read(
    tmp_path, driver_commands, report: str
) -> None:
    """JSON that isn't shaped like the report (an array, or a build or probe
    that isn't an object) ended doctor in an AttributeError."""
    driver_commands.state["report"] = report

    checks = doctor._driver_doctor(FAKE)

    assert any(check.marker == "FAIL" and check.blocking for check in checks)


def test_doctor_keeps_the_probes_it_can_read(tmp_path, driver_commands) -> None:
    driver_commands.state["report"] = json.dumps(
        {
            "build": {"version": "0.38.0"},
            "probes": ["AT-SPI", {"label": "AT-SPI", "message": "reachable", "status": "ok"}],
        }
    )

    checks = {check.label: check.marker for check in doctor._driver_doctor(FAKE)}

    assert checks["cua-driver AT-SPI"] == "OK"
    assert checks["cua-driver probes"] == "FAIL"


def test_doctor_reports_a_driver_error_instead_of_crashing(
    tmp_path, monkeypatch, capsys, driver_commands, a_fake_driver
) -> None:
    """The driver can answer a probe with a JSON-RPC error, which used to
    end doctor in a traceback."""
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "darwin")
    a_fake_driver["platform"] = "darwin"
    monkeypatch.setenv("FAKE_CUA_RPC_ERROR", "get_screen_size,check_permissions")
    monkeypatch.setattr(doctor, "_request_macos_grants", lambda _missing: None)

    rc = doctor.run_doctor(_with_driver(tmp_path))

    lines = _report(capsys.readouterr().out)
    assert rc == doctor.EX_CONFIG
    assert lines["display"][0] == "WARN"
    assert "get_screen_size broke" in lines["display"][1]
    assert lines["permissions"][0] == "FAIL"


def test_doctor_blocks_computer_use_on_wsl2(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "wsl2")

    assert doctor.run_doctor(_with_driver(tmp_path)) == doctor.EX_CONFIG


def test_doctor_on_macos_reports_missing_grants_and_requests_them(
    tmp_path, monkeypatch, capsys, driver_commands, a_fake_driver
) -> None:
    """The driver is the shim's child and leaves prompting to its host, so
    doctor asks macOS for the shim's app, as it did before cua-driver."""
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "darwin")
    a_fake_driver["platform"] = "darwin"
    monkeypatch.setenv("FAKE_CUA_PERMISSIONS", '{"accessibility": true, "screen_recording": false}')
    requested: list[list[str]] = []
    monkeypatch.setattr(doctor, "_request_macos_grants", requested.append)

    rc = doctor.run_doctor(_with_driver(tmp_path))

    lines = _report(capsys.readouterr().out)
    assert rc == doctor.EX_CONFIG
    assert lines["Accessibility"] == ("OK", "granted")
    assert lines["Screen Recording"][0] == "FAIL"
    assert "the app that runs the local executor" in lines["Screen Recording"][1]
    assert requested == [["screen_recording"]]
    assert ["permissions", "grant"] not in driver_commands


def test_doctor_on_macos_passes_with_both_grants(
    tmp_path, monkeypatch, driver_commands, a_fake_driver
) -> None:
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "darwin")
    a_fake_driver["platform"] = "darwin"
    requested: list[list[str]] = []
    monkeypatch.setattr(doctor, "_request_macos_grants", requested.append)

    assert doctor.run_doctor(_with_driver(tmp_path)) == 0
    assert requested == []


@pytest.mark.parametrize(("present", "marker"), [(True, "OK"), (False, "WARN")])
def test_doctor_checks_the_tool_foreign_clipboard_reads_need(
    monkeypatch, present: bool, marker: str
) -> None:
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(doctor.shutil, "which", lambda name: name if present else None)

    result = doctor._check_linux_clipboard_types()

    assert result.marker == marker
    assert "xclip" in result.detail


# ── Daemon preflight ─────────────────────────────────────────────────────


def _daemon(config: ShimConfig):
    from autogpt_local_executor.daemon import ShimDaemon

    audit = AuditWriter(path=config.audit_log_path, audit_key=b"a" * 32)
    daemon = ShimDaemon(config=config, token_store=MagicMock(), audit=audit)
    daemon._computer_handler._driver = fake_bridge(platform="darwin")
    return daemon


@pytest.mark.asyncio
async def test_daemon_preflight_passes_when_computer_use_off(tmp_path) -> None:
    config = _config(tmp_path, enable_computer_use=False)
    await _daemon(config)._preflight_or_raise()


async def test_daemon_preflight_stops_for_good_without_cua_driver(tmp_path, monkeypatch) -> None:
    config = _config(tmp_path, enable_computer_use=True, cua_driver_path=str(tmp_path / "x"))

    with pytest.raises(DaemonPreflightError) as raised:
        await _daemon(config)._preflight_or_raise()

    assert raised.value.permanent is True
    assert "cua_driver" in (tmp_path / "_audit.log").read_text()


async def test_daemon_preflight_needs_no_grants_off_macos(tmp_path, monkeypatch, fake_log) -> None:
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "linux")
    daemon = _daemon(_with_driver(tmp_path))

    await daemon._preflight_or_raise()

    assert daemon._computer_handler.driver.info is None  # never started


@pytest.mark.parametrize(
    ("grants", "missing"),
    [
        ({"accessibility": False, "screen_recording": True}, ["accessibility"]),
        ({"accessibility": True, "screen_recording": False}, ["screen_recording"]),
        ({}, ["accessibility", "screen_recording"]),
    ],
)
async def test_daemon_preflight_on_macos_needs_both_grants(
    tmp_path, monkeypatch, fake_log, grants: dict, missing: list[str]
) -> None:
    """Without Screen Recording a capture shows only the wallpaper and menu
    bar; without Accessibility there is no input."""
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "darwin")
    monkeypatch.setenv("FAKE_CUA_PERMISSIONS", json.dumps(grants))
    daemon = _daemon(_with_driver(tmp_path))
    try:
        with pytest.raises(DaemonPreflightError) as raised:
            await daemon._preflight_or_raise()
    finally:
        await daemon._computer_handler.close()

    assert raised.value.permanent is False
    record = next(
        json.loads(line)
        for line in (tmp_path / "_audit.log").read_text().splitlines()
        if "DAEMON_PREFLIGHT_FAILED" in line
    )
    assert record["details"]["missing_permissions"] == missing
    assert record["result"]["error_code"] == ErrorCode.PERMISSION_PENDING.value


async def test_daemon_preflight_on_macos_fails_cleanly_on_a_driver_error(
    tmp_path, monkeypatch, fake_log
) -> None:
    """A JSON-RPC error from the permission check left `start` with a
    traceback, which launchd restarts, instead of the preflight's exit."""
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "darwin")
    monkeypatch.setenv("FAKE_CUA_RPC_ERROR", "check_permissions")
    daemon = _daemon(_with_driver(tmp_path))
    try:
        with pytest.raises(DaemonPreflightError) as raised:
            await daemon._preflight_or_raise()
    finally:
        await daemon._computer_handler.close()

    assert raised.value.permanent is False
    record = next(
        json.loads(line)
        for line in (tmp_path / "_audit.log").read_text().splitlines()
        if "DAEMON_PREFLIGHT_FAILED" in line
    )
    assert record["result"]["error_code"] == ErrorCode.INTERNAL_ERROR.value


async def test_daemon_preflight_on_macos_passes_with_both_grants(
    tmp_path, monkeypatch, fake_log
) -> None:
    monkeypatch.setattr(doctor.platform_info, "detect_platform", lambda: "darwin")
    daemon = _daemon(_with_driver(tmp_path))
    try:
        await daemon._preflight_or_raise()
    finally:
        await daemon._computer_handler.close()


# ── Keychain ─────────────────────────────────────────────────────────────


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
