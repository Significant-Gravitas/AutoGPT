"""Tests for platform_info detection helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from autogpt_local_executor import platform_info


def test_detect_platform_darwin() -> None:
    with patch.object(sys, "platform", "darwin"):
        assert platform_info.detect_platform() == "darwin"


def test_detect_platform_windows() -> None:
    with patch.object(sys, "platform", "win32"):
        assert platform_info.detect_platform() == "windows"


def test_detect_platform_linux_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    fake_proc_version = "Linux version 6.1.0-12-amd64 (Debian)"
    m = patch("builtins.open", new=_fake_open({"/proc/version": fake_proc_version}))
    with m:
        assert platform_info.detect_platform() == "linux"


def test_detect_platform_wsl2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    fake_proc_version = "Linux version 5.15.0 Microsoft@WSL2 (x86_64-linux-gnu)"
    m = patch("builtins.open", new=_fake_open({"/proc/version": fake_proc_version}))
    with m:
        assert platform_info.detect_platform() == "wsl2"


def test_detect_arch_normalizes() -> None:
    with patch("platform.machine", return_value="AMD64"):
        assert platform_info.detect_arch() == "x86_64"
    with patch("platform.machine", return_value="aarch64"):
        assert platform_info.detect_arch() == "arm64"
    with patch("platform.machine", return_value="arm64"):
        assert platform_info.detect_arch() == "arm64"


def test_detect_arch_rejects_unknown() -> None:
    with patch("platform.machine", return_value="riscv64"):
        with pytest.raises(ValueError):
            platform_info.detect_arch()


def test_detect_capabilities_minimal() -> None:
    caps = platform_info.detect_capabilities()
    assert "shell" not in caps
    assert "files" in caps
    assert "computer_use" not in caps


def test_detect_capabilities_includes_shell_only_when_enabled() -> None:
    caps = platform_info.detect_capabilities(enable_shell=True)
    assert "shell" in caps
    assert "files" in caps


@pytest.mark.parametrize("platform_name", ["darwin", "linux", "windows", "wsl2"])
def test_recording_design_preview_is_never_advertised(platform_name: str) -> None:
    with patch.object(platform_info, "detect_platform", return_value=platform_name):
        assert platform_info.available_recording_channels() == []
        assert "recording" not in platform_info.detect_capabilities(enable_recording=True)


def test_default_allowed_root_per_os(monkeypatch: pytest.MonkeyPatch) -> None:
    with patch.object(platform_info, "detect_platform", return_value="darwin"):
        assert (
            platform_info.default_allowed_root() == Path.home() / "Documents" / "autogpt-workspace"
        )
    with patch.object(platform_info, "detect_platform", return_value="linux"):
        assert platform_info.default_allowed_root() == Path.home() / "autogpt-workspace"
    with patch.object(platform_info, "detect_platform", return_value="windows"):
        assert platform_info.default_allowed_root() == Path.home() / "autogpt-workspace"


def test_default_audit_log_path_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    with patch.object(platform_info, "detect_platform", return_value="linux"):
        p = platform_info.default_audit_log_path()
        assert p.as_posix().endswith(".local/state/autogpt-local-executor/audit.log")


def test_default_audit_log_path_macos() -> None:
    with patch.object(platform_info, "detect_platform", return_value="darwin"):
        p = platform_info.default_audit_log_path()
        assert "Library/Logs/autogpt-local-executor" in p.as_posix()


def test_default_audit_log_path_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\test\\AppData\\Local")
    with patch.object(platform_info, "detect_platform", return_value="windows"):
        p = platform_info.default_audit_log_path()
        assert "autogpt-local-executor" in str(p)
        assert "audit.log" in str(p)


@pytest.mark.parametrize("platform_name", ["darwin", "linux", "windows"])
def test_resolve_shell_unknown_returns_none(
    monkeypatch: pytest.MonkeyPatch, platform_name: str
) -> None:
    monkeypatch.setattr(platform_info, "detect_platform", lambda: platform_name)
    monkeypatch.setattr(platform_info.shutil, "which", lambda name: f"/bin/{name}")

    assert platform_info.resolve_shell("fish") is None


def test_powershell_resolves_only_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform_info.shutil, "which", lambda name: f"/bin/{name}")

    monkeypatch.setattr(platform_info, "detect_platform", lambda: "windows")
    assert platform_info.resolve_shell("powershell") == (
        "/bin/powershell.exe",
        ["-NoLogo", "-NoProfile", "-Command"],
    )
    monkeypatch.setattr(platform_info, "detect_platform", lambda: "linux")
    assert platform_info.resolve_shell("powershell") is None


def _fake_open(file_contents: dict[str, str]):
    """Create a mock for builtins.open that returns canned data for given paths."""
    import io

    real_open = open  # noqa: A001

    def _open(file, mode="r", *args, **kwargs):
        path_str = str(file)
        if path_str in file_contents and "b" not in mode:
            return io.StringIO(file_contents[path_str])
        return real_open(file, mode, *args, **kwargs)

    return _open


_SESSION_VARIABLES = (
    "DISPLAY",
    "WAYLAND_DISPLAY",
    "XDG_SESSION_TYPE",
    "CUA_DRIVER_RS_ENABLE_WAYLAND",
)


def _session(
    monkeypatch: pytest.MonkeyPatch,
    platform_name: str = "linux",
    *,
    driver: bool = True,
    **env: str,
) -> list:
    """A machine of `platform_name` with `env` set, and cua-driver installed
    or not. Returns the paths the driver lookup was asked for."""
    for name in _SESSION_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(platform_info, "detect_platform", lambda: platform_name)
    asked: list = []

    def find(explicit=None):
        asked.append(explicit)
        return Path("/opt/cua-driver") if driver else None

    from autogpt_local_executor.computer_use import driver as driver_module

    monkeypatch.setattr(driver_module, "find_driver_binary", find)
    return asked


@pytest.mark.parametrize(
    "env",
    [
        {"DISPLAY": ":0"},
        {"WAYLAND_DISPLAY": "wayland-0", "CUA_DRIVER_RS_ENABLE_WAYLAND": "1"},
        {"DISPLAY": ":0", "WAYLAND_DISPLAY": "w"},
    ],
    ids=["x11", "native-wayland", "xwayland"],
)
def test_linux_computer_use_is_advertised_with_a_display(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    """cua-driver drives X11, which XWayland provides on a Wayland desktop,
    and Wayland itself only with its experimental backend turned on."""
    _session(monkeypatch, **env)

    assert "computer_use" in platform_info.detect_capabilities(enable_computer_use=True)


@pytest.mark.parametrize(
    "env",
    [
        {},
        {"WAYLAND_DISPLAY": "wayland-0"},
        {"WAYLAND_DISPLAY": "wayland-0", "CUA_DRIVER_RS_ENABLE_WAYLAND": "0"},
        {"WAYLAND_DISPLAY": "wayland-0", "CUA_DRIVER_RS_ENABLE_WAYLAND": " FALSE "},
        {"CUA_DRIVER_RS_ENABLE_WAYLAND": "1"},
    ],
    ids=["none", "wayland-only", "opted-out-0", "opted-out-false", "opted-in-without-wayland"],
)
def test_linux_computer_use_needs_a_display_the_driver_can_drive(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str]
) -> None:
    """cua-driver 0.38 takes its X11 path unless CUA_DRIVER_RS_ENABLE_WAYLAND
    is set, to anything but empty, 0 or false, in a Wayland session. With
    WAYLAND_DISPLAY alone HELLO offered computer use, and then
    get_screen_size and every action failed for want of an X display."""
    _session(monkeypatch, **env)

    assert "computer_use" not in platform_info.detect_capabilities(enable_computer_use=True)


@pytest.mark.parametrize("platform_name", ["darwin", "linux", "windows"])
def test_computer_use_needs_cua_driver(monkeypatch: pytest.MonkeyPatch, platform_name: str) -> None:
    _session(monkeypatch, platform_name, driver=False, DISPLAY=":0")

    assert "computer_use" not in platform_info.detect_capabilities(enable_computer_use=True)


@pytest.mark.parametrize("platform_name", ["darwin", "windows"])
def test_computer_use_is_advertised_with_cua_driver(
    monkeypatch: pytest.MonkeyPatch, platform_name: str
) -> None:
    _session(monkeypatch, platform_name)

    assert "computer_use" in platform_info.detect_capabilities(enable_computer_use=True)


def test_wsl2_never_advertises_computer_use(monkeypatch: pytest.MonkeyPatch) -> None:
    _session(monkeypatch, "wsl2", DISPLAY=":0")

    assert "computer_use" not in platform_info.detect_capabilities(enable_computer_use=True)


def test_the_configured_driver_path_is_the_one_looked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _session(monkeypatch, "darwin")

    platform_info.detect_capabilities(enable_computer_use=True, cua_driver_path="/x/cua-driver")

    assert asked == ["/x/cua-driver"]
