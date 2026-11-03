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


def test_resolve_shell_unknown_returns_none() -> None:
    assert platform_info.resolve_shell("powershell") is None or sys.platform == "win32"


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
