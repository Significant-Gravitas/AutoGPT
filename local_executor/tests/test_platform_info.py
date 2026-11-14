"""Tests for platform_info detection helpers."""

from __future__ import annotations

import concurrent.futures
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any
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
    "AUTOGPT_SHIM_SUPERVISOR",
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
    for name in dict.fromkeys(_SESSION_VARIABLES + platform_info.SESSION_DISPLAY_VARIABLES):
        # Set first, so the test's end restores the variable even when it
        # wasn't set: delenv records nothing for one that isn't there, and a
        # SessionDisplay writes os.environ itself, outside monkeypatch.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
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


class _Manager:
    """The systemd user manager's environment, as `systemctl --user
    show-environment` would give it, and how often it was read."""

    def __init__(self, **env: str) -> None:
        self.env: dict[str, str] | None = dict(env)
        self.reads = 0

    def __call__(self) -> dict[str, str] | None:
        self.reads += 1
        return None if self.env is None else dict(self.env)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _follower(environ: dict[str, str], manager: _Manager, clock: _Clock | None = None):
    return platform_info.SessionDisplay(environ, read=manager, clock=clock or _Clock())


def test_a_service_started_before_the_desktop_takes_its_display() -> None:
    """systemd starts the service at login, before GNOME or KDE export the
    display to the manager, so the shim had no DISPLAY for as long as it ran,
    and computer use stayed off until it restarted."""
    environ = {"AUTOGPT_SHIM_SUPERVISOR": "systemd", "PATH": "/usr/bin"}
    manager = _Manager(
        DISPLAY=":0",
        XAUTHORITY="/run/user/1000/.mutter-Xwaylandauth.ABC",
        XDG_SESSION_TYPE="wayland",
        WAYLAND_DISPLAY="wayland-0",
        PATH="/elsewhere",
        LD_PRELOAD="/tmp/evil.so",
    )

    assert _follower(environ, manager).follow() is True

    assert environ == {
        "AUTOGPT_SHIM_SUPERVISOR": "systemd",
        "PATH": "/usr/bin",
        "DISPLAY": ":0",
        "XAUTHORITY": "/run/user/1000/.mutter-Xwaylandauth.ABC",
        "XDG_SESSION_TYPE": "wayland",
        "WAYLAND_DISPLAY": "wayland-0",
    }
    assert platform_info.linux_display_problem(environ) is None


def test_the_display_follows_a_new_login_and_goes_with_the_session() -> None:
    environ = {"AUTOGPT_SHIM_SUPERVISOR": "systemd"}
    manager = _Manager(DISPLAY=":0", XAUTHORITY="/run/user/1000/xauth_A")
    clock = _Clock()
    follower = _follower(environ, manager, clock)
    follower.follow()

    manager.env = {"DISPLAY": ":1", "XAUTHORITY": "/run/user/1000/xauth_B"}
    clock.now += 11
    assert follower.follow() is True
    assert (environ["DISPLAY"], environ["XAUTHORITY"]) == (":1", "/run/user/1000/xauth_B")

    manager.env = {"XAUTHORITY": "/run/user/1000/xauth_B"}
    clock.now += 11
    assert follower.follow() is True
    assert environ == {"AUTOGPT_SHIM_SUPERVISOR": "systemd"}


def test_the_sessions_xauthority_replaces_one_the_service_started_with() -> None:
    """A service that started without a display but with an XAUTHORITY (one
    an earlier session left in the manager, say) kept it, and took the new
    session's DISPLAY without the authority file that opens it, so computer
    use stayed off. The session's variables go with its display, and the
    service's own come back once the session ends."""
    environ = {
        "AUTOGPT_SHIM_SUPERVISOR": "systemd",
        "XAUTHORITY": "/run/user/1000/xauth_old",
        "XDG_CURRENT_DESKTOP": "sway",
    }
    manager = _Manager(DISPLAY=":0", XAUTHORITY="/run/user/1000/xauth_new", XDG_SESSION_TYPE="x11")
    clock = _Clock()
    follower = _follower(environ, manager, clock)

    assert follower.follow() is True
    assert environ == {
        "AUTOGPT_SHIM_SUPERVISOR": "systemd",
        "DISPLAY": ":0",
        "XAUTHORITY": "/run/user/1000/xauth_new",
        "XDG_SESSION_TYPE": "x11",
        # The manager has none, so the service's own stands.
        "XDG_CURRENT_DESKTOP": "sway",
    }
    assert follower.following() is True

    manager.env = {}
    clock.now += 11
    assert follower.follow() is True
    assert environ == {
        "AUTOGPT_SHIM_SUPERVISOR": "systemd",
        "XAUTHORITY": "/run/user/1000/xauth_old",
        "XDG_CURRENT_DESKTOP": "sway",
    }


def test_a_session_without_an_xauthority_clears_the_one_the_service_started_with() -> None:
    """A session that opens its display without an XAUTHORITY, for Xlib to
    read ~/.Xauthority, left the one the service started with in place (one
    an earlier session left in the manager, say), and Xlib read that file
    instead, which doesn't open this display. While such a session lasts
    there is none, after a session that had one too, and the service's own
    comes back once it ends."""
    environ = {"AUTOGPT_SHIM_SUPERVISOR": "systemd", "XAUTHORITY": "/run/user/1000/xauth_old"}
    manager = _Manager(DISPLAY=":0")
    clock = _Clock()
    follower = _follower(environ, manager, clock)

    assert follower.follow() is True
    assert environ == {"AUTOGPT_SHIM_SUPERVISOR": "systemd", "DISPLAY": ":0"}

    manager.env = {"DISPLAY": ":1", "XAUTHORITY": "/run/user/1000/xauth_new"}
    clock.now += 11
    assert follower.follow() is True
    manager.env = {"DISPLAY": ":2"}
    clock.now += 11
    assert follower.follow() is True
    assert environ == {"AUTOGPT_SHIM_SUPERVISOR": "systemd", "DISPLAY": ":2"}

    manager.env = {}
    clock.now += 11
    assert follower.follow() is True
    assert environ == {
        "AUTOGPT_SHIM_SUPERVISOR": "systemd",
        "XAUTHORITY": "/run/user/1000/xauth_old",
    }


def test_the_manager_is_read_at_most_every_ten_seconds() -> None:
    manager = _Manager(DISPLAY=":0")
    clock = _Clock()
    follower = _follower({"AUTOGPT_SHIM_SUPERVISOR": "systemd"}, manager, clock)

    follower.follow()
    clock.now += 5
    follower.follow()
    clock.now += 6
    follower.follow()

    assert manager.reads == 2


@pytest.mark.parametrize(
    "environ",
    [
        {"DISPLAY": ":5"},
        {"AUTOGPT_SHIM_SUPERVISOR": "systemd", "DISPLAY": ":5"},
        {"AUTOGPT_SHIM_SUPERVISOR": "systemd", "WAYLAND_DISPLAY": "wayland-5"},
        {"AUTOGPT_SHIM_SUPERVISOR": "launchd"},
    ],
    ids=["not-a-service", "own-display", "own-wayland", "launchd"],
)
def test_a_shim_keeps_its_own_environment(environ: dict[str, str]) -> None:
    """Only the systemd service, started without a display, takes the
    manager's: a shim run by hand keeps the display it was given, or none."""
    before = dict(environ)
    manager = _Manager(DISPLAY=":0", XAUTHORITY="/x")

    assert _follower(environ, manager).follow() is False

    assert environ == before
    assert manager.reads == 0


@pytest.mark.parametrize(
    "manager",
    [
        _Manager(DISPLAY=":0\x00"),
        _Manager(DISPLAY=":0\n1"),
        _Manager(XAUTHORITY="/run/user/1000/xauth"),
    ],
    ids=["nul", "newline", "no-display"],
)
def test_nothing_is_taken_without_a_usable_display(manager: _Manager) -> None:
    environ = {"AUTOGPT_SHIM_SUPERVISOR": "systemd"}

    assert _follower(environ, manager).follow() is False

    assert environ == {"AUTOGPT_SHIM_SUPERVISOR": "systemd"}


def test_a_manager_that_cannot_be_read_changes_nothing() -> None:
    environ = {"AUTOGPT_SHIM_SUPERVISOR": "systemd"}
    manager = _Manager()
    manager.env = None

    assert _follower(environ, manager).follow() is False
    assert environ == {"AUTOGPT_SHIM_SUPERVISOR": "systemd"}


class _HeldManager(_Manager):
    """A read of the manager's environment that hangs until `go` is set, as
    systemctl's lookup on a stalled PATH mount (NFS, FUSE) does."""

    def __init__(self, **env: str) -> None:
        super().__init__(**env)
        self.reading = threading.Event()
        self.go = threading.Event()

    def __call__(self) -> dict[str, str] | None:
        self.reads += 1
        self.reading.set()
        self.go.wait(60)
        return None if self.env is None else dict(self.env)


def _answer_within(seconds: float, call: Callable[[], Any]) -> Any:
    """What `call()` returns, called in a thread of its own; TimeoutError
    when it hasn't returned within `seconds`."""
    answer: concurrent.futures.Future[Any] = concurrent.futures.Future()

    def run() -> None:
        try:
            answer.set_result(call())
        except BaseException as exc:
            answer.set_exception(exc)

    threading.Thread(target=run, daemon=True).start()
    return answer.result(seconds)


def test_a_manager_read_that_hangs_holds_no_caller_past_its_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manager was read under the lock every caller takes, systemctl's
    lookup on PATH included, and a stalled mount there holds that lookup for
    as long as it stalls. HELLO's look at whether computer use waits for a
    display (following) waited for the lock, and so did every later caller,
    each in a thread of its own. The read runs apart from the lock now, one
    at a time: a caller waits for the one under way within its wait, and
    goes on without it."""
    monkeypatch.setattr(platform_info, "MANAGER_WAIT_SECONDS", 0.2, raising=False)
    environ = {"AUTOGPT_SHIM_SUPERVISOR": "systemd"}
    manager = _HeldManager(DISPLAY=":0")
    clock = _Clock()
    follower = _follower(environ, manager, clock)
    try:
        threading.Thread(target=follower.follow, daemon=True).start()
        assert manager.reading.wait(20)

        assert _answer_within(20, follower.following) is True
        # Even once the next read is due, the one under way is waited for.
        clock.now += 11
        assert _answer_within(20, follower.follow) is False

        assert manager.reads == 1
        assert "DISPLAY" not in environ
    finally:
        manager.go.set()


def test_hello_goes_on_while_the_manager_read_hangs(monkeypatch: pytest.MonkeyPatch) -> None:
    """HELLO's capabilities (can_do_computer_use) waited for the manager's
    read to the end, so a read that hung kept HELLO from going out at all.
    It goes without computer use now, and says HELLO again once the display
    is up."""
    _session(monkeypatch, AUTOGPT_SHIM_SUPERVISOR="systemd")
    monkeypatch.setattr(platform_info, "MANAGER_WAIT_SECONDS", 0.2, raising=False)
    # No display to take: the read, let go at the end, writes os.environ,
    # and may finish after the test has put it back.
    manager = _HeldManager()
    monkeypatch.setattr(
        platform_info, "session_display", platform_info.SessionDisplay(read=manager)
    )
    try:
        caps = _answer_within(
            20, lambda: platform_info.detect_capabilities(enable_computer_use=True)
        )

        assert "computer_use" not in caps
        assert _answer_within(20, platform_info.computer_use_waits_for_a_display) is True
    finally:
        manager.go.set()


def test_systemctl_is_looked_up_on_path_until_it_is_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """shutil.which stats a file in each PATH folder, and every read of the
    manager looked systemctl up again, where a stalled mount holds the
    lookup for as long as it stalls. The service's PATH stays as it
    started, so the path is kept once found."""
    lookups: list[str] = []

    def which(name: str) -> str | None:
        lookups.append(name)
        return None if len(lookups) == 1 else "/usr/bin/systemctl"

    def run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, b"DISPLAY=:0\n", b"")

    monkeypatch.setattr(platform_info, "_systemctl_path", None, raising=False)
    monkeypatch.setattr(platform_info.shutil, "which", which)
    monkeypatch.setattr(platform_info.subprocess, "run", run)

    assert platform_info._manager_environment() is None
    assert platform_info._manager_environment() == {"DISPLAY": ":0"}
    assert platform_info._manager_environment() == {"DISPLAY": ":0"}
    assert lookups == ["systemctl", "systemctl"]


def test_the_systemd_service_advertises_computer_use_once_the_desktop_exports_its_display(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _session(monkeypatch, AUTOGPT_SHIM_SUPERVISOR="systemd")
    monkeypatch.setattr(
        platform_info,
        "session_display",
        platform_info.SessionDisplay(read=_Manager(DISPLAY=":0")),
    )

    assert "computer_use" in platform_info.detect_capabilities(enable_computer_use=True)


@pytest.mark.parametrize(
    ("platform_name", "env", "waits"),
    [
        ("linux", {"AUTOGPT_SHIM_SUPERVISOR": "systemd"}, True),
        ("linux", {"AUTOGPT_SHIM_SUPERVISOR": "systemd", "DISPLAY": ":0"}, False),
        ("linux", {}, False),  # run by hand: its environment is its own
        ("darwin", {"AUTOGPT_SHIM_SUPERVISOR": "launchd"}, False),
    ],
    ids=["service-before-the-desktop", "service-with-a-display", "by-hand", "macos"],
)
def test_computer_use_waits_for_a_display_only_where_one_can_come(
    monkeypatch: pytest.MonkeyPatch, platform_name: str, env: dict[str, str], waits: bool
) -> None:
    """Only the systemd service, started without a display, gets one later
    (from the user manager), so only its HELLO is said again for it."""
    _session(monkeypatch, platform_name, **env)
    monkeypatch.setattr(
        platform_info, "session_display", platform_info.SessionDisplay(read=_Manager())
    )

    assert platform_info.computer_use_waits_for_a_display() is waits


def test_reading_the_manager_environment_parses_systemctl(monkeypatch: pytest.MonkeyPatch) -> None:
    """systemctl quotes a value holding spaces or shell characters as $'...',
    and those were dropped: an XAUTHORITY under a home folder with a space
    in its name went missing, so the driver had a DISPLAY it couldn't use.
    They are decoded now, without a shell. An older systemctl prints such a
    value bare."""
    printed = (
        b"DISPLAY=:0\n"
        b"XAUTHORITY=$'/home/first last/.Xauthority'\n"
        b"QUOTES=$'it\\'s a \\\\ and a \"'\n"
        b"BYTES=$'caf\\303\\251\\ttab'\n"
        b"OLDER=/run/user/1000/a b\n"
        b"UNFINISHED=$'x y\n"
        b"UNKNOWN=$'\\q'\n"
        b"notavariable\n"
    )
    monkeypatch.setattr(platform_info, "_systemctl_path", None, raising=False)
    monkeypatch.setattr(platform_info.shutil, "which", lambda name: "/usr/bin/systemctl")

    def run(argv, **kwargs):
        assert argv == ["/usr/bin/systemctl", "--user", "show-environment"]
        assert kwargs.get("timeout")
        return subprocess.CompletedProcess(argv, 0, printed, b"")

    monkeypatch.setattr(platform_info.subprocess, "run", run)

    assert platform_info._manager_environment() == {
        "DISPLAY": ":0",
        "XAUTHORITY": "/home/first last/.Xauthority",
        "QUOTES": "it's a \\ and a \"",
        "BYTES": "café\ttab",
        "OLDER": "/run/user/1000/a b",
    }


def test_wsl2_never_advertises_computer_use(monkeypatch: pytest.MonkeyPatch) -> None:
    _session(monkeypatch, "wsl2", DISPLAY=":0")

    assert "computer_use" not in platform_info.detect_capabilities(enable_computer_use=True)


def test_the_configured_driver_path_is_the_one_looked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asked = _session(monkeypatch, "darwin")

    platform_info.detect_capabilities(enable_computer_use=True, cua_driver_path="/x/cua-driver")

    assert asked == ["/x/cua-driver"]
