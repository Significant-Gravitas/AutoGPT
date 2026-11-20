"""`autogpt-shim uninstall` stops the service its autostart entry started.

It removed the entry's file first and then printed a command to stop the
service: `launchctl unload <plist>`, which reads the file just removed, or
`systemctl --user disable --now`, which fails once the unit file is gone. So
the shim ran on, connected, until the person logged out.

Here launchctl, systemctl and schtasks are stand-ins that answer as the real
ones do; the tests at the end run the real ones, on a service of their own.
"""

from __future__ import annotations

import os
import plistlib
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from autogpt_local_executor import cli
from autogpt_local_executor import install as install_mod

# The real command runner, kept before conftest.py stands a refusing one in
# for it: only the tests at the end, on services of their own, may use it.
_REAL_RUN_COMMAND = getattr(install_mod, "run_command", None)

LABEL = "co.agpt.shim"
UNIT = "autogpt-shim.service"
TASK = r"\AutoGPT\Shim"
PLATFORMS = ("darwin", "linux", "wsl2", "windows")


def _result(returncode: int | None, output: str = "", failure: str | None = None) -> Any:
    return install_mod.CommandResult(returncode, output, failure)


class _Manager:
    """A stand-in for the service manager's command-line tool. `answer`
    gives each command line's result. `calls` keeps every command line with
    whether the autostart entry's file was still there when it ran."""

    def __init__(self, target: Path, answer: Callable[[list[str]], Any]) -> None:
        self.target = target
        self.answer = answer
        self.calls: list[tuple[str, bool]] = []

    def __call__(self, argv: Sequence[str], timeout: float) -> Any:
        assert timeout > 0
        if argv[-1].endswith(f"/{install_mod.FORMER_LAUNCHD_LABEL}"):
            # What `uninstall` asks launchd about the LaunchAgent's earlier
            # label is tests/test_launchd_label.py's: here there is none.
            assert list(argv[:2]) == ["launchctl", "print"], argv
            return _result(113, "Could not find service\n")
        self.calls.append((" ".join(argv), self.target.exists()))
        return self.answer(list(argv))

    @property
    def ran(self) -> list[str]:
        return [line for line, _ in self.calls]

    def file_was_there_for(self, start: str) -> list[bool]:
        """Whether the entry's file was there for each command line that
        starts with `start`."""
        return [there for line, there in self.calls if line.startswith(start)]


class _Launchd:
    """launchd, as `launchctl print` and `launchctl bootout` show it:
    `loaded` holds the domains the label is loaded in. With `slow`, a
    service takes that many looks to go, and bootout reports it as launchd
    does (36, in progress); with `stuck`, it never goes."""

    def __init__(self, *loaded: str, slow: int = 0, stuck: bool = False) -> None:
        self.loaded = set(loaded)
        self.slow = slow
        self.stuck = stuck
        self.going: dict[str, int] = {}

    def __call__(self, argv: list[str]) -> Any:
        tool, verb, target = argv
        assert tool == "launchctl"
        domain, _, label = target.rpartition("/")
        assert label == LABEL
        if verb == "print":
            if domain in self.going:
                self.going[domain] -= 1
                if self.going[domain] <= 0 and not self.stuck:
                    del self.going[domain]
                    self.loaded.discard(domain)
            if domain in self.loaded:
                return _result(0, f"{target} = {{\n\tstate = running\n}}\n")
            return _result(113, f'Could not find service "{label}" in domain for {domain}\n')
        assert verb == "bootout"
        if domain not in self.loaded:
            return _result(3, "Boot-out failed: 3: No such process\n")
        if self.slow or self.stuck:
            self.going[domain] = self.slow
            return _result(36, "Boot-out failed: 36: Operation now in progress\n")
        self.loaded.discard(domain)
        return _result(0)


class _Systemd:
    """systemd's user manager, as `systemctl --user` shows it. As the real
    one does, `disable` fails once the unit file is gone, and `stop` works
    without it for a unit that is loaded."""

    def __init__(
        self, unit_file: Path, *, active: str = "inactive", enabled: bool = False, bus: bool = True
    ) -> None:
        self.unit_file = unit_file
        self.active = active
        self.enabled = enabled
        self.bus = bus
        self.reloads = 0

    def __call__(self, argv: list[str]) -> Any:
        assert argv[:2] == ["systemctl", "--user"]
        if not self.bus:
            return _result(1, "Failed to connect to bus: No medium found\n")
        verb, rest = argv[2], argv[3:]
        if verb == "daemon-reload":
            self.reloads += 1
            return _result(0)
        assert rest[-1] == UNIT
        if verb == "show":
            assert rest == ["--property=ActiveState", UNIT]
            return _result(0, f"ActiveState={self.active}\n")
        if verb == "disable":
            if not self.unit_file.exists():
                return _result(1, f"Failed to disable unit: Unit file {UNIT} does not exist.\n")
            self.enabled = False
            if "--now" in rest:
                self.active = "inactive"
            return _result(0, f'Removed "{self.unit_file.parent}/default.target.wants/{UNIT}".\n')
        assert verb == "stop"
        if self.active == "inactive":
            return _result(5, f"Failed to stop {UNIT}: Unit {UNIT} not loaded.\n")
        self.active = "inactive"
        return _result(0)


class _TaskScheduler:
    """Task Scheduler, as schtasks shows it. Every failure is exit status 1,
    told apart only by text in the system's language."""

    def __init__(
        self,
        *,
        registered: bool = False,
        running: bool = False,
        visible: bool = True,
        may_end: bool = True,
        may_delete: bool = True,
    ) -> None:
        self.registered = registered
        self.running = running
        self.visible = visible
        self.may_end = may_end
        self.may_delete = may_delete

    def __call__(self, argv: list[str]) -> Any:
        tool, verb, flag, task, *rest = argv
        assert (tool, flag, task) == ("schtasks", "/TN", TASK)
        if not (self.registered and self.visible):
            return _result(1, "ERROR: The system cannot find the file specified.\n")
        if verb == "/Query":
            return _result(0, f'"{TASK}","N/A","Running"\n')
        if verb == "/End":
            if not self.may_end:
                return _result(1, "ERROR: Access is denied.\n")
            self.running = False
            return _result(0, f'SUCCESS: The scheduled task "{TASK}" has been terminated.\n')
        assert (verb, rest) == ("/Delete", ["/F"])
        if not self.may_delete:
            return _result(1, "ERROR: Access is denied.\n")
        self.registered = False
        return _result(0, f'SUCCESS: The scheduled task "{TASK}" was successfully deleted.\n')


class _Clock:
    """Time that passes only when the code under test sleeps."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.now = 1000.0
        self.slept: list[float] = []
        monkeypatch.setattr(install_mod, "_monotonic", lambda: self.now, raising=False)
        monkeypatch.setattr(install_mod, "_sleep", self._sleep, raising=False)

    def _sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A home folder of the test's own, for every OS's autostart entry."""
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    monkeypatch.setattr(install_mod, "default_stdout_log", lambda: tmp_path / "logs" / "out.log")
    monkeypatch.setattr(install_mod, "default_stderr_log", lambda: tmp_path / "logs" / "err.log")
    monkeypatch.setattr(install_mod, "default_config_path", lambda: tmp_path / "config.toml")
    monkeypatch.setattr(install_mod, "resolve_shim_command", lambda: ("/opt/bin/autogpt-shim",))
    monkeypatch.setattr(install_mod, "_user_id", lambda: 501, raising=False)
    monkeypatch.setattr(install_mod, "_is_elevated", lambda: True, raising=False)
    return tmp_path


def _whoami(argv: Sequence[str], timeout: float) -> Any:
    """What `install` asks before it writes an entry: `whoami /user`, for
    whose the Windows task is, and launchd, for a LaunchAgent under the
    earlier label, of which there is none here."""
    if argv[0] == "launchctl":
        assert list(argv[:2]) == ["launchctl", "print"], argv
        assert argv[-1].endswith(f"/{install_mod.FORMER_LAUNCHD_LABEL}"), argv
        return _result(113, "Could not find service\n")
    assert list(argv) == ["whoami", "/user", "/fo", "csv", "/nh"], argv
    return _result(0, '"desktop\\alice","S-1-5-21-1004336348-1177238915-682003330-1001"\r\n')


def _installed(platform_name: str, home: Path) -> Path:
    """Install `platform_name`'s autostart entry under `home`; its file."""
    status = install_mod.install_autostart(home=home, platform_name=platform_name, run=_whoami)
    assert status.target_path.is_file()
    return status.target_path


def _manager_for(platform_name: str, target: Path, **state: Any) -> _Manager:
    """The stand-in service manager for `platform_name`, running the
    service unless `state` says otherwise."""
    if platform_name == "darwin":
        return _Manager(target, _Launchd(*state.get("loaded", ("gui/501",))))
    if platform_name == "windows":
        return _Manager(target, _TaskScheduler(**{"registered": True, "running": True, **state}))
    return _Manager(target, _Systemd(target, **{"active": "active", "enabled": True, **state}))


def _idle_manager_for(platform_name: str, target: Path) -> _Manager:
    """The stand-in service manager for `platform_name` with no service."""
    if platform_name == "darwin":
        return _Manager(target, _Launchd())
    if platform_name == "windows":
        return _Manager(target, _TaskScheduler())
    return _Manager(target, _Systemd(target))


# What stops the service on each OS: every one of these has to run while the
# entry's file is still there.
STOPS = {
    "darwin": f"launchctl bootout gui/501/{LABEL}",
    "linux": f"systemctl --user disable --now {UNIT}",
    "wsl2": f"systemctl --user disable --now {UNIT}",
    "windows": f"schtasks /End /TN {TASK}",
}


def _run_cli(
    monkeypatch: pytest.MonkeyPatch, platform_name: str, manager: _Manager, command: str
) -> int:
    """Run `autogpt-shim <command>` as on `platform_name`; its exit status."""
    monkeypatch.setattr(install_mod, "_current_platform_for_install", lambda: platform_name)
    monkeypatch.setattr(install_mod, "run_command", manager, raising=False)
    monkeypatch.setattr(sys, "argv", ["autogpt-shim", command])
    try:
        cli.main()
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    return 0


def _commands(text: str) -> list[str]:
    """The command lines in what the CLI printed: the indented lines, and
    what follows `enable:` or `disable:`."""
    found = []
    for line in text.splitlines():
        stripped = line.strip()
        for prefix in ("enable:", "disable:", "set up with:"):
            if stripped.startswith(prefix):
                found.append(stripped[len(prefix) :].strip())
                break
        else:
            if line.startswith("  ") and stripped and not stripped.startswith(("path:", "note:")):
                found.append(stripped)
    return found


# ── The stop runs first, on every OS ─────────────────────────────────────────


@pytest.mark.parametrize("platform_name", PLATFORMS)
def test_uninstall_stops_the_service_before_it_removes_the_entry(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    platform_name: str,
) -> None:
    target = _installed(platform_name, home)
    manager = _manager_for(platform_name, target)

    status = _run_cli(monkeypatch, platform_name, manager, "uninstall")

    out, err = capsys.readouterr()
    assert status == 0, err
    assert manager.file_was_there_for(STOPS[platform_name]) == [True]
    assert not target.exists()
    assert "Stopped the service" in out
    assert f"Removed the autostart entry: {target}" in out
    assert "may still be running" not in out + err


def test_launchd_is_asked_by_label_in_both_of_the_persons_sessions(home: Path) -> None:
    """A LaunchAgent loaded over SSH is in the background session
    (user/<uid>), where the login session's name for it finds nothing."""
    target = _installed("darwin", home)
    launchd = _Launchd("user/501")
    manager = _Manager(target, launchd)

    done = install_mod.uninstall_autostart(home=home, platform_name="darwin", run=manager)

    assert manager.ran == [
        f"launchctl print gui/501/{LABEL}",
        f"launchctl print user/501/{LABEL}",
        f"launchctl bootout user/501/{LABEL}",
        f"launchctl print user/501/{LABEL}",
    ]
    assert launchd.loaded == set()
    assert done.service.found is True and done.service.problem is None
    assert "user/501" in done.service.said
    assert str(target) not in " ".join(manager.ran)


def test_launchd_is_waited_for_while_the_service_exits(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """launchd answers bootout at once for a service that is still exiting
    (36, in progress), and gives it 20 s before it kills it."""
    clock = _Clock(monkeypatch)
    target = _installed("darwin", home)
    launchd = _Launchd("gui/501", slow=3)
    manager = _Manager(target, launchd)

    done = install_mod.uninstall_autostart(home=home, platform_name="darwin", run=manager)

    assert done.service.found is True and done.service.problem is None
    assert launchd.loaded == set()
    assert len(clock.slept) == 2
    assert not done.may_still_run
    assert not target.exists()


def test_systemd_reads_its_units_again_once_the_file_is_gone(home: Path) -> None:
    target = _installed("linux", home)
    systemd = _Systemd(target, active="active", enabled=True)
    manager = _Manager(target, systemd)

    done = install_mod.uninstall_autostart(home=home, platform_name="linux", run=manager)

    assert manager.calls == [
        (f"systemctl --user show --property=ActiveState {UNIT}", True),
        (f"systemctl --user disable --now {UNIT}", True),
        ("systemctl --user daemon-reload", False),
    ]
    assert (systemd.active, systemd.enabled) == ("inactive", False)
    assert done.service.said == f"Stopped the service: systemd stopped and disabled {UNIT}."
    assert done.notes == ()


def test_a_task_is_ended_before_it_is_deleted(home: Path) -> None:
    """Deleting a task leaves what it started running, with nothing left to
    end it by."""
    target = _installed("windows", home)
    scheduler = _TaskScheduler(registered=True, running=True)
    manager = _Manager(target, scheduler)

    done = install_mod.uninstall_autostart(home=home, platform_name="windows", run=manager)

    assert manager.calls == [
        (f"schtasks /Query /TN {TASK}", True),
        (f"schtasks /End /TN {TASK}", True),
        (f"schtasks /Delete /TN {TASK} /F", True),
    ]
    assert (scheduler.registered, scheduler.running) == (False, False)
    assert done.service.found is True and done.service.problem is None


# ── Nothing installed, and installed but not running ─────────────────────────


@pytest.mark.parametrize("platform_name", PLATFORMS)
def test_uninstall_with_nothing_installed_and_nothing_running_succeeds_and_says_so(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    platform_name: str,
) -> None:
    target = install_mod.target_path_for(platform_name, home)
    manager = _idle_manager_for(platform_name, target)

    first = _run_cli(monkeypatch, platform_name, manager, "uninstall")
    out, err = capsys.readouterr()
    second = _run_cli(monkeypatch, platform_name, manager, "uninstall")

    assert (first, second) == (0, 0)
    assert err == ""
    assert f"No autostart entry is installed at {target}." in out
    assert "The service wasn't" in out
    # It only asked: nothing was stopped, unregistered or reloaded.
    asked = ("launchctl print", "systemctl --user show", "schtasks /Query")
    assert all(line.startswith(asked) for line in manager.ran), manager.ran


@pytest.mark.parametrize("platform_name", PLATFORMS)
def test_uninstall_of_an_entry_whose_service_is_not_running_succeeds_and_says_so(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    platform_name: str,
) -> None:
    target = _installed(platform_name, home)
    manager = _idle_manager_for(platform_name, target)

    status = _run_cli(monkeypatch, platform_name, manager, "uninstall")

    out, err = capsys.readouterr()
    assert status == 0
    assert err == ""
    assert "The service wasn't" in out
    assert "Stopped the service" not in out
    assert f"Removed the autostart entry: {target}" in out
    assert not target.exists()


def test_an_enabled_unit_that_is_not_running_is_disabled_while_its_file_is_there(
    home: Path,
) -> None:
    target = _installed("linux", home)
    systemd = _Systemd(target, active="failed", enabled=True)
    manager = _Manager(target, systemd)

    done = install_mod.uninstall_autostart(home=home, platform_name="linux", run=manager)

    assert manager.file_was_there_for("systemctl --user disable --now") == [True]
    assert systemd.enabled is False
    assert done.service.found is False
    assert done.service.said == f"The service wasn't running; systemd disabled {UNIT}."


@pytest.mark.parametrize("platform_name", PLATFORMS)
def test_a_service_left_running_by_a_removed_entry_is_stopped(
    home: Path, platform_name: str
) -> None:
    """What the old uninstall left: no file, and the service still running."""
    target = install_mod.target_path_for(platform_name, home)
    manager = _manager_for(platform_name, target)

    done = install_mod.uninstall_autostart(home=home, platform_name=platform_name, run=manager)

    assert done.entry_installed is False
    assert done.service.found is True and done.service.problem is None
    assert not done.may_still_run
    state = manager.answer
    if isinstance(state, _Launchd):
        assert state.loaded == set()
    elif isinstance(state, _Systemd):
        assert state.active == "inactive"
        assert f"systemctl --user stop {UNIT}" in manager.ran
    else:
        assert isinstance(state, _TaskScheduler)
        assert (state.registered, state.running) == (False, False)


# ── Each way the stop can fail ───────────────────────────────────────────────


def _missing_tool(argv: list[str]) -> Any:
    return _result(None, failure=f"`{argv[0]}` was not found on PATH")


@pytest.mark.parametrize("platform_name", PLATFORMS)
def test_a_missing_tool_is_named_and_the_entry_is_still_removed(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    platform_name: str,
) -> None:
    target = _installed(platform_name, home)
    manager = _Manager(target, _missing_tool)

    status = _run_cli(monkeypatch, platform_name, manager, "uninstall")

    out, err = capsys.readouterr()
    tool = {"darwin": "launchctl", "windows": "schtasks"}.get(platform_name, "systemctl")
    assert status == 1
    assert "Could not stop the service: " in err
    assert f"`{tool}` was not found on PATH" in err
    assert "The service may still be running. To stop it, run" in err
    assert not target.exists()
    assert f"Removed the autostart entry: {target}" in out


@pytest.mark.parametrize("platform_name", PLATFORMS)
def test_a_service_manager_that_cannot_be_asked_with_nothing_installed_is_not_a_failure(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    platform_name: str,
) -> None:
    """With no entry installed and a service manager that can't be asked,
    there is no sign of a service: it says what it couldn't check, and
    succeeds."""
    target = install_mod.target_path_for(platform_name, home)
    manager = _Manager(target, _missing_tool)

    status = _run_cli(monkeypatch, platform_name, manager, "uninstall")

    out, err = capsys.readouterr()
    assert status == 0
    assert "Could not check for a running service" in err
    assert "may still be running" not in out + err
    assert f"No autostart entry is installed at {target}." in out
    assert "If one is still running from an earlier install" in out


def test_no_user_session_bus_is_said_as_systemctl_said_it(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _installed("linux", home)
    wanted = target.parent / "default.target.wants" / UNIT
    wanted.parent.mkdir()
    _link_or_skip(wanted, target)
    systemd = _Systemd(target, active="active", enabled=True, bus=False)
    manager = _Manager(target, systemd)

    status = _run_cli(monkeypatch, "linux", manager, "uninstall")

    out, err = capsys.readouterr()
    assert status == 1
    assert "Failed to connect to bus: No medium found" in err
    assert f"  systemctl --user stop {UNIT}" in err
    assert f"Removed the autostart entry: {target}" in out
    assert not target.exists()
    # The link that enabled the unit goes with the file: left behind, it
    # would enable a later install without anyone asking.
    assert not wanted.is_symlink()
    assert manager.ran == [f"systemctl --user show --property=ActiveState {UNIT}"]


def test_a_link_to_another_unit_file_of_the_same_name_is_left(home: Path) -> None:
    """Only the link to this entry's own unit file is removed by hand."""
    target = _installed("linux", home)
    elsewhere = home / "elsewhere" / UNIT
    elsewhere.parent.mkdir()
    elsewhere.write_text("[Unit]\n", encoding="utf-8")
    wanted = target.parent / "default.target.wants" / UNIT
    wanted.parent.mkdir()
    _link_or_skip(wanted, elsewhere)
    manager = _Manager(target, _Systemd(target, bus=False))

    install_mod.uninstall_autostart(home=home, platform_name="linux", run=manager)

    assert wanted.is_symlink()
    assert not target.exists()


def test_a_unit_systemd_did_not_stop_is_reported_with_its_reason(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _installed("linux", home)
    systemd = _Systemd(target, active="active", enabled=True)

    def answer(argv: list[str]) -> Any:
        if argv[2] == "disable":
            return _result(1, "Failed to disable unit: Access denied\n")
        return systemd(argv)

    status = _run_cli(monkeypatch, "linux", _Manager(target, answer), "uninstall")

    out, err = capsys.readouterr()
    assert status == 1
    assert f"Could not stop the service: systemd didn't stop {UNIT}: " in err
    assert "Failed to disable unit: Access denied.\n" in err
    assert not target.exists()


def test_a_unit_that_was_not_running_and_could_not_be_disabled_is_no_failure(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Nothing runs and the entry is gone, with the link that enabled it:
    what couldn't be done is said, and nothing is left to do."""
    target = _installed("linux", home)
    wanted = target.parent / "default.target.wants" / UNIT
    wanted.parent.mkdir()
    _link_or_skip(wanted, target)
    systemd = _Systemd(target, active="inactive", enabled=True)

    def answer(argv: list[str]) -> Any:
        if argv[2] == "disable":
            return _result(1, "Failed to disable unit: Access denied\n")
        return systemd(argv)

    status = _run_cli(monkeypatch, "linux", _Manager(target, answer), "uninstall")

    out, err = capsys.readouterr()
    assert status == 0
    assert f"The service wasn't running, but systemd didn't disable {UNIT}: " in err
    assert "may still be running" not in out + err
    assert not target.exists()
    assert not wanted.is_symlink()
    assert systemd.reloads == 1


@pytest.mark.parametrize("platform_name", ["linux", "windows"])
def test_a_stop_that_outlasts_its_wait_is_reported(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    platform_name: str,
) -> None:
    target = _installed(platform_name, home)
    working = _manager_for(platform_name, target).answer
    stop = STOPS[platform_name]

    def answer(argv: list[str]) -> Any:
        if " ".join(argv) == stop:
            return _result(None, failure=f"`{stop}` did not finish within 100 s")
        return working(argv)

    status = _run_cli(monkeypatch, platform_name, _Manager(target, answer), "uninstall")

    out, err = capsys.readouterr()
    assert status == 1
    assert f"`{stop}` did not finish within 100 s" in err
    assert "The service may still be running." in err
    assert not target.exists()


def test_a_launchagent_that_never_unloads_is_reported_once_the_wait_runs_out(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    clock = _Clock(monkeypatch)
    target = _installed("darwin", home)
    manager = _Manager(target, _Launchd("gui/501", stuck=True))

    status = _run_cli(monkeypatch, "darwin", manager, "uninstall")

    out, err = capsys.readouterr()
    assert status == 1
    assert f"launchd still has gui/501/{LABEL} loaded 30 s after it was told to unload it" in err
    assert "Boot-out failed: 36: Operation now in progress).\n" in err
    assert f"  launchctl bootout gui/$(id -u)/{LABEL}" in err
    # 150 looks, 0.2 s apart: the sum of those isn't exactly 30 in floating point.
    assert 29.5 < sum(clock.slept) < 30.5
    assert not target.exists()


def test_the_command_for_a_stuck_launchagent_names_the_session_it_is_in(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _Clock(monkeypatch)
    target = _installed("darwin", home)
    manager = _Manager(target, _Launchd("user/501", stuck=True))

    done = install_mod.uninstall_autostart(home=home, platform_name="darwin", run=manager)

    assert done.may_still_run
    assert done.service.stop_command == f"launchctl bootout user/$(id -u)/{LABEL}"


@pytest.mark.parametrize(
    ("may_end", "may_delete", "kept", "said"),
    [
        (False, True, True, "Task Scheduler didn't end the task"),
        (True, False, True, "but didn't delete it, so it starts again at the next logon"),
    ],
    ids=["cannot-end", "cannot-delete"],
)
def test_a_task_this_prompt_may_not_change_is_reported(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    may_end: bool,
    may_delete: bool,
    kept: bool,
    said: str,
) -> None:
    """A task registered from an elevated prompt, seen from one that isn't.
    One that can't be ended is kept: deleted, nothing could end what it
    started."""
    monkeypatch.setattr(install_mod, "_is_elevated", lambda: False)
    target = _installed("windows", home)
    scheduler = _TaskScheduler(
        registered=True, running=True, may_end=may_end, may_delete=may_delete
    )
    manager = _Manager(target, scheduler)

    status = _run_cli(monkeypatch, "windows", manager, "uninstall")

    out, err = capsys.readouterr()
    assert status == 1
    assert said in err
    assert "ERROR: Access is denied." in err
    assert "To stop it, run from an elevated prompt:" in err
    assert f'  cmd /c "schtasks /End /TN {TASK} && schtasks /Delete /TN {TASK} /F"' in err
    assert scheduler.registered is kept
    if not may_end:
        assert not any(line.startswith("schtasks /Delete") for line in manager.ran)
    assert not target.exists()


def test_the_command_to_stop_a_task_by_hand_deletes_it_only_once_it_is_ended(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command `uninstall` prints for a task it couldn't end joined its
    two steps with `&`, which runs the second whatever the first did. Run
    where ending the task is refused again, it deleted the task all the
    same, and what the task had started ran on with no task left to end it
    by: what `uninstall` itself takes care not to do."""
    monkeypatch.setattr(install_mod, "_is_elevated", lambda: False)
    target = _installed("windows", home)
    manager = _Manager(target, _TaskScheduler(registered=True, running=True, may_end=False))

    done = install_mod.uninstall_autostart(home=home, platform_name="windows", run=manager)

    assert done.may_still_run
    assert done.service.stop_command == (
        f'cmd /c "schtasks /End /TN {TASK} && schtasks /Delete /TN {TASK} /F"'
    )


@pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe is Windows'")
def test_cmd_runs_the_second_step_only_after_a_first_that_succeeded(tmp_path: Path) -> None:
    """What the command above rests on, with the real cmd.exe: a first step
    that fails (a file that isn't there) and one that succeeds, each written
    both ways. Run in a folder of the test's own, which is empty: in the
    folder the tests were started from, a file of that name would have made
    the first step succeed."""
    fails = "type autogpt-shim-no-such-file.txt 2>nul"
    succeeds = "ver >nul"
    assert list(tmp_path.iterdir()) == []

    def ran_the_second(first: str, joined_with: str) -> bool:
        ran = _real(["cmd", "/c", f"{first} {joined_with} echo deleted"], cwd=tmp_path)
        return "deleted" in ran.stdout

    assert not ran_the_second(fails, "&&")
    assert ran_the_second(succeeds, "&&")
    # And what the command did before: the second step whatever the first did.
    assert ran_the_second(fails, "&")


@pytest.mark.parametrize("installed", [True, False])
def test_a_task_another_account_registered_is_not_this_accounts_service(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    installed: bool,
) -> None:
    """schtasks answers for a task this prompt can't see as for one that
    isn't there, and only an elevated prompt sees every task. The task
    `install` writes is the account's own, which its own prompt sees: one
    it can't see is another account's, or was registered from an elevated
    prompt. So seeing none is no failure, and the answer says what a
    prompt that isn't elevated can't see."""
    monkeypatch.setattr(install_mod, "_is_elevated", lambda: False)
    target = (
        _installed("windows", home) if installed else install_mod.target_path_for("windows", home)
    )
    scheduler = _TaskScheduler(registered=True, running=True, visible=False)
    manager = _Manager(target, scheduler)

    status = _run_cli(monkeypatch, "windows", manager, "uninstall")

    out, err = capsys.readouterr()
    assert status == 0
    assert err == ""
    said = " ".join(out.split())
    assert f"this account has no task {TASK} registered" in said
    assert "run `autogpt-shim uninstall` from one" in said
    assert manager.ran == [f"schtasks /Query /TN {TASK}"]
    # The other account's task is as it was.
    assert scheduler.registered and scheduler.running
    assert not target.exists()


def test_from_an_elevated_prompt_a_task_that_is_not_there_is_not_there(home: Path) -> None:
    target = _installed("windows", home)
    manager = _Manager(target, _TaskScheduler())

    done = install_mod.uninstall_autostart(home=home, platform_name="windows", run=manager)

    assert done.service.found is False and done.service.problem is None
    assert not done.may_still_run


def test_an_entry_that_cannot_be_removed_is_reported(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = _installed("linux", home)
    manager = _manager_for("linux", target)
    real_unlink = Path.unlink

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self == target:
            raise PermissionError(13, "Permission denied", str(self))
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)

    status = _run_cli(monkeypatch, "linux", manager, "uninstall")

    out, err = capsys.readouterr()
    assert status == 1
    assert "Stopped the service" in out
    assert f"Could not remove the autostart entry {target}: " in err
    assert "Permission denied" in err
    assert target.exists()


# ── What install, status and uninstall print as commands ─────────────────────


@pytest.mark.parametrize("platform_name", PLATFORMS)
def test_no_command_uninstall_or_status_prints_names_the_removed_file(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    platform_name: str,
) -> None:
    """`launchctl unload <plist>` was printed as the way to stop the service
    once the plist was gone, and `status` went on naming the file."""
    target = _installed(platform_name, home)
    failing = _Manager(target, _missing_tool)

    _run_cli(monkeypatch, platform_name, failing, "uninstall")
    out, err = capsys.readouterr()
    uninstall_commands = _commands(out) + _commands(err)
    (home / "config.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["autogpt-shim", "--config", str(home / "config.toml")])
    monkeypatch.setattr(install_mod, "_current_platform_for_install", lambda: platform_name)
    cli._print_autostart_status()
    status_commands = _commands(capsys.readouterr().out)

    assert not target.exists()
    assert uninstall_commands, (out, err)
    assert status_commands == ["autogpt-shim install"]
    for command in uninstall_commands + status_commands:
        assert str(target) not in command, command


@pytest.mark.parametrize("platform_name", PLATFORMS)
def test_what_install_and_status_print_to_disable_the_service_works_without_the_file(
    home: Path, capsys: pytest.CaptureFixture[str], platform_name: str
) -> None:
    """The way to turn the service off is `autogpt-shim uninstall`, which
    stops it whether or not the entry's file is there. The service
    managers' own commands for it read the file."""
    _installed(platform_name, home)

    status = install_mod.status_for(platform_name, home)

    assert status.installed is True
    assert status.enable_command
    assert status.disable_command == "autogpt-shim uninstall"


@pytest.mark.parametrize("platform_name", PLATFORMS)
def test_status_of_an_installed_entry_says_how_to_enable_and_disable_it(
    home: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    platform_name: str,
) -> None:
    _installed(platform_name, home)
    monkeypatch.setattr(install_mod, "_current_platform_for_install", lambda: platform_name)

    cli._print_autostart_status()

    out = capsys.readouterr().out
    assert f"Autostart entry ({platform_name}): installed" in out
    assert "  enable: " in out
    assert "  disable: autogpt-shim uninstall" in out


def test_install_says_uninstall_is_how_to_stop_the_service(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = home / "config.toml"
    config.write_text("", encoding="utf-8")
    monkeypatch.setattr(install_mod, "_current_platform_for_install", lambda: "darwin")
    monkeypatch.setattr(install_mod, "run_command", _whoami)
    monkeypatch.setattr(sys, "argv", ["autogpt-shim", "--config", str(config), "install"])
    for field in ("platform_url", "platform_ws_url", "platform_oauth_url", "oauth_client_id"):
        monkeypatch.delenv(f"AUTOGPT_SHIM_{field.upper()}", raising=False)

    cli.main()

    out = capsys.readouterr().out
    assert "To stop the service and remove this entry later, run:\n  autogpt-shim uninstall" in out
    assert "launchctl unload" not in out


# ── The command runner ───────────────────────────────────────────────────────


def _python(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_a_command_runs_with_no_input_and_its_output_is_kept() -> None:
    assert _REAL_RUN_COMMAND is not None
    done = _REAL_RUN_COMMAND(
        _python(
            "import sys; data = sys.stdin.read(); print('out', repr(data)); "
            "print('err', file=sys.stderr); sys.exit(3)"
        ),
        120,
    )

    assert done.returncode == 3
    assert done.failure is None
    assert "out ''" in done.output and "err" in done.output
    assert done.said() == " ".join(done.output.split())


def test_a_command_that_is_not_there_is_said_to_be_missing() -> None:
    assert _REAL_RUN_COMMAND is not None
    done = _REAL_RUN_COMMAND(["autogpt-shim-no-such-tool", "--user"], 120)

    assert done.returncode is None
    assert done.failure == "`autogpt-shim-no-such-tool` was not found on PATH"
    assert not done.succeeded


def test_a_command_that_outlasts_its_time_is_ended_and_said_so() -> None:
    assert _REAL_RUN_COMMAND is not None
    command = _python("import time; time.sleep(3600)")

    done = _REAL_RUN_COMMAND(command, 1)

    assert done.returncode is None
    assert done.failure == f"`{' '.join(command)}` did not finish within 1 s"


def test_tests_cannot_reach_the_real_service_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    """conftest.py keeps every test from launchd, systemd and Task
    Scheduler: an uninstall here would stop the service of whoever runs the
    tests. It stands a refusing runner in for the real one, which is the
    one a stop uses when it is given none."""
    assert install_mod.run_command is not _REAL_RUN_COMMAND
    with pytest.raises(AssertionError, match="real service manager"):
        install_mod.run_command(["launchctl", "print", f"gui/501/{LABEL}"], 1)

    ran: list[list[str]] = []

    def record(argv: Sequence[str], timeout: float) -> Any:
        ran.append(list(argv))
        return _result(1, "ERROR: The system cannot find the file specified.")

    monkeypatch.setattr(install_mod, "run_command", record)
    monkeypatch.setattr(install_mod, "_is_elevated", lambda: True)
    install_mod.stop_service("windows", entry_installed=False)

    assert ran == [["schtasks", "/Query", "/TN", TASK]]


# ── The real service managers, on a service of the test's own ────────────────


def real_service_manager_asked_for(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the tests that register a throwaway service with the OS run:
    in CI, or with AUTOGPT_SHIM_TEST_SERVICE_MANAGER=1. Any value used to
    do, so `=0`, set to keep them off, turned them on."""
    environ = os.environ if environ is None else environ
    return bool(environ.get("CI")) or environ.get("AUTOGPT_SHIM_TEST_SERVICE_MANAGER") == "1"


@pytest.mark.parametrize(
    ("environ", "asked_for"),
    [
        ({}, False),
        ({"AUTOGPT_SHIM_TEST_SERVICE_MANAGER": "1"}, True),
        ({"AUTOGPT_SHIM_TEST_SERVICE_MANAGER": "0"}, False),
        ({"AUTOGPT_SHIM_TEST_SERVICE_MANAGER": ""}, False),
        ({"AUTOGPT_SHIM_TEST_SERVICE_MANAGER": "false"}, False),
        ({"CI": "true"}, True),
        ({"CI": "true", "AUTOGPT_SHIM_TEST_SERVICE_MANAGER": "0"}, True),
    ],
)
def test_the_real_service_managers_are_reached_in_ci_or_when_asked_for_by_1(
    environ: dict[str, str], asked_for: bool
) -> None:
    assert real_service_manager_asked_for(environ) is asked_for


# These register a throwaway service with the OS, so they run where that is
# expected: in CI, or where AUTOGPT_SHIM_TEST_SERVICE_MANAGER is 1.
_real_manager = pytest.mark.skipif(
    not real_service_manager_asked_for(),
    reason="registers a throwaway service with the OS: runs in CI, or with "
    "AUTOGPT_SHIM_TEST_SERVICE_MANAGER=1",
)

# A program that keeps counting in a file while it runs: what the service
# manager is asked to stop. Its folder is written into the script, so the
# command that starts it stays short (`schtasks /TR` takes 261 characters).
_BEATING = (
    "import os, time\n"
    "folder = {folder!r}\n"
    "count = 0\n"
    "while True:\n"
    "    count += 1\n"
    "    with open(os.path.join(folder, 'beat.tmp'), 'w', encoding='utf-8') as file:\n"
    "        file.write(str(count))\n"
    "    os.replace(os.path.join(folder, 'beat.tmp'), os.path.join(folder, 'beat'))\n"
    "    time.sleep(0.05)\n"
)


def _beating_script(folder: Path) -> Path:
    """The counting program, as a script in `folder` that counts there."""
    script = folder / "beating.py"
    script.write_text(_BEATING.format(folder=str(folder)), encoding="utf-8")
    return script


def _wait_for(condition: Callable[[], bool], what: str, seconds: float = 120) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, f"timed out waiting for {what}"
        time.sleep(0.05)


def _beat(folder: Path) -> str:
    try:
        return (folder / "beat").read_text(encoding="utf-8")
    except OSError:
        return ""


def _is_beating(folder: Path) -> bool:
    """Whether the program in `folder` is still counting, watched for 3 s:
    60 of its steps."""
    before = _beat(folder)
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if _beat(folder) != before:
            return True
        time.sleep(0.05)
    return False


def _real(argv: Sequence[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=120,
        check=False,
    )


@_real_manager
@pytest.mark.skipif(sys.platform != "darwin", reason="launchd is macOS's")
@pytest.mark.parametrize("exits_at_once", [True, False], ids=["exits-at-once", "has-to-be-killed"])
def test_real_launchd_stops_a_loaded_agent_by_its_label_alone(
    tmp_path: Path, exits_at_once: bool
) -> None:
    """With the plist already gone, as the old uninstall left it. The agent
    that ignores SIGTERM is one launchd has to kill, three seconds after it
    asks: bootout may answer before then."""
    assert _REAL_RUN_COMMAND is not None
    uid = os.getuid()
    label = f"co.agpt.shim.test-{uuid.uuid4().hex[:12]}"
    domain = next(
        (
            domain
            for domain in (f"gui/{uid}", f"user/{uid}")
            if _real(["launchctl", "print", domain]).returncode == 0
        ),
        None,
    )
    if domain is None:
        pytest.skip("launchd has no session for this user")
    # Nothing by this name is loaded: what the stop finds when nothing runs.
    nothing = install_mod._stop_launchd(_REAL_RUN_COMMAND, label=label, uid=uid)
    assert (nothing.found, nothing.problem) == (False, None)

    script = _beating_script(tmp_path)
    if not exits_at_once:
        ignoring = "import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        script.write_text(ignoring + script.read_text(encoding="utf-8"), encoding="utf-8")
    plist = tmp_path / f"{label}.plist"
    plist.write_bytes(
        plistlib.dumps(
            {
                "Label": label,
                "ProgramArguments": [sys.executable, str(script)],
                "RunAtLoad": True,
                # As the shim's own plist: launchd restarts it unless it is unloaded.
                "KeepAlive": {"SuccessfulExit": False},
                "ExitTimeOut": 3,
            }
        )
    )
    loaded = _real(["launchctl", "bootstrap", domain, str(plist)])
    assert loaded.returncode == 0, loaded.stderr
    bootouts: list[int | None] = []

    def run(argv: Sequence[str], timeout: float) -> Any:
        done = _REAL_RUN_COMMAND(argv, timeout)
        if argv[1] == "bootout":
            bootouts.append(done.returncode)
        return done

    try:
        _wait_for(lambda: _is_beating(tmp_path), "the agent to start")
        plist.unlink()  # the stop needs no file

        stopped = install_mod._stop_launchd(run, label=label, uid=uid)

        assert stopped.problem is None, stopped.problem
        assert stopped.found is True
        assert domain in stopped.said
        assert _real(["launchctl", "print", f"{domain}/{label}"]).returncode != 0
        assert not _is_beating(tmp_path)
        # What launchd answers a bootout with: done, or still in progress.
        assert bootouts in ([0], [36]), bootouts
    finally:
        _real(["launchctl", "bootout", f"{domain}/{label}"])


@_real_manager
@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="systemd is Linux's")
def test_real_systemd_stops_and_disables_a_unit_before_its_file_goes(tmp_path: Path) -> None:
    assert _REAL_RUN_COMMAND is not None
    if _real(["systemctl", "--user", "show-environment"]).returncode != 0:
        pytest.skip("no systemd user manager to ask")
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if not runtime:
        pytest.skip("no XDG_RUNTIME_DIR to keep a throwaway unit in")
    unit = f"autogpt-shim-test-{uuid.uuid4().hex[:12]}.service"
    # The user manager reads units from here too, and it is gone at logout.
    units = Path(runtime) / "systemd" / "user"
    units.mkdir(parents=True, exist_ok=True)
    unit_file = units / unit
    nothing = install_mod._stop_systemd(_REAL_RUN_COMMAND, unit=unit, entry_installed=False)
    assert (nothing.found, nothing.problem) == (False, None)

    unit_file.write_text(
        "[Service]\n"
        f'ExecStart="{sys.executable}" "{_beating_script(tmp_path)}"\n'
        "Restart=on-failure\n"
        "[Install]\nWantedBy=default.target\n",
        encoding="utf-8",
    )
    try:
        assert _real(["systemctl", "--user", "daemon-reload"]).returncode == 0
        started = _real(["systemctl", "--user", "enable", "--now", unit])
        assert started.returncode == 0, started.stderr
        _wait_for(lambda: _is_beating(tmp_path), "the unit to start")

        stopped = install_mod._stop_systemd(_REAL_RUN_COMMAND, unit=unit, entry_installed=True)

        assert stopped.problem is None, stopped.problem
        assert stopped.found is True
        assert not _is_beating(tmp_path)
        shown = _real(["systemctl", "--user", "show", "-p", "ActiveState,UnitFileState", unit])
        assert "ActiveState=inactive" in shown.stdout
        assert "UnitFileState=enabled" not in shown.stdout
    finally:
        _real(["systemctl", "--user", "disable", "--now", unit])
        unit_file.unlink(missing_ok=True)
        _real(["systemctl", "--user", "daemon-reload"])


@_real_manager
@pytest.mark.skipif(sys.platform != "win32", reason="Task Scheduler is Windows'")
def test_real_task_scheduler_ends_a_running_task_and_deletes_it(tmp_path: Path) -> None:
    assert _REAL_RUN_COMMAND is not None
    if not install_mod._is_elevated():
        pytest.skip("registering a task for this test takes an elevated prompt")
    task = rf"\AutoGPT-test\Shim-{uuid.uuid4().hex[:12]}"
    nothing = install_mod._stop_scheduled_task(_REAL_RUN_COMMAND, task=task, elevated=True)
    assert (nothing.found, nothing.problem) == (False, None)

    command = subprocess.list2cmdline([sys.executable, str(_beating_script(tmp_path))])
    assert len(command) <= 261, command
    # As SYSTEM, which needs nobody signed in at a desktop to run now.
    made = _real(
        ["schtasks", "/Create", "/TN", task, "/TR", command, "/SC", "ONCE", "/ST", "00:00"]
        + ["/RU", "SYSTEM", "/F"]
    )
    assert made.returncode == 0, made.stdout + made.stderr
    try:
        # Registered and not running: ending it is no failure.
        idle = install_mod._stop_scheduled_task(_REAL_RUN_COMMAND, task=task, elevated=True)
        assert idle.problem is None, idle.problem
        assert idle.found is True
        assert _real(["schtasks", "/Query", "/TN", task]).returncode != 0

        made = _real(
            ["schtasks", "/Create", "/TN", task, "/TR", command, "/SC", "ONCE", "/ST", "00:00"]
            + ["/RU", "SYSTEM", "/F"]
        )
        assert made.returncode == 0, made.stdout + made.stderr
        ran = _real(["schtasks", "/Run", "/TN", task])
        assert ran.returncode == 0, ran.stdout + ran.stderr
        _wait_for(lambda: _is_beating(tmp_path), "the task to start")

        stopped = install_mod._stop_scheduled_task(_REAL_RUN_COMMAND, task=task, elevated=True)

        assert stopped.problem is None, stopped.problem
        assert stopped.found is True
        assert _real(["schtasks", "/Query", "/TN", task]).returncode != 0
        assert not _is_beating(tmp_path)
    finally:
        _real(["schtasks", "/End", "/TN", task])
        _real(["schtasks", "/Delete", "/TN", task, "/F"])


def _link_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"this account can't make symlinks: {exc}")
