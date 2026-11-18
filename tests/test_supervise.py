"""Under the Windows task the shim comes back after an exit that asks for it.

Task Scheduler's RestartOnFailure applies when the task's action can't be
started. A program that starts and then exits, with whatever code, is a run
that completed: a shim that exited 75 (the "try again" code launchd and
systemd both restart), or crashed after logon, stayed down until the next
logon. The task now runs `autogpt-shim start --supervise`, which starts the
daemon again as those two do.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import autogpt_local_executor.cli as cli
from autogpt_local_executor import install, supervise
from autogpt_local_executor.install import AutostartContext


class _Runs:
    """A daemon that exits with each of `codes` in turn, and the pauses the
    supervisor made between its runs."""

    def __init__(self, *codes: int, lasting: float = 1.0) -> None:
        self.codes = list(codes)
        self.ran = 0
        self.pauses: list[float] = []
        self.now = 0.0
        self.lasting = lasting
        self.stop = threading.Event()

    def run(self) -> int:
        self.ran += 1
        self.now += self.lasting
        return self.codes.pop(0)

    def wait(self, seconds: float) -> bool:
        self.pauses.append(seconds)
        self.now += seconds
        return self.stop.is_set()

    def supervise(self) -> int:
        return supervise.supervise(self.run, self.stop, wait=self.wait, clock=lambda: self.now)


def test_exit_75_leads_to_another_run_after_the_pause() -> None:
    """75 is EX_TEMPFAIL: a keychain not up yet, audit storage that is full.
    The shim exits with it to be started again by a new process."""
    runs = _Runs(75, 0)

    assert runs.supervise() == 0

    assert runs.ran == 2
    assert runs.pauses == [supervise.FIRST_PAUSE_SECONDS]


@pytest.mark.parametrize("code", [0, 77, 78])
def test_an_exit_that_means_stop_ends_the_loop(code: int) -> None:
    """0: a stop that was asked for, and the errors the shim itself won't
    retry (a revoked session, a protocol mismatch). 77: the sign-in was
    refused. 78: a setting or a missing package, which fails the same way
    every time. The systemd unit doesn't restart after these either."""
    runs = _Runs(code, 0)

    assert runs.supervise() == code

    assert runs.ran == 1
    assert runs.pauses == []


@pytest.mark.parametrize("code", [1, 2, -9, -11, 0xC0000005, 0xC0000409, 255])
def test_a_crash_leads_to_another_run(code: int) -> None:
    """An unhandled error, a signal, a Windows exception code."""
    runs = _Runs(code, 0)

    assert runs.supervise() == 0

    assert runs.ran == 2
    assert len(runs.pauses) == 1


def test_a_stop_asked_for_during_the_pause_ends_the_loop_at_once() -> None:
    runs = _Runs(75, 75, 0)

    def stop_meanwhile(seconds: float) -> bool:
        runs.pauses.append(seconds)
        runs.stop.set()
        return True

    assert supervise.supervise(runs.run, runs.stop, wait=stop_meanwhile) == 75

    assert runs.ran == 1
    assert runs.pauses == [supervise.FIRST_PAUSE_SECONDS]


def test_a_stop_asked_for_while_the_daemon_runs_ends_the_loop_without_a_pause() -> None:
    """The daemon ended because it was told to, whatever it exited with."""
    runs = _Runs(1, 0)
    run = runs.run

    def stopped_while_running() -> int:
        runs.stop.set()
        return run()

    assert supervise.supervise(stopped_while_running, runs.stop, wait=runs.wait) == 1

    assert runs.ran == 1
    assert runs.pauses == []


def test_the_pause_is_a_real_wait_that_a_stop_ends() -> None:
    """With no wait of a test's own, the pause is a wait on the stop."""
    stop = threading.Event()
    codes = [75, 0]
    waited: list[float] = []
    real_wait = stop.wait

    def quick_wait(timeout: float | None = None) -> bool:
        waited.append(timeout or 0.0)
        return real_wait(0)

    stop.wait = quick_wait  # type: ignore[method-assign]

    assert supervise.supervise(lambda: codes.pop(0), stop) == 0
    assert waited == [supervise.FIRST_PAUSE_SECONDS]


def test_a_daemon_that_cannot_stay_up_is_not_started_in_a_tight_loop() -> None:
    """The pause doubles after each run that ended soon, up to a minute."""
    runs = _Runs(*[75] * 7, 0, lasting=1.0)

    runs.supervise()

    assert runs.pauses == [5.0, 10.0, 20.0, 40.0, 60.0, 60.0, 60.0]


def test_a_daemon_that_ran_a_while_starts_the_pauses_over() -> None:
    runs = _Runs(75, 75, 75, 0)
    lasting = iter([1.0, 1.0, supervise.SETTLED_SECONDS, 1.0])

    def run() -> int:
        runs.lasting = next(lasting)
        return runs.run()

    supervise.supervise(run, runs.stop, wait=runs.wait, clock=lambda: runs.now)

    assert runs.pauses == [5.0, 10.0, 5.0]


# ── The daemon as a process of its own ───────────────────────────────────────


def test_the_daemon_is_run_with_the_supervisors_arguments_less_its_own_option() -> None:
    assert supervise.daemon_command(
        ["--config", "C:\\conf dir\\config.toml", "start", "--supervise", "--enable-shell"]
    ) == [
        sys.executable,
        "-m",
        "autogpt_local_executor",
        "--config",
        "C:\\conf dir\\config.toml",
        "start",
        "--enable-shell",
    ]


def test_a_real_child_is_run_again_until_it_exits_to_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real processes, on the OS the tests run on: one that exits 75 the
    first two times and 0 the third, and has no input to read."""
    runs = tmp_path / "runs"
    script = (
        "import pathlib, sys\n"
        f"runs = pathlib.Path({str(runs)!r})\n"
        "count = len(runs.read_text(encoding='utf-8')) if runs.exists() else 0\n"
        "runs.write_text('x' * (count + 1), encoding='utf-8')\n"
        "assert sys.stdin.read() == ''\n"
        "sys.exit(0 if count == 2 else 75)\n"
    )
    monkeypatch.setattr(
        supervise, "daemon_command", lambda _arguments: [sys.executable, "-c", script]
    )
    monkeypatch.setattr(supervise, "FIRST_PAUSE_SECONDS", 0.01)
    monkeypatch.setattr(supervise, "LONGEST_PAUSE_SECONDS", 0.01)

    assert supervise.run_supervised(["start", "--supervise"]) == 0

    assert runs.read_text(encoding="utf-8") == "xxx"


def test_a_child_is_not_started_once_a_stop_was_asked_for() -> None:
    stop = threading.Event()
    stop.set()

    assert supervise._Child().run([sys.executable, "-c", "raise SystemExit(75)"], stop) == 0


# ── Who gets the loop ────────────────────────────────────────────────────────


def _run_cli(monkeypatch: pytest.MonkeyPatch, *arguments: str) -> int:
    monkeypatch.setattr(sys, "argv", ["autogpt-shim", *arguments])
    try:
        cli.main()
    except SystemExit as exit_:
        return int(exit_.code or 0)
    return 0


def test_start_with_the_option_runs_the_supervisor_and_no_daemon_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked: list[list[str]] = []

    def supervised(arguments: list[str]) -> int:
        asked.append(list(arguments))
        return 75

    async def no_daemon(_config: Any) -> int:
        raise AssertionError("the supervisor ran a daemon in its own process")

    monkeypatch.setattr(supervise, "run_supervised", supervised)
    monkeypatch.setattr(cli, "_cmd_start", no_daemon)
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")

    assert _run_cli(monkeypatch, "--config", str(config), "start", "--supervise") == 75

    assert asked == [["--config", str(config), "start", "--supervise"]]


def test_start_in_a_terminal_gets_no_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[Any] = []

    async def daemon(config: Any) -> int:
        started.append(config)
        return 75

    def no_supervisor(_arguments: list[str]) -> int:
        raise AssertionError("a plain `start` was supervised")

    monkeypatch.setattr(cli, "_cmd_start", daemon)
    monkeypatch.setattr(supervise, "run_supervised", no_supervisor)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert _run_cli(monkeypatch, "start") == 75
    assert len(started) == 1


def _context() -> AutostartContext:
    return AutostartContext(
        shim_command=("/opt/shim/bin/autogpt-shim",),
        config_path="/home/u/.config/autogpt-local-executor/config.toml",
        stdout_log="/home/u/shim.out.log",
        stderr_log="/home/u/shim.err.log",
        user_id="S-1-5-21-1004336348-1177238915-682003330-1001",
    )


def test_the_windows_task_turns_the_supervisor_on() -> None:
    rendered = install.render_template("taskscheduler.xml.template", _context())

    [arguments] = [line for line in rendered.splitlines() if "<Arguments>" in line]
    assert arguments.strip().endswith("start --supervise</Arguments>")
    # What didn't bring the shim back is gone, so nobody reads it as if it did.
    assert "<RestartOnFailure>" not in rendered


@pytest.mark.parametrize("template", ["launchd.plist.template", "systemd.service.template"])
def test_launchd_and_systemd_restart_the_shim_themselves(template: str) -> None:
    rendered = install.render_template(template, _context())

    assert "--supervise" not in rendered
    assert "start" in rendered


def test_the_help_says_what_the_option_is_for(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run_cli(monkeypatch, "start", "--help") == 0

    text = " ".join(capsys.readouterr().out.split())
    assert "--supervise" in text
    assert "Windows task" in text
