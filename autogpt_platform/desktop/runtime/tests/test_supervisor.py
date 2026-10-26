import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import psutil
import pytest

from autogpt_desktop import (
    apps,
    bootstrap,
    events,
    postgres,
    rabbitmq,
    runs,
    supervisor,
    valkey,
)
from autogpt_desktop.layout import EXE, Bundle, DataDir
from autogpt_desktop.process import ChildRegistry, ManagedProcess, wait_until
from autogpt_desktop.supervisor import Stack, StartupError

DESKTOP = Path(__file__).resolve().parents[2]

posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows ties children to the runtime with a Job Object",
)
posix_links_only = pytest.mark.skipif(
    sys.platform == "win32", reason="Windows uses junctions, which winlinks.py clears"
)


@pytest.fixture
def stack(tmp_path: Path) -> Stack:
    data = DataDir(tmp_path / "data")
    data.prepare()
    return Stack(Bundle(tmp_path / "runtime"), data)


def python_process(stack: Stack, code: str) -> ManagedProcess:
    process = ManagedProcess(
        name="service",
        argv=[sys.executable, "-c", code],
        env={},
        cwd=stack.data.root,
        log_dir=stack.data.logs,
    )
    process.start()
    return process


def sleeper(stack: Stack, name: str, on_stop: Callable[[str], None]) -> ManagedProcess:
    def stop(process: ManagedProcess) -> None:
        on_stop(name)
        assert process.popen
        process.popen.terminate()

    return ManagedProcess(
        name=name,
        argv=[sys.executable, "-c", "import time; time.sleep(60)"],
        env={},
        cwd=stack.data.root,
        log_dir=stack.data.logs,
        graceful_stop=stop,
    )


def launcher(stack: Stack, tmp_path: Path) -> tuple[ManagedProcess, int]:
    """Like rabbitmq-server: a process that starts the real one and waits.
    Returns it with the pid of what it started."""
    pid_file = tmp_path / "started.pid"
    process = python_process(
        stack,
        "import subprocess, sys, time;"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid));"
        "time.sleep(60)",
    )
    assert wait_until(lambda: pid_file.exists() and pid_file.read_text() != "", 20, 0.1)
    return process, int(pid_file.read_text())


def is_gone(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_tiers_stop_in_reverse_and_a_tier_stops_all_at_once(stack: Stack):
    stopped: list[str] = []

    def slow_stop(name: str) -> None:
        stopped.append(name)
        time.sleep(1)

    services = ["rest", "executor", "scheduler", "frontend"]
    stack.launch([sleeper(stack, "database", slow_stop)])
    stack.launch([sleeper(stack, name, slow_stop) for name in services])
    started = time.monotonic()
    stack.stop()

    assert sorted(stopped[:-1]) == sorted(services)
    assert stopped[-1] == "database"
    assert time.monotonic() - started < 4  # one at a time would take 5
    assert all(process.exit_code() is not None for process in stack.processes)
    assert not stack.registry.path.exists()


@posix_only
def test_killing_a_service_kills_what_it_started(stack: Stack, tmp_path: Path):
    process, started = launcher(stack, tmp_path)
    process.kill()
    assert wait_until(lambda: is_gone(started), 5, 0.1)


@posix_only
def test_what_a_dead_runtime_left_running_is_killed_on_the_next_start(
    stack: Stack, tmp_path: Path
):
    process, started = launcher(stack, tmp_path)
    stack.registry.record([process])

    ChildRegistry(stack.registry.path).reap_leftovers()

    assert wait_until(lambda: is_gone(started), 5, 0.1)
    assert process.popen and process.popen.wait(5) is not None
    assert not stack.registry.path.exists()


@pytest.mark.usefixtures("cache_home")
def test_the_port_mapper_is_a_supervised_process_not_a_daemon(tmp_path: Path):
    """Erlang would otherwise start `epmd -daemon`, which leaves the session
    and outlives the app."""
    bundle = Bundle(tmp_path / "runtime")
    data = DataDir(tmp_path / "data")
    data.prepare()
    epmd = bundle.erlang_home / "erts-15.0" / "bin" / f"epmd{EXE}"
    epmd.parent.mkdir(parents=True)
    epmd.touch()

    process = rabbitmq.epmd_process(
        bundle, data, {"epmd": 15001, "rabbitmq_dist": 15002}
    )

    assert process.argv == [str(epmd)]
    assert process.env["ERL_EPMD_PORT"] == "15001"
    assert process.env["ERL_EPMD_ADDRESS"] == "127.0.0.1"


def test_a_service_that_dies_while_starting_fails_at_once(stack: Stack):
    process = python_process(stack, "import sys; sys.exit(3)")
    started = time.monotonic()
    with pytest.raises(StartupError, match="exited while starting"):
        stack.await_ready(process, lambda: False, timeout=60)
    assert time.monotonic() - started < 10


def test_a_stop_request_ends_the_wait_for_a_slow_service(stack: Stack):
    process = python_process(stack, "import time; time.sleep(60)")
    stack.stop_requested.set()
    started = time.monotonic()
    try:
        with pytest.raises(StartupError, match="cancelled"):
            stack.await_ready(process, lambda: False, timeout=60)
        assert time.monotonic() - started < 10
    finally:
        process.stop()


def test_a_service_that_never_answers_times_out(stack: Stack):
    process = python_process(stack, "import time; time.sleep(60)")
    try:
        with pytest.raises(StartupError, match="did not start"):
            stack.await_ready(process, lambda: False, timeout=1)
    finally:
        process.stop()


def test_a_ready_service_returns_normally(stack: Stack):
    process = python_process(stack, "import time; time.sleep(60)")
    try:
        stack.await_ready(process, lambda: True, timeout=5)
    finally:
        process.stop()


def test_a_stop_during_a_database_update_asks_the_shell_to_wait(stack: Stack, capsys):
    """The shell kills a runtime that is slow to stop, and a migration that
    is killed leaves a database Prisma refuses to touch."""
    stack.migrating = True
    stack.request_stop()

    event = json.loads(capsys.readouterr().out)
    assert stack.stop_requested.is_set()
    assert event["grace_seconds"] > bootstrap.MIGRATION_TIMEOUT_SECONDS


def test_an_ordinary_stop_asks_for_nothing(stack: Stack, capsys):
    stack.request_stop()
    assert capsys.readouterr().out == ""


def test_a_shell_that_is_gone_does_not_stop_the_runtime_reporting(
    monkeypatch, tmp_path: Path
):
    with open(tmp_path / "pipe", "w") as closed:
        pass
    monkeypatch.setattr(sys, "stdout", closed)
    events.error("the executor keeps stopping", fatal=True)


def fake_initdb(monkeypatch) -> None:
    def run_tool(argv: list[str], **kwargs) -> subprocess.CompletedProcess[bytes]:
        pgdata = Path(
            next(arg for arg in argv if arg.startswith("--pgdata=")).split("=", 1)[1]
        )
        pgdata.mkdir(parents=True)
        (pgdata / "PG_VERSION").write_text("18")
        (pgdata / "postgresql.conf").write_text("")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(postgres, "run_tool", run_tool)


def test_a_first_start_that_was_cut_short_starts_over(stack: Stack, monkeypatch):
    """Nobody has used that database yet, and half-created it cannot be
    repaired, only refused."""
    fake_initdb(monkeypatch)
    bundle, data = stack.bundle, stack.data
    postgres.initialize(bundle, data, "password")
    (data.postgres / "half-applied-migration").write_text("")
    assert not postgres.is_initialized(data)

    postgres.initialize(bundle, data, "password")

    assert not (data.postgres / "half-applied-migration").exists()
    assert (data.postgres / "PG_VERSION").is_file()


def test_a_database_that_finished_its_first_start_is_never_recreated(
    stack: Stack, monkeypatch
):
    fake_initdb(monkeypatch)
    bundle, data = stack.bundle, stack.data
    postgres.initialize(bundle, data, "password")
    postgres.first_run_completed(data)
    (data.postgres / "user-data").write_text("")

    assert postgres.is_initialized(data)
    postgres.initialize(bundle, data, "password")

    assert (data.postgres / "user-data").exists()


@pytest.mark.parametrize(("stored", "compatible"), [("18", True), ("16", False)])
def test_data_from_another_postgres_major_is_refused_with_an_explanation(
    stack: Stack, monkeypatch, stored: str, compatible: bool
):
    stack.data.postgres.mkdir()
    (stack.data.postgres / "PG_VERSION").write_text(stored)
    monkeypatch.setattr(
        postgres,
        "run_tool",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, b"postgres (PostgreSQL) 18.6", b""
        ),
    )
    if compatible:
        postgres.check_compatible(stack.bundle, stack.data)
        return
    with pytest.raises(postgres.IncompatibleDataError, match="PostgreSQL 16"):
        postgres.check_compatible(stack.bundle, stack.data)


def test_a_cache_that_cannot_read_its_data_starts_empty(stack: Stack, monkeypatch):
    """Another build's file format, or a damaged file: Valkey exits at once.
    What it holds is expendable, so that must not keep the app from starting."""
    data = stack.data
    unreadable = data.valkey / "dump.rdb"
    unreadable.write_text("written by something else")
    cache = ManagedProcess(
        name="valkey",
        argv=[
            sys.executable,
            "-c",
            "import os, sys, time; sys.exit(1) if os.path.exists('dump.rdb') else time.sleep(60)",
        ],
        env={},
        cwd=data.valkey,
        log_dir=data.logs,
    )
    monkeypatch.setattr(
        valkey, "is_ready", lambda port, password: cache.exit_code() is None
    )
    stack.launch([cache])
    wait_until(lambda: cache.exit_code() is not None, 20, 0.1)
    try:
        stack.await_cache(cache, {"valkey": 20000, "valkey_bus": 20001}, "password")

        assert cache.exit_code() is None
        assert not unreadable.exists()
        assert (data.valkey / valkey.CONFIG_NAME).is_file()
        assert [
            path.name for path in data.root.glob("valkey.unreadable-*/dump.rdb")
        ] == ["dump.rdb"]
    finally:
        cache.stop()


@pytest.fixture
def cache_home(tmp_path: Path, monkeypatch) -> None:
    """Keep the space-free aliases out of the real cache directory."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "home" / ".cache"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "home" / "local"))
    monkeypatch.setenv("PROGRAMDATA", str(tmp_path / "home" / "shared"))


@pytest.mark.usefixtures("cache_home")
def test_rabbitmq_is_given_a_path_without_spaces(tmp_path: Path):
    """Its launch scripts break on a space, and macOS keeps application data
    under "Application Support"."""
    spaced = tmp_path / "Application Support" / "AutoGPT" / "rabbitmq"
    spaced.mkdir(parents=True)  # as DataDir.prepare leaves it
    alias = Path(rabbitmq._short(spaced))
    (spaced / "marker").write_text("x")

    assert (alias / "marker").read_text() == "x"
    if not any(character.isspace() for character in str(tmp_path)):
        assert not any(character.isspace() for character in str(alias))
    assert rabbitmq._short(spaced) == str(alias)


@pytest.mark.usefixtures("cache_home")
def test_rabbitmq_scripts_are_run_through_the_alias(tmp_path: Path):
    bundle = Bundle(tmp_path / "My Apps" / "runtime")
    bundle.rabbitmq_home.mkdir(parents=True)  # an alias is to a directory that exists
    script = Path(rabbitmq._script(bundle, "rabbitmq-server"))
    assert script.parent.name == "sbin"
    if not any(character.isspace() for character in str(tmp_path)):
        assert not any(character.isspace() for character in str(script))


def symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:  # Windows without the right to make symbolic links
        pytest.skip(f"this account cannot make symbolic links: {exc}")


def test_links_to_folders_that_are_gone_are_removed(tmp_path: Path):
    """A data directory that was deleted, a disk image that was unmounted:
    seven such links had gathered on one Mac."""
    root = tmp_path / "links"
    root.mkdir()
    here, gone = tmp_path / "data here", tmp_path / "data gone"
    here.mkdir()
    gone.mkdir()
    symlink_or_skip(root / "0123456789abcdef", here)
    symlink_or_skip(root / "fedcba9876543210", gone)
    gone.rmdir()
    (root / "a folder someone put here").mkdir()
    (root / "a file someone put here").write_text("x")

    rabbitmq.forget_dangling(root)

    assert sorted(entry.name for entry in root.iterdir()) == [
        "0123456789abcdef",
        "a file someone put here",
        "a folder someone put here",
    ]
    assert here.is_dir()
    rabbitmq.forget_dangling(tmp_path / "no such folder")  # nothing to do, and no error


@posix_links_only
@pytest.mark.usefixtures("cache_home")
def test_every_start_clears_the_dangling_links_of_this_install(tmp_path: Path):
    """Also when this start needs no link itself (no space in its paths)."""
    spaced = tmp_path / "Application Support" / "rabbitmq"
    spaced.mkdir(parents=True)
    stale = Path(rabbitmq._short(spaced))
    assert stale.is_symlink()
    spaced.rmdir()
    data = DataDir(tmp_path / "data")
    data.prepare()

    rabbitmq.prepare(data, 15000, "user", "password")

    assert not stale.is_symlink()


# --- quitting with agents running ---------------------------------------------


def test_only_the_exact_request_is_taken_for_stop_the_runs():
    assert (
        supervisor.shell_request(b'{"command": "stop", "drain": true}')
        == supervisor.STOP_THE_RUNS
    )
    assert (
        supervisor.shell_request(b'{"command":"stop","drain":true}\r')
        == supervisor.STOP_THE_RUNS
    )
    assert supervisor.shell_request(b'{"command": "stop"}') == supervisor.STOP
    assert (
        supervisor.shell_request(b'{"command": "stop", "drain": "yes"}')
        == supervisor.STOP
    )
    for line in (
        b"",
        b"stop",
        b'{"command": "drain"}',
        b'["stop"]',
        b'{"drain": true}',
        b"\xff\xfe",
    ):
        assert supervisor.shell_request(line) is None


class Shell:
    """What stands where the runtime's stdin is: a pipe the test writes to."""

    def __init__(self) -> None:
        self.read_end, self.write_end = os.pipe()

    def fileno(self) -> int:
        return self.read_end

    def isatty(self) -> bool:
        return False

    def write(self, data: bytes) -> None:
        os.write(self.write_end, data)

    def close(self) -> None:
        os.close(self.write_end)


@pytest.fixture
def shell(monkeypatch):
    shell = Shell()
    monkeypatch.setattr(sys, "stdin", shell)
    yield shell
    with contextlib.suppress(OSError):
        shell.close()
    os.close(shell.read_end)


def test_closing_stdin_is_a_plain_stop_that_leaves_runs_to_resume(
    stack: Stack, shell: Shell
):
    supervisor._listen_to_the_shell(stack)
    shell.close()

    assert stack.stop_requested.wait(10)
    assert not stack.stop_the_runs
    assert not stack.hurry.is_set()


def test_a_request_written_in_pieces_is_read_as_one_line(stack: Stack, shell: Shell):
    supervisor._listen_to_the_shell(stack)
    shell.write(b'{"command": "st')
    time.sleep(0.2)
    assert not stack.stop_requested.is_set()
    shell.write(b'op", "drain": true}\n')

    assert stack.stop_requested.wait(10)
    assert stack.stop_the_runs
    assert not stack.hurry.is_set()  # stdin is still open: the runs get their time

    shell.close()
    assert stack.hurry.wait(10)  # the second request: wait for nothing more


def test_a_shell_that_dies_right_after_asking_is_still_obeyed(
    stack: Stack, shell: Shell
):
    """The request and the end of stdin can arrive in one read, and the
    request may lack its newline."""
    shell.write(b'{"command": "stop", "drain": true}')
    shell.close()
    supervisor._listen_to_the_shell(stack)

    assert stack.hurry.wait(10)
    assert stack.stop_the_runs


def test_noise_on_stdin_is_not_a_request(stack: Stack, shell: Shell, caplog):
    supervisor._listen_to_the_shell(stack)
    shell.write(b'hello\n\n{"command": "restart"}\n')
    time.sleep(0.3)

    assert not stack.stop_requested.is_set()
    assert "asks for nothing known" in caplog.text


def test_a_request_to_stop_the_runs_after_a_plain_stop_changes_nothing(stack: Stack):
    stack.request_stop()
    stack.request_stop(stop_the_runs=True)
    assert not stack.stop_the_runs
    assert stack.hurry.is_set()


class Running:
    """What the cache says is running: one answer per question, the last one
    from then on."""

    def __init__(self, *answers: runs.Activity | None) -> None:
        self.answers = list(answers)
        self.asked = 0
        self.closed = False

    def active(self) -> runs.Activity | None:
        self.asked += 1
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]

    def close(self) -> None:
        self.closed = True


class Cancels:
    def __init__(self) -> None:
        self.cancelled: list[runs.Activity] = []

    def cancel(self, activity: runs.Activity) -> None:
        self.cancelled.append(activity)


ONE_RUN = runs.Activity(("run",), ())
NOTHING = runs.Activity()


def events_of(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines()]


def test_the_shell_is_told_when_the_number_of_runs_changes_and_only_then(
    stack: Stack, capsys, monkeypatch
):
    busy = runs.Activity(("a", "b"), ("chat",))
    other = runs.Activity(("c", "d"), ("chat",))
    monkeypatch.setattr(stack, "cache", Running(NOTHING, busy, other, None, NOTHING))
    monkeypatch.setattr(
        runs, "running", lambda connect, among=None: ()
    )  # those without a lock have ended
    stack.connect = lambda: None

    for _ in range(5):
        stack.report_activity()

    assert events_of(capsys) == [
        {"event": "activity", "graph_runs": 2, "copilot_turns": 1},
        {"event": "activity", "graph_runs": 0, "copilot_turns": 0},
    ]


def running_in_the_database(
    monkeypatch, stack: Stack, *answers: tuple[str, ...] | None
) -> list:
    """Make `runs.running` answer these, one per question and the last one
    from then on. Returns what it was asked: None for all, or the ids."""
    left, asked = list(answers), []

    def running(connect, among=None):
        asked.append(None if among is None else tuple(among))
        found = left.pop(0) if len(left) > 1 else left[0]
        return (
            found
            if among is None or found is None
            else tuple(run for run in found if run in among)
        )

    monkeypatch.setattr(runs, "running", running)
    stack.connect = lambda: None
    return asked


def test_a_run_that_lost_its_lock_and_goes_on_is_still_counted(
    stack: Stack, capsys, monkeypatch
):
    """After a sleep longer than a lock lasts, the tray would otherwise say
    nothing is running and a quit would ask nothing."""
    monkeypatch.setattr(stack, "cache", Running(ONE_RUN, NOTHING))
    running_in_the_database(monkeypatch, stack, ("run",))

    for _ in range(3):
        stack.report_activity()

    assert events_of(capsys) == [
        {"event": "activity", "graph_runs": 1, "copilot_turns": 0}
    ]
    assert stack.activity == ONE_RUN


def test_stopping_the_runs_ends_one_that_lost_its_lock_too(
    stack: Stack, capsys, monkeypatch
):
    """One run slept through its lock and goes on. Left out, it would be
    killed with its executor and resume on the next start, after its user
    chose to stop the runs."""
    both = runs.Activity(("run", "no-lock"), ())
    monkeypatch.setattr(
        stack, "cache", Running(both, ONE_RUN, ONE_RUN, ONE_RUN, NOTHING)
    )
    asked = running_in_the_database(
        monkeypatch, stack, ("no-lock",), ("no-lock",), ("no-lock",), ()
    )
    stack.report_activity()
    stack.report_activity()  # its lock has gone; the database has it running
    assert stack.activity.counts == (2, 0)
    capsys.readouterr()
    stack.queue = cancels = Cancels()

    assert stack.stop_runs() is True

    assert stack.stopping == both
    assert runs.read_stopped(stack.stopped_runs) == both
    assert cancels.cancelled == [both]
    assert events_of(capsys)[0]["message"] == "Stopping 2 agent runs"
    # Waited for until the database no longer has it as running: it has no
    # lock whose going would say so.
    assert asked == [("no-lock",), None, ("no-lock",), ("no-lock",)]


def test_stopping_the_runs_ends_every_run_the_database_has_running(
    stack: Stack, capsys, monkeypatch
):
    """Also one nobody is working on: its executor died a moment ago, or it
    lost its lock unseen. It would start again on the next launch. It is not
    waited for: there is nobody to end it but the stop itself."""
    monkeypatch.setattr(supervisor, "CANCEL_WAIT_SECONDS", 30)
    monkeypatch.setattr(stack, "cache", Running(ONE_RUN, NOTHING))
    running_in_the_database(monkeypatch, stack, ("in-nobodys-hands",))
    stack.queue = cancels = Cancels()
    all_of_it = runs.Activity(("run", "in-nobodys-hands"), ())
    started = time.monotonic()

    assert stack.stop_runs() is True

    assert time.monotonic() - started < 10
    assert stack.stopping == all_of_it
    assert runs.read_stopped(stack.stopped_runs) == all_of_it
    assert cancels.cancelled == [all_of_it]
    assert events_of(capsys)[0]["message"] == "Stopping 2 agent runs"


def test_with_only_runs_in_nobodys_hands_there_is_nothing_to_wait_for(
    stack: Stack, monkeypatch
):
    monkeypatch.setattr(supervisor, "CANCEL_WAIT_SECONDS", 30)
    monkeypatch.setattr(stack, "cache", Running(NOTHING))
    running_in_the_database(monkeypatch, stack, ("in-nobodys-hands",))
    stack.queue = cancels = Cancels()
    started = time.monotonic()

    assert stack.stop_runs() is True

    assert time.monotonic() - started < 5
    assert cancels.cancelled == [runs.Activity(("in-nobodys-hands",), ())]


def test_a_database_that_does_not_say_still_stops_the_unlocked_runs_it_knew_of(
    stack: Stack, monkeypatch
):
    monkeypatch.setattr(supervisor, "CANCEL_WAIT_SECONDS", 0.2)
    monkeypatch.setattr(stack, "cache", Running(ONE_RUN, NOTHING))
    running_in_the_database(monkeypatch, stack, ("run",), None)
    stack.report_activity()
    stack.report_activity()  # its lock has gone, and it runs on
    stack.queue = cancels = Cancels()

    assert stack.stop_runs() is True

    assert cancels.cancelled == [ONE_RUN]


def test_a_run_whose_executor_died_is_not_in_progress(
    stack: Stack, capsys, monkeypatch
):
    """The database has it as running until its message comes round again,
    or for ever when the message is lost. Nobody is working on it: the tray
    must not say so for the rest of the session."""
    monkeypatch.setattr(stack, "cache", Running(ONE_RUN, NOTHING))
    asked = running_in_the_database(monkeypatch, stack, ("run",))
    stack.hosts = {
        "workers": apps.Group("workers", ("executor", "scheduler")),
        "api": apps.Group("api", ("rest",)),
    }
    stack.report_activity()
    stack.forget_runs_of("api")  # not the executor's host: nothing changes
    assert stack.watching.seen == ONE_RUN

    stack.forget_runs_of("workers")  # as `watch` does before it restarts it
    stack.report_activity()

    assert stack.activity == NOTHING
    assert asked == []
    assert [event["graph_runs"] for event in events_of(capsys)] == [1, 0]


def test_the_supervisor_forgets_the_runs_of_an_executor_it_restarts():
    text = (DESKTOP / "runtime" / "autogpt_desktop" / "supervisor.py").read_text(
        encoding="utf-8"
    )
    restarting = text[
        text.index("    def watch(self) -> int:") : text.index(
            "    def report_activity(self)"
        )
    ]
    assert restarting.index("self.forget_runs_of(process.name)") < restarting.index(
        "process.start()"
    )
    isolating = text[
        text.index("    def isolate(self, process") : text.index("    def log_of(self")
    ]
    assert isolating.index("self.forget_runs_of(process.name)") < isolating.index(
        "del self.hosts[process.name]"
    )


@pytest.mark.parametrize(
    "answers, asked",
    [((None, ONE_RUN, NOTHING), 3), ((None, None, None, NOTHING), 4)],
    ids=["answers when asked again", "what it last said"],
)
def test_a_cache_that_does_not_answer_at_the_stop_is_not_taken_to_say_nothing_runs(
    stack: Stack, capsys, monkeypatch, caplog, answers, asked: int
):
    """Its kept connection had broken, or it was slow for once. The user
    chose to stop the runs: a plain stop instead would resume every one."""
    cache = Running(*answers)
    monkeypatch.setattr(stack, "cache", cache)
    stack.activity = ONE_RUN  # what the shell asked its user about
    stack.queue = cancels = Cancels()

    assert stack.stop_runs() is True

    assert cache.asked == asked
    assert runs.read_stopped(stack.stopped_runs) == ONE_RUN
    assert cancels.cancelled == [ONE_RUN]
    assert events_of(capsys)[0]["message"] == "Stopping 1 agent run"
    assert ("stopping what it last said" in caplog.text) == (asked == 4)


def test_stopping_the_runs_records_the_choice_asks_them_to_cancel_and_waits_for_them(
    stack: Stack, capsys, monkeypatch
):
    monkeypatch.setattr(stack, "cache", Running(ONE_RUN, ONE_RUN, ONE_RUN, NOTHING))
    stack.queue = cancels = Cancels()
    started = time.monotonic()

    assert stack.stop_runs() is True

    assert 0.9 < time.monotonic() - started < 10
    assert runs.read_stopped(stack.stopped_runs) == ONE_RUN
    assert cancels.cancelled == [ONE_RUN]
    [event, cancellation] = events_of(capsys)
    assert event["step"] == "stopping" and event["message"] == "Stopping 1 agent run"
    assert event["grace_seconds"] == supervisor.STOPPING_RUNS_GRACE_SECONDS
    assert cancellation["event"] == "shutdown"
    assert cancellation["phase"] == "cancellation"
    assert cancellation["outcome"] == "cancelled"
    assert 0.9 < cancellation["seconds"] < 10


def test_the_shell_is_asked_for_the_time_such_a_stop_can_take():
    """It kills the runtime after a minute otherwise (src/runtime.js), in the
    middle of the wait or of the broker's own stop."""
    asked = supervisor.STOPPING_RUNS_GRACE_SECONDS
    assert asked >= supervisor.CANCEL_WAIT_SECONDS + supervisor.SHELL_STOP_GRACE_SECONDS
    shell = (DESKTOP / "src" / "runtime.js").read_text(encoding="utf-8")
    assert "const STOP_GRACE_MS = 60_000;" in shell
    assert "const MAX_GRACE_SECONDS = 30 * 60;" in shell and asked <= 30 * 60
    assert 'const STOP_THE_RUNS = { command: "stop", drain: true };' in shell


def test_a_run_started_meanwhile_is_not_waited_for(stack: Stack, monkeypatch):
    monkeypatch.setattr(
        stack, "cache", Running(ONE_RUN, runs.Activity(("started-since",), ()))
    )
    stack.queue = Cancels()
    started = time.monotonic()

    assert stack.stop_runs() is True

    assert time.monotonic() - started < 5
    assert runs.read_stopped(stack.stopped_runs) == ONE_RUN


def test_a_run_that_does_not_end_is_waited_for_only_so_long(stack: Stack, monkeypatch):
    monkeypatch.setattr(supervisor, "CANCEL_WAIT_SECONDS", 1.5)
    monkeypatch.setattr(stack, "cache", Running(ONE_RUN))
    stack.queue = Cancels()
    started = time.monotonic()

    assert stack.stop_runs() is True

    assert 1.4 < time.monotonic() - started < 6


def test_asked_to_stop_again_the_runs_are_not_waited_for(stack: Stack, monkeypatch):
    monkeypatch.setattr(stack, "cache", Running(ONE_RUN))
    stack.queue = cancels = Cancels()
    stack.request_stop(stop_the_runs=True)
    threading.Timer(0.5, stack.request_stop).start()
    started = time.monotonic()

    assert stack.stop_runs() is True

    assert time.monotonic() - started < 5
    # Told all the same, and on record: they do not resume.
    assert cancels.cancelled == [ONE_RUN]
    assert runs.read_stopped(stack.stopped_runs) == ONE_RUN


def test_with_nothing_running_there_is_nothing_to_stop_or_to_ask_time_for(
    stack: Stack, capsys, monkeypatch
):
    for answer in (NOTHING, None):
        monkeypatch.setattr(stack, "cache", Running(answer))
        assert stack.stop_runs() is False
    assert capsys.readouterr().out == ""
    assert not stack.stopped_runs.exists()


def serving(stack: Stack, order: list[str]) -> None:
    """A stack that is up: a server tier, then two tiers of its own."""
    database, workers, frontend = (
        sleeper(stack, name, order.append)
        for name in ("postgres", "workers", apps.FRONTEND)
    )
    stack.hosts = {"workers": apps.Group("workers", ("executor", "scheduler"))}
    stack.launch([database])
    stack.launch([workers])
    stack.launch([frontend])
    stack.ready_at = time.monotonic()
    stack.connect = lambda: None


def test_stopped_runs_are_ended_between_the_services_and_the_database(
    stack: Stack, monkeypatch
):
    """Earlier would race the executor's own last write; later there is no
    database to write to."""
    order: list[str] = []
    serving(stack, order)
    monkeypatch.setattr(stack, "cache", Running(ONE_RUN, NOTHING))
    stack.queue = Cancels()
    told: list[runs.Activity] = []

    def end_stopped(connect, path, cache, queue, chosen) -> None:
        order.append("runs ended")
        told.append(chosen)

    monkeypatch.setattr(runs, "end_stopped", end_stopped)
    monkeypatch.setattr(runs, "running", lambda connect, among=None: None)
    stack.request_stop(stop_the_runs=True)

    stack.stop()

    assert order == [apps.FRONTEND, "workers", "runs ended", "postgres"]
    # What it was told to stop, not only what the disk took.
    assert told == [ONE_RUN]
    assert stack.cache.closed


def test_a_look_for_orphans_under_way_does_not_hold_up_a_stop(
    stack: Stack, monkeypatch
):
    """The broker's tool can take its time; the stop does not wait for it
    before it begins, and only briefly once everything is down."""
    order: list[str] = []
    serving(stack, order)
    monkeypatch.setattr(supervisor, "ORPHAN_SWEEP_STOP_WAIT_SECONDS", 0.2)
    asked, answer = threading.Event(), threading.Event()

    def slow() -> bool:
        asked.set()
        return answer.wait(30)

    stack.orphans = runs.OrphanSweep(slow, lambda: 1000.0)
    stack.orphans.due = 0.0
    stack.orphans.tend()
    assert asked.wait(10)
    started = time.monotonic()
    stack.request_stop()

    stack.stop()

    assert time.monotonic() - started < 15
    assert order == [apps.FRONTEND, "workers", "postgres"]
    answer.set()


def test_no_look_for_orphans_begins_once_a_stop_was_asked_for(stack: Stack):
    stack.connect = lambda: (_ for _ in ()).throw(
        AssertionError("the database is not asked")
    )
    stack.request_stop()
    assert stack.sweep_orphans() is False


def test_a_plain_stop_ends_no_run(stack: Stack, monkeypatch):
    order: list[str] = []
    serving(stack, order)
    cache = Running(ONE_RUN)
    monkeypatch.setattr(stack, "cache", cache)
    stack.queue = cancels = Cancels()
    monkeypatch.setattr(runs, "end_stopped", lambda *args: order.append("runs ended"))
    stack.request_stop()

    stack.stop()

    assert order == [apps.FRONTEND, "workers", "postgres"]
    assert cancels.cancelled == [] and cache.asked == 0
    assert not stack.stopped_runs.exists()


def test_a_start_that_was_cut_short_has_no_runs_to_stop(stack: Stack, monkeypatch):
    cache = Running(ONE_RUN)
    monkeypatch.setattr(stack, "cache", cache)
    stack.request_stop(stop_the_runs=True)

    stack.stop()  # never became ready

    assert cache.asked == 0


@pytest.mark.parametrize("merged", [True, False], ids=["merged", "isolated"])
def test_every_host_is_stopped_before_runs_are_ended_whatever_the_layout(
    stack: Stack, monkeypatch, merged: bool
):
    """The process that hosts the executor is whichever apps.py puts it in."""
    groups = apps.copilot_alone(apps.layout(merged)) if merged else apps.layout(merged)
    stack.hosts = {group.name: group for group in groups}
    stopped: list[str] = []
    stack.launch([sleeper(stack, "valkey", stopped.append)])
    stack.launch([sleeper(stack, group.name, stopped.append) for group in groups])
    stack.launch([sleeper(stack, apps.FRONTEND, stopped.append)])

    services, servers = stack.tiers_to_stop()

    names = [[process.name for process in tier] for tier in services]
    assert names == [[group.name for group in groups], [apps.FRONTEND]]
    assert any("executor" in group.services for group in groups)
    assert [[process.name for process in tier] for tier in servers if tier] == [
        ["valkey"]
    ]
    stack.stop()
