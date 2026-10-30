"""PostgreSQL on Windows is started by `pg_ctl start`, which does not stay:
the supervisor watches, records and stops the server it finds through
postmaster.pid. The class has nothing Windows-only in it, so it is driven
here on every system, with a stand-in for pg_ctl and for the server."""

import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from autogpt_desktop import postgres, resources
from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.process import ChildRegistry, ManagedProcess, wait_until
from autogpt_desktop.supervisor import Stack


@pytest.fixture
def data(tmp_path: Path) -> DataDir:
    data = DataDir(tmp_path / "data")
    data.prepare()
    data.postgres.mkdir()
    return data


@pytest.fixture
def bundle(tmp_path: Path) -> Bundle:
    return Bundle(tmp_path / "runtime")


def test_windows_starts_the_server_through_pg_ctl_and_the_others_directly(
    bundle: Bundle, data: DataDir, monkeypatch
):
    monkeypatch.setattr(postgres.sys, "platform", "win32")
    assert isinstance(postgres.process(bundle, data, 25432), postgres.WindowsServer)
    for other in ("darwin", "linux"):
        monkeypatch.setattr(postgres.sys, "platform", other)
        assert type(postgres.process(bundle, data, 25432)) is ManagedProcess


def test_pg_ctl_is_handed_the_port_and_every_limit_unchanged(
    bundle: Bundle, data: DataDir
):
    """What the server gets on the other systems, it gets through -o."""
    server = postgres.process(bundle, data, 25432, resources.POSTGRES_LIMITS).argv
    log = data.logs / "postgres.log"

    command = postgres.pg_ctl_start(server, log)

    pg_ctl = Path(server[0]).with_name(
        Path(server[0]).name.replace("postgres", "pg_ctl")
    )
    assert command[:7] == [
        str(pg_ctl),
        "start",
        "-D",
        str(data.postgres),
        "-W",
        "-l",
        str(log),
    ]
    assert command[7] == "-o"
    assert command[8].split() == server[3:]
    assert command[8] == (
        "-p 25432 -c unix_socket_directories= -c max_connections=50 "
        "-c max_worker_processes=4 -c max_parallel_workers=0 -c autovacuum_max_workers=1"
    )
    assert len(command) == 9


def test_an_option_with_a_space_is_quoted_and_a_quote_is_refused():
    command = postgres.pg_ctl_start(
        ["postgres", "-D", "data", "-c", "search_path=a, b"], Path("log")
    )
    assert command[-1] == '-c "search_path=a, b"'
    with pytest.raises(ValueError, match="double quote"):
        postgres.pg_ctl_start(["postgres", "-D", "data", "-c", 'x="y"'], Path("log"))


def test_the_servers_pid_is_the_first_line_of_postmaster_pid(tmp_path: Path):
    assert postgres.read_postmaster_pid(tmp_path) is None
    (tmp_path / "postmaster.pid").write_text("4242\nC:/data\n1759480000\n15432\n")
    assert postgres.read_postmaster_pid(tmp_path) == 4242
    (tmp_path / "postmaster.pid").write_text("")
    assert postgres.read_postmaster_pid(tmp_path) is None
    (tmp_path / "postmaster.pid").write_text("not a pid\n")
    assert postgres.read_postmaster_pid(tmp_path) is None


def test_a_pid_file_left_by_a_dead_server_is_not_taken_for_the_new_one(
    tmp_path: Path, monkeypatch
):
    """Its id now belongs to nobody, to something that is not PostgreSQL, or
    to a PostgreSQL that serves another data directory. No clock is asked: a
    server is this directory's by what it was started with."""
    monkeypatch.setattr(postgres, "SERVER_NAME", psutil.Process().name().lower()[:6])
    mine, other = tmp_path / "data", tmp_path / "other data"
    stand_in = subprocess.Popen(
        [sys.executable, "-c", SERVER, "-D", str(mine).replace("\\", "/")]
    )
    try:
        assert postgres.running_server(stand_in.pid, mine) is not None
        assert postgres.running_server(stand_in.pid, other) is None
        assert postgres.running_server(None, mine) is None
        assert postgres.running_server(2**31 - 5, mine) is None
        monkeypatch.setattr(postgres, "SERVER_NAME", "postgres")
        assert postgres.running_server(stand_in.pid, mine) is None
    finally:
        stand_in.kill()
        stand_in.wait(10)


# --- the lifecycle, with stand-ins --------------------------------------------

SERVER = "import time; time.sleep(120)"
NO_SUCH_PROCESS = 2**31 - 5


class FakePgCtl:
    """`pg_ctl start`: starts a server that writes its pid, and returns.
    `pg_ctl stop`: ends it. `slow`: the server is started, and writes its
    pid only when `appear` is called."""

    def __init__(self, data: DataDir) -> None:
        self.data = data
        self.calls: list[list[str]] = []
        self.servers: list[subprocess.Popen[bytes]] = []
        self.fails = False
        self.slow = False

    def start(
        self, argv: list[str], output, env: dict[str, str], cwd: Path
    ) -> tuple[int, int]:
        self.calls.append(argv)
        # Its output must not be a pipe: see WindowsServer.start.
        assert output.name.endswith("postgres-pg_ctl.log")
        if self.fails:
            output.write(b"pg_ctl: could not start server\n")
            return NO_SUCH_PROCESS, 1
        command = [
            sys.executable,
            "-c",
            SERVER,
            "-D",
            str(self.data.postgres).replace("\\", "/"),
        ]
        self.servers.append(subprocess.Popen(command, stdin=subprocess.DEVNULL))
        if not self.slow:
            self.appear()
        return NO_SUCH_PROCESS, 0

    def appear(self) -> None:
        (self.data.postgres / "postmaster.pid").write_text(f"{self.servers[-1].pid}\n")

    def __call__(self, argv: list[str], **kwargs) -> subprocess.CompletedProcess[bytes]:
        self.calls.append(argv)
        assert argv[1] == "stop"
        self.servers[-1].terminate()
        self.servers[-1].wait(10)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    def end(self) -> None:
        for server in self.servers:
            server.kill()
            server.wait(10)


@pytest.fixture
def pg_ctl(data: DataDir, monkeypatch):
    fake = FakePgCtl(data)
    monkeypatch.setattr(postgres, "run_tool", fake)
    monkeypatch.setattr(postgres, "start_and_leave", fake.start)
    monkeypatch.setattr(postgres, "SERVER_NAME", psutil.Process().name().lower()[:6])
    monkeypatch.setattr(postgres, "POSTMASTER_PID_SECONDS", 3)
    yield fake
    fake.end()


@pytest.fixture
def server(
    bundle: Bundle, data: DataDir, pg_ctl: FakePgCtl, monkeypatch
) -> postgres.WindowsServer:
    with monkeypatch.context() as patch:  # only to choose the class
        patch.setattr(postgres.sys, "platform", "win32")
        process = postgres.process(bundle, data, 25432, resources.POSTGRES_LIMITS)
    assert isinstance(process, postgres.WindowsServer)
    return process


def test_the_server_not_pg_ctl_is_what_is_watched_and_recorded(
    server: postgres.WindowsServer, data: DataDir, pg_ctl: FakePgCtl, tmp_path: Path
):
    assert server.pid is None and server.exit_code() == 1  # not started yet
    server.start()

    assert server.pid == pg_ctl.servers[0].pid
    assert server.exit_code() is None
    assert pg_ctl.calls[0][8].startswith(
        "-p 25432 -c unix_socket_directories= -c max_connections=50"
    )

    registry = ChildRegistry(tmp_path / "children.json")
    registry.record([server])
    import json

    (entry,) = json.loads(registry.path.read_text())
    assert entry["name"] == "postgres" and entry["pid"] == pg_ctl.servers[0].pid
    assert abs(entry["started"] - psutil.Process(entry["pid"]).create_time()) < 1


def test_it_is_stopped_through_pg_ctl_and_then_gone(
    server: postgres.WindowsServer, pg_ctl: FakePgCtl
):
    server.start()
    server.stop()

    stop = pg_ctl.calls[-1]
    assert stop[1] == "stop" and stop[stop.index("-m") + 1] == "fast"
    assert pg_ctl.servers[0].poll() is not None
    assert server.exit_code() == 1


def test_a_server_that_ignores_pg_ctl_is_killed(
    server: postgres.WindowsServer, pg_ctl: FakePgCtl
):
    server.start()
    server.graceful_stop = lambda managed: None
    server.stop_timeout = 0.5

    server.stop()

    assert pg_ctl.servers[0].wait(10) is not None


def test_a_server_that_ended_is_seen_and_started_again_with_the_same_options(
    server: postgres.WindowsServer, pg_ctl: FakePgCtl
):
    """What the supervisor's watch does with any process that ends."""
    server.start()
    pg_ctl.servers[0].kill()
    pg_ctl.servers[0].wait(10)
    assert wait_until(lambda: server.exit_code() is not None, 10, 0.1)

    server.start()

    assert server.exit_code() is None
    assert server.pid == pg_ctl.servers[1].pid
    starts = [call for call in pg_ctl.calls if call[1] == "start"]
    assert len(starts) == 2 and starts[0] == starts[1]


def test_a_restart_leaves_nothing_of_the_previous_run(
    server: postgres.WindowsServer, pg_ctl: FakePgCtl
):
    server.start()
    server.start()

    assert pg_ctl.servers[0].wait(10) is not None
    assert server.pid == pg_ctl.servers[1].pid


def test_a_start_that_fails_reads_as_a_process_that_exited(
    server: postgres.WindowsServer, data: DataDir, pg_ctl: FakePgCtl, caplog
):
    """So the supervisor stops waiting and points at the log, as it does for
    any server that dies while starting: and the log it points at is where
    pg_ctl's reason is."""
    pg_ctl.fails = True
    server.start()

    assert server.exit_code() == 1 and server.pid is None
    assert "pg_ctl start failed (1): pg_ctl: could not start server" in caplog.text
    log = (data.logs / "postgres.log").read_text(encoding="utf-8")
    assert "pg_ctl start failed (1): pg_ctl: could not start server" in log
    server.stop()  # nothing to stop, and no error


def test_a_stale_pid_file_and_no_new_server_is_a_failed_start(
    server: postgres.WindowsServer, data: DataDir, monkeypatch, caplog
):
    monkeypatch.setattr(
        postgres, "start_and_leave", lambda *arguments: (NO_SUCH_PROCESS, 0)
    )
    monkeypatch.setattr(postgres, "POSTMASTER_PID_SECONDS", 0.3)
    (data.postgres / "postmaster.pid").write_text(f"{psutil.Process().ppid()}\n")

    server.start()

    assert server.exit_code() == 1
    assert "did not start" in caplog.text


def test_a_server_whose_command_prompt_is_gone_is_a_failed_start_at_once(
    server: postgres.WindowsServer, monkeypatch
):
    """pg_ctl said it started the server (with -W it always does), and the
    command prompt it ran it through has already ended: a lock file held, a
    bad option. Nothing is waited for."""
    ended = subprocess.Popen([sys.executable, "-c", "pass"])
    ended.wait(20)
    launcher = psutil.Process()  # any process will do: it is told to have ended
    monkeypatch.setattr(postgres, "start_and_leave", lambda *arguments: (ended.pid, 0))
    monkeypatch.setattr(postgres, "launcher_of", lambda pid, since: Gone(launcher))
    monkeypatch.setattr(postgres, "POSTMASTER_PID_SECONDS", 60)

    started = time_of(server.start)

    assert server.exit_code() == 1
    assert started < 10


class Gone:
    def __init__(self, process: psutil.Process) -> None:
        self.process = process

    def is_running(self) -> bool:
        return False

    def children(self, recursive: bool = False) -> list:
        return []

    def kill(self) -> None:
        pass

    def wait(self, timeout: float) -> None:
        pass


def time_of(call) -> float:
    import time

    began = time.monotonic()
    call()
    return time.monotonic() - began


def test_a_server_slow_to_write_its_pid_is_taken_up_when_it_does(
    server: postgres.WindowsServer, pg_ctl: FakePgCtl, monkeypatch, tmp_path: Path
):
    """A machine short of memory: the start gives up watching, the server
    comes up after all. It is not left as a process nobody knows."""
    pg_ctl.slow = True
    monkeypatch.setattr(postgres, "POSTMASTER_PID_SECONDS", 0.3)
    server.start()
    assert server.pid is None

    pg_ctl.appear()

    assert server.exit_code() is None
    assert server.pid == pg_ctl.servers[0].pid
    registry = ChildRegistry(tmp_path / "children.json")
    registry.record([server])
    assert str(pg_ctl.servers[0].pid) in registry.path.read_text()


def test_a_server_nobody_saw_is_still_stopped_through_pg_ctl(
    server: postgres.WindowsServer, pg_ctl: FakePgCtl, monkeypatch
):
    pg_ctl.slow = True
    monkeypatch.setattr(postgres, "POSTMASTER_PID_SECONDS", 0.3)
    server.start()
    assert server.exit_code() == 1  # given up on
    pg_ctl.appear()

    server.stop()

    assert pg_ctl.calls[-1][1] == "stop"
    assert pg_ctl.servers[0].poll() is not None


def test_a_new_start_first_stops_a_server_nobody_saw(
    server: postgres.WindowsServer, pg_ctl: FakePgCtl, monkeypatch
):
    """Else the new one fails against the lock file the old one holds, three
    times over, and the app says PostgreSQL keeps stopping while it runs."""
    pg_ctl.slow = True
    monkeypatch.setattr(postgres, "POSTMASTER_PID_SECONDS", 0.3)
    server.start()
    assert server.exit_code() == 1

    # What the supervisor's watch does next. The first server has come up
    # meanwhile.
    pg_ctl.slow = False
    lost = pg_ctl.servers[0]
    (pg_ctl.data.postgres / "postmaster.pid").write_text(f"{lost.pid}\n")
    assert psutil.pid_exists(lost.pid)
    server.server = None  # as it was when the watch read its exit code
    server.give_up_at = 0.0
    server.start()

    assert lost.wait(10) is not None
    assert [call[1] for call in pg_ctl.calls] == ["start", "stop", "start"]
    assert server.pid == pg_ctl.servers[1].pid and server.exit_code() is None


def test_the_supervisor_starts_stops_and_restarts_it_like_any_process(
    server: postgres.WindowsServer, bundle: Bundle, data: DataDir, pg_ctl: FakePgCtl
):
    stack = Stack(bundle, data)
    stack.launch([server])
    try:
        stack.await_ready(server, lambda: True, timeout=5)
        pg_ctl.servers[0].kill()
        pg_ctl.servers[0].wait(10)  # the stand-in is this test's child, to be reaped
        assert wait_until(lambda: server.exit_code() is not None, 10, 0.1)
        assert stack.may_restart(server.name)
        server.start()
        stack.record()
        assert server.pid == pg_ctl.servers[1].pid
    finally:
        stack.stop()
    assert pg_ctl.servers[1].wait(10) is not None
    assert not stack.registry.path.exists()
