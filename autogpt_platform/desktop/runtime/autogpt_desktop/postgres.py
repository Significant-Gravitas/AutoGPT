"""PostgreSQL, adapted from single-container/entrypoint.sh `initialize_postgres`.

Differences from the appliance: no peer authentication (Windows has none),
so every role, the frontend's included, authenticates with scram over
loopback TCP; and the port is whatever ports.py picked.

On Windows the server is started through `pg_ctl start`, for every user:
postgres.exe refuses to run for an account with administrative rights (an
elevated shell, a CI runner), and pg_ctl starts it with those rights taken
away. See `WindowsServer`.
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autogpt_desktop.layout import Bundle, DataDir
from autogpt_desktop.process import (
    CREATE_NO_WINDOW,
    LOG_ROTATE_BYTES,
    ManagedProcess,
    _windows_descendants,
    listening,
    run_tool,
    send_posix_signal,
    wait_until,
)

logger = logging.getLogger("autogpt_desktop")

CONFIG_MARKER = "# autogpt-desktop"
# The server writes postmaster.pid before it does anything that takes time
# (crash recovery comes after), so this is not a wait for it to be ready. It
# is how long a start waits to see the server, and how long a server that
# cannot be seen being started is given before it counts as not coming.
POSTMASTER_PID_SECONDS = 30
LAUNCHER_NAME = "cmd"
STOP_TIMEOUT_SECONDS = 30
SERVER_NAME = "postgres"


class IncompatibleDataError(RuntimeError):
    pass


def is_initialized(data: DataDir) -> bool:
    return (data.postgres / "PG_VERSION").is_file() and not _first_run_marker(
        data
    ).exists()


def first_run_completed(data: DataDir) -> None:
    _first_run_marker(data).unlink(missing_ok=True)


def _first_run_marker(data: DataDir) -> Path:
    """Present from before the database is created until its tables exist.

    A first start that is cut short (the app closed, the machine switched
    off) leaves a database nobody has used yet, possibly half-migrated, and
    Prisma will not continue a migration it did not finish. While the marker
    exists there is nothing in the database to lose, so the next start
    creates it again from scratch instead of refusing to run.
    """
    return data.config / "first-run.incomplete"


def check_compatible(bundle: Bundle, data: DataDir) -> None:
    """PostgreSQL only opens data written by its own major version."""
    version_file = data.postgres / "PG_VERSION"
    if not version_file.is_file():
        return
    stored = version_file.read_text(encoding="ascii").strip()
    result = run_tool(
        [str(bundle.postgres_bin("postgres")), "--version"],
        capture_output=True,
        env=_tool_env(bundle),
    )
    match = re.search(r"(\d+)\.", result.stdout.decode(errors="replace"))
    if match and match.group(1) != stored:
        raise IncompatibleDataError(
            f"Your AutoGPT data was created with PostgreSQL {stored}, and this version "
            f"of AutoGPT includes PostgreSQL {match.group(1)}, which cannot open it. "
            "Install the AutoGPT version you used before to keep your data, or move "
            f"{data.postgres} away to start again without it."
        )


def initialize(bundle: Bundle, data: DataDir, password: str) -> None:
    """Create the cluster on first boot."""
    pgdata = data.postgres
    if is_initialized(data):
        return
    marker = _first_run_marker(data)
    if marker.exists():
        shutil.rmtree(pgdata, ignore_errors=True)
    marker.touch()
    password_file = data.run / f"postgres-password.{secrets.token_hex(8)}"
    descriptor = os.open(password_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(password + "\n")
    try:
        run_tool(
            [
                str(bundle.postgres_bin("initdb")),
                f"--pgdata={pgdata}",
                "--username=postgres",
                f"--pwfile={password_file}",
                "--auth=scram-sha-256",
                "--encoding=UTF8",
                "--locale=C",
            ],
            check=True,
            capture_output=True,
            env=_tool_env(bundle),
        )
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or b"").decode(errors="replace") + (
            exc.stderr or b""
        ).decode(errors="replace")
        raise RuntimeError(f"initdb failed:\n{output}") from exc
    finally:
        password_file.unlink(missing_ok=True)
    with open(pgdata / "postgresql.conf", "a", encoding="utf-8") as conf:
        conf.write(
            f"\n{CONFIG_MARKER}\n"
            "listen_addresses = '127.0.0.1'\n"
            "password_encryption = 'scram-sha-256'\n"  # pragma: allowlist secret
            "max_connections = 100\n"
            "shared_buffers = 128MB\n"
        )


def process(
    bundle: Bundle, data: DataDir, port: int, limits: dict[str, str] | None = None
) -> ManagedProcess:
    # The port is passed per boot rather than written to postgresql.conf, so a
    # port that ports.py had to move never leaves the config stale. Unix
    # sockets are off: everything connects over loopback TCP on every OS.
    # `limits` are passed the same way so that they reach installs whose
    # postgresql.conf was written before they existed.
    settings = [f"{name}={value}" for name, value in (limits or {}).items()]
    kind = WindowsServer if sys.platform == "win32" else ManagedProcess
    return kind(
        name="postgres",
        argv=[
            str(bundle.postgres_bin("postgres")),
            "-D",
            str(data.postgres),
            "-p",
            str(port),
            "-c",
            "unix_socket_directories=",
            *[argument for setting in settings for argument in ("-c", setting)],
        ],
        env=_tool_env(bundle),
        cwd=data.postgres,
        log_dir=data.logs,
        graceful_stop=_pg_ctl_stop(bundle, data),
        stop_timeout=STOP_TIMEOUT_SECONDS,
    )


@dataclass
class WindowsServer(ManagedProcess):
    """PostgreSQL on Windows, started by `pg_ctl start`.

    `argv` is the server's own command line, as on the other systems; pg_ctl
    is handed everything after the data directory as the server's options.
    It starts the server through a command prompt and does not stay, so there
    is no child process to hold. What is watched, recorded for the next start
    to clean up, and stopped is the server itself, found through the
    postmaster.pid it writes. It is still inside the runtime's Job Object
    (pg_ctl makes a job of its own only for a process that is in none), so
    it cannot outlive the runtime either way.

    The server is whatever postmaster.pid names that is a PostgreSQL serving
    this data directory, whenever it shows up: a start does not give up on
    one that is slow to write the file, and a stop or a new start does not
    leave one running because this object never saw it.
    """

    server: Any = field(default=None, repr=False)  # psutil.Process
    # The command prompt pg_ctl left, which lives exactly as long as the
    # server it runs: while it is there, the server is still coming.
    launcher: Any = field(default=None, repr=False)  # psutil.Process
    give_up_at: float = field(default=0.0, repr=False)

    @property
    def data(self) -> Path:
        return Path(self.argv[self.argv.index("-D") + 1])

    def start(self) -> None:
        self._end_what_is_running()
        log = self.log_dir / f"{self.name}.log"
        _rotate(log)
        logger.info(f"starting {self.name}")
        self.started_at = time.time()
        self.server, self.launcher, self.give_up_at = None, None, 0.0
        # pg_ctl's own words go to a file, never to a pipe: the command
        # prompt it leaves behind inherits its output, and a pipe would not
        # close, and this call not return, until the server exits. Nor to the
        # server's log, which that command prompt must be able to open.
        said = self.log_dir / f"{self.name}-pg_ctl.log"
        with open(said, "wb", buffering=0) as output:
            pid, code = start_and_leave(
                pg_ctl_start(self.argv, log), output, self.env, self.cwd
            )
        if code != 0:
            words = said.read_text(encoding="utf-8", errors="replace").strip()
            logger.warning(f"pg_ctl start failed ({code}): {words}")
            # Where the supervisor sends the user; nothing else holds it now.
            with contextlib.suppress(OSError), open(
                log, "a", encoding="utf-8"
            ) as stream:
                stream.write(f"pg_ctl start failed ({code}): {words}\n")
            return
        self.launcher = launcher_of(pid, self.started_at)
        self.give_up_at = time.monotonic() + POSTMASTER_PID_SECONDS
        wait_until(
            lambda: self.exit_code() is not None or self.server is not None,
            POSTMASTER_PID_SECONDS,
            interval=0.1,
        )
        if self.exit_code() is not None:
            logger.warning(f"{self.name} did not start; see {log}")
        elif not self.server:
            logger.warning(
                f"{self.name} has written no postmaster.pid yet; still waiting for it"
            )

    def _end_what_is_running(self) -> None:
        """Before a start: the previous run of this object, or a server for
        this data directory that an earlier start lost sight of. It holds
        the lock file the new one needs."""
        self._find_server()
        if self.server and self.server.is_running():
            self.stop()
        else:
            self.kill()

    def _find_server(self) -> None:
        if self.server is None:
            self.server = running_server(read_postmaster_pid(self.data), self.data)

    def _starting(self) -> bool:
        if self.launcher is not None:
            return self.launcher.is_running()
        return time.monotonic() < self.give_up_at

    @property
    def pid(self) -> int | None:
        return self.server.pid if self.server else None

    def exit_code(self) -> int | None:
        """Not its real exit code, which only a parent can read: 1 for a
        server that is gone (or is not coming), whatever it ended with."""
        self._find_server()
        if self.server is not None:
            return None if self.server.is_running() else 1
        return None if self._starting() else 1

    def stop(self) -> None:
        self._find_server()
        if self.exit_code() is None:
            logger.info(f"stopping {self.name}")
            try:
                if self.graceful_stop:
                    self.graceful_stop(self)
                if self.server:
                    self.server.wait(self.stop_timeout)
            except Exception as exc:  # the process is going away regardless
                logger.warning(f"{self.name} did not stop cleanly ({exc}); killing it")
        self.kill()

    def kill(self) -> None:
        """The server, the backends it started, and the command prompt it
        was started through."""
        import psutil

        self.give_up_at = 0.0
        for parent in (self.launcher, self.server):
            if parent is None:
                continue
            started: list = []
            with contextlib.suppress(psutil.Error):
                started = parent.children(recursive=True)
            for process in (parent, *started):  # psutil refuses a reused pid
                with contextlib.suppress(psutil.Error):
                    process.kill()
        for parent in (self.server, self.launcher):
            with contextlib.suppress(Exception):
                if parent is not None:
                    parent.wait(5)


def start_and_leave(
    argv: list[str], output, env: dict[str, str], cwd: Path
) -> tuple[int, int]:
    """Run `pg_ctl start` to its end. Its process id and its exit code: the
    id is how the command prompt it leaves behind is found."""
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=cwd,
        creationflags=CREATE_NO_WINDOW,
    )
    return process.pid, process.wait()


def launcher_of(pg_ctl_pid: int, since: float):
    """The command prompt `pg_ctl start` ran the server through, or None
    when it cannot be seen (it has already ended, or this is not Windows)."""
    import psutil

    for process in _windows_descendants(pg_ctl_pid, since):
        with contextlib.suppress(psutil.Error):
            if process.name().lower().startswith(LAUNCHER_NAME):
                return process
    return None


def pg_ctl_start(server_argv: list[str], log: Path) -> list[str]:
    """`pg_ctl start` for a server that would be run as `server_argv`
    (postgres -D <data> <options>). -W: do not wait for it to accept
    connections; the supervisor does that, alongside the other servers."""
    server, flag, data, *options = server_argv
    assert flag == "-D", server_argv
    pg_ctl = str(Path(server).with_name(f"pg_ctl{Path(server).suffix}"))
    return [
        pg_ctl,
        "start",
        "-D",
        data,
        "-W",
        "-l",
        str(log),
        "-o",
        " ".join(_quoted(option) for option in options),
    ]


def _quoted(option: str) -> str:
    """pg_ctl pastes the options into a command line as they are."""
    if '"' in option:
        raise ValueError(
            f"a PostgreSQL option may not contain a double quote: {option}"
        )
    return f'"{option}"' if not option or any(c.isspace() for c in option) else option


def read_postmaster_pid(data: Path) -> int | None:
    """The first line of postmaster.pid: the server's process id. (pg_ctl
    itself only knows the command prompt it started the server through.)"""
    try:
        text = (data / "postmaster.pid").read_text(encoding="ascii", errors="replace")
        return int(text.splitlines()[0].strip())
    except (OSError, ValueError, IndexError):
        return None


def running_server(pid: int | None, data: Path):
    """The process `pid` if it is a PostgreSQL server for `data`. A
    postmaster.pid left by a server that was killed names a process that is
    gone, or by now somebody else's: something that is not PostgreSQL, or a
    PostgreSQL that serves another data directory."""
    import psutil

    if pid is None:
        return None
    try:
        process = psutil.Process(pid)
        if not process.name().lower().startswith(SERVER_NAME):
            return None
        return process if serves(process, data) else None
    except psutil.Error:
        return None


def serves(process, data: Path) -> bool:
    """Whether the server's command line names `data` (pg_ctl passes it as
    -D, spelled with forward slashes)."""
    import psutil

    wanted = _spelling(str(data))
    try:
        return any(_spelling(argument) == wanted for argument in process.cmdline())
    except psutil.AccessDenied:
        return True  # it is a PostgreSQL, and its pid is in this directory's file


def _spelling(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _rotate(path: Path) -> None:
    with contextlib.suppress(OSError):
        if path.stat().st_size > LOG_ROTATE_BYTES:
            os.replace(path, path.with_suffix(path.suffix + ".1"))


def is_ready(port: int, password: str) -> bool:
    if not listening(port):
        return False
    import psycopg2

    try:
        connection = psycopg2.connect(
            host="127.0.0.1",
            port=port,
            user="postgres",
            password=password,
            dbname="postgres",
            connect_timeout=3,
        )
    except psycopg2.Error:
        return False
    connection.close()
    return True


def _pg_ctl_stop(bundle: Bundle, data: DataDir):
    # Fast shutdown checkpoints and exits; on Windows pg_ctl is also the only
    # way to signal the postmaster, which ignores TerminateProcess etiquette.
    def stop(managed: ManagedProcess) -> None:
        result = run_tool(
            [
                str(bundle.postgres_bin("pg_ctl")),
                "stop",
                "-D",
                str(data.postgres),
                "-m",
                "fast",
                "-w",
                "-t",
                "25",
            ],
            capture_output=True,
            env=_tool_env(bundle),
        )
        if result.returncode != 0 and managed.popen:
            send_posix_signal(signal.SIGINT)(managed)
        elif result.returncode != 0:
            words = (result.stdout + result.stderr).decode(errors="replace").strip()
            logger.warning(f"pg_ctl stop failed ({result.returncode}): {words}")

    return stop


def _tool_env(bundle: Bundle) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PG", "LC_"))
    }
    env["LANG"] = "C"
    if sys.platform == "darwin":
        lib = str(bundle.root / "postgres" / "lib")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = lib
    elif sys.platform != "win32":
        lib = str(bundle.root / "postgres" / "lib")
        env["LD_LIBRARY_PATH"] = lib
    return env
