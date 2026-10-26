"""Child processes that cannot outlive the runtime.

Supervisor in the appliance relies on the container dying with PID 1. A
desktop has no such boundary, so it is rebuilt here:

* Windows: the runtime joins a Job Object with KILL_ON_JOB_CLOSE before it
  starts anything. Children inherit the job, and Windows kills all of them
  the moment the runtime's last handle closes, crash or not.
* POSIX: each child leads its own session, so a terminal's Ctrl+C reaches
  only the runtime (which then stops services in order) and a child can be
  killed together with whatever it started: the process that matters is
  often not the one launched (rabbitmq-server is a shell script that waits
  on the Erlang VM). Every child's (pid, start time) is recorded so that
  the shell, or the next boot, can kill what a dead runtime left running.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("autogpt_desktop")

LOG_ROTATE_BYTES = 20 * 1024 * 1024
WINDOWS = sys.platform == "win32"
# Services are background processes; on Windows they must not flash consoles.
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

# psutil's one-snapshot map of every process's parent. Private to psutil;
# tests/test_hosts.py says so when an upgrade takes it away.
PPID_SNAPSHOT = "_ppid_map"
# Open files and sockets a process may have. macOS gives a process started
# from a terminal 256, and a service host holds five services' worth.
FILE_LIMIT = 8192

_job_handle: int | None = None
_warned: set[str] = set()


@dataclass
class ManagedProcess:
    name: str
    argv: list[str]
    env: dict[str, str]
    cwd: Path
    log_dir: Path
    # Called to ask the process to exit; terminate() is the fallback.
    graceful_stop: Callable[[ManagedProcess], None] | None = None
    stop_timeout: float = 10.0
    # Called before every start, restarts included. It may fail; the process
    # is started regardless.
    before_start: Callable[[], None] | None = None
    popen: subprocess.Popen[bytes] | None = field(default=None, repr=False)
    started_at: float = field(default=0.0, repr=False)
    windows_quote_executable: bool = False

    def start(self) -> None:
        if self.popen:  # a restart: nothing of the previous run may linger
            self.kill()
        if self.before_start:
            try:
                self.before_start()
            except Exception as exc:
                logger.warning(f"preparing to start {self.name} failed: {exc}")
        log_path = self.log_dir / f"{self.name}.log"
        _rotate(log_path)
        logger.info(f"starting {self.name}")
        self.started_at = time.time()
        argv: list[str] | str = self.argv
        if WINDOWS and self.windows_quote_executable:
            # MSYS2 parses apostrophes in an unquoted argv[0] as shell quotes.
            argv = f'"{self.argv[0]}" {subprocess.list2cmdline(self.argv[1:])}'
        # The child inherits its own handle to the log; ours closes here.
        with open(log_path, "ab", buffering=0) as log:
            self.popen = subprocess.Popen(
                argv,
                cwd=self.cwd,
                env=self.env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW,
                start_new_session=not WINDOWS,
            )

    @property
    def pid(self) -> int | None:
        return self.popen.pid if self.popen else None

    def exit_code(self) -> int | None:
        return self.popen.poll() if self.popen else None

    def stop(self) -> None:
        if not self.popen:
            return
        if self.popen.poll() is not None:
            # It left by itself, as a service host does when a service of its
            # ends. What it started (a database query engine, a coding
            # agent's CLI) may not have.
            self.kill()
            return
        logger.info(f"stopping {self.name}")
        # While it is alive and its whole tree can still be seen: once a
        # process in the middle has exited, what it started is out of reach.
        started = (
            _windows_descendants(self.popen.pid, self.started_at) if WINDOWS else []
        )
        try:
            if self.graceful_stop:
                self.graceful_stop(self)
            else:
                self.popen.terminate()
            self.popen.wait(self.stop_timeout)
        except subprocess.TimeoutExpired:
            logger.info(f"{self.name} was still running after {self.stop_timeout:g}s")
        except Exception as exc:  # the process is going away regardless
            logger.warning(f"{self.name} did not stop cleanly ({exc}); killing it")
        # Also after a clean exit: a service's own children (a coding agent's
        # CLI, a browser) must not carry on without it.
        self.kill()
        _kill_all(started)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.popen.wait(5)

    def kill(self) -> None:
        """The process and everything it started: its session on POSIX, its
        descendants on Windows."""
        if not self.popen:
            return
        if sys.platform == "win32":
            started = _windows_descendants(self.popen.pid, self.started_at)
            self.popen.kill()
            _kill_all(started)
            return
        if self.popen.poll() is not None and _pid_exists(self.popen.pid):
            # No process is given an id that a process group still goes by,
            # so nothing of its group is left, and the group this id names
            # now is a stranger's.
            return
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(self.popen.pid, signal.SIGKILL)


def _pid_exists(pid: int) -> bool:
    import psutil

    return psutil.pid_exists(pid)


def _kill_all(processes: list) -> None:
    for process in processes:  # psutil refuses a pid that has been reused
        with contextlib.suppress(Exception):
            process.kill()


def _windows_descendants(pid: int, since: float) -> list:
    """What `pid` started and what those started, whether or not `pid` is
    still alive: a database-manager that has exited leaves its query engine
    running, holding its connections. Windows keeps a dead parent's id on
    its children, and gives that id to no one else while the runtime holds
    the parent's handle; a start time guards the generations below. Only
    living processes are listed, so the search ends at a dead one further
    down the tree."""
    import psutil

    # One snapshot of every process's parent. psutil's public way to the same
    # (Process.ppid) takes a snapshot per process: 18 s on a busy machine.
    snapshot = getattr(psutil, PPID_SNAPSHOT, None)
    if snapshot is None:
        _warn_once(
            f"psutil has no {PPID_SNAPSHOT} any more: stopping and restarting a process "
            "takes a scan per running process, and what a process that already exited "
            "had started is not found. Update process._windows_descendants."
        )
        with contextlib.suppress(Exception):  # psutil's own use of it may be gone too
            return psutil.Process(pid).children(recursive=True)
        return []
    children: dict[int, list[int]] = {}
    for child, parent in snapshot().items():
        children.setdefault(parent, []).append(child)
    found: dict[int, psutil.Process] = {}
    parents = [(pid, since - 1)]
    while parents:
        parent, born = parents.pop()
        for child in children.get(parent, []):
            with contextlib.suppress(psutil.Error):
                process = psutil.Process(child)
                created = process.create_time()
                if created >= born and child != pid and child not in found:
                    found[child] = process
                    parents.append((child, created))
    return list(found.values())


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        logger.warning(message)


def stop_together(processes: list[ManagedProcess]) -> None:
    """Stop processes that do not depend on each other, all at once. One at a
    time, eight services that each take two seconds to exit take sixteen."""
    threads = [
        threading.Thread(target=process.stop, name=f"stop-{process.name}")
        for process in processes
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


class StopEvent:
    """Windows only: how a service host is asked to stop.

    A signal cannot reach a process that has no console, and TerminateProcess
    gives it no chance to clean up. The runtime creates a named event per
    host instead; the host waits on it (servicehost.wait_for_windows_event)
    and the runtime sets it. Manual-reset, so a host that opens it after it
    was set still sees the request.
    """

    def __init__(self, name: str, handle: int) -> None:
        self.name = name
        self.handle = handle

    @classmethod
    def create(cls, service: str) -> StopEvent | None:
        """None where there is no such thing, or when Windows refuses; the
        host is then stopped the old way, by ending its process."""
        if not WINDOWS:
            return None
        name = f"Local\\autogpt-desktop-stop-{os.getpid()}-{service}"
        manual_reset, initially_set = True, False
        handle = _kernel32().CreateEventW(None, manual_reset, initially_set, name)
        if not handle:
            logger.warning(
                f"could not create the stop event for {service}; it will be killed"
            )
            return None
        return cls(name, handle)

    def reset(self) -> None:
        _kernel32().ResetEvent(self.handle)

    def set(self) -> None:
        if not _kernel32().SetEvent(self.handle):
            raise OSError(f"could not set the stop event {self.name}")


def _kernel32():
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateEventW.restype = wintypes.HANDLE
    kernel32.CreateEventW.argtypes = (
        wintypes.LPVOID,
        wintypes.BOOL,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    )
    kernel32.SetEvent.argtypes = (wintypes.HANDLE,)
    kernel32.ResetEvent.argtypes = (wintypes.HANDLE,)
    return kernel32


def adopt_kill_on_exit_job() -> None:
    """Windows only: tie every future child to this process's lifetime."""
    global _job_handle
    if not WINDOWS or _job_handle is not None:
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", ctypes.c_ulonglong * 6),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE
    job_object_extended_limit_information = 9
    if not kernel32.SetInformationJobObject(
        job,
        job_object_extended_limit_information,
        ctypes.byref(limits),
        ctypes.sizeof(limits),
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    _job_handle = job


def raise_file_limit() -> None:
    """POSIX only: lift the soft limit on open files towards FILE_LIMIT, as
    far as the hard limit allows. Children inherit it."""
    if sys.platform == "win32":
        return
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    unlimited = resource.RLIM_INFINITY
    wanted = FILE_LIMIT if hard == unlimited else min(FILE_LIMIT, hard)
    if soft == unlimited or soft >= wanted:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (wanted, hard))
    except (ValueError, OSError) as exc:
        logger.warning(
            f"could not raise the open-file limit from {soft} to {wanted}: {exc}"
        )


def base_env() -> dict[str, str]:
    """What a child needs from the user's environment to function at all
    (system paths, temp dirs, locale on Windows), without inheriting settings
    the user may have exported for something else: AutoGPT variables from a
    source checkout, or ERL_FLAGS from their own Erlang work."""
    keep = (
        "PATH",
        "SYSTEMROOT",
        "SystemRoot",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "NUMBER_OF_PROCESSORS",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
        "PROGRAMDATA",
        "ALLUSERSPROFILE",
        "USERNAME",
        "USER",
        "LOGNAME",
        "LANG",
        "SSL_CERT_FILE",
        "AUTOGPT_DESKTOP_NODE",
    )
    return {name: os.environ[name] for name in keep if name in os.environ}


def run_tool(argv: list[str], **kwargs) -> subprocess.CompletedProcess[bytes]:
    """Run a short-lived helper (initdb, pg_ctl, the Prisma CLI, rabbitmqctl).

    The runtime's stdin is its shutdown channel. A helper that inherited it
    could block on it forever (the Prisma CLI does) or swallow the EOF the
    runtime is waiting for, so helpers always get a null stdin.
    """
    return subprocess.run(
        argv, stdin=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW, **kwargs
    )


class ChildRegistry:
    """Remembers running children on disk so a later boot can clean up after
    a runtime that died without stopping them (POSIX has no Job Objects)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def record(self, processes: list[ManagedProcess]) -> None:
        import psutil

        entries = []
        for process in processes:
            if process.pid is None or process.exit_code() is not None:
                continue
            try:
                started = psutil.Process(process.pid).create_time()
            except psutil.Error:
                continue
            entries.append(
                {"name": process.name, "pid": process.pid, "started": started}
            )
        # Whole or not at all: a runtime killed in the middle of this must
        # not leave half a list, which the next start could not read.
        fresh = self.path.with_name(self.path.name + ".new")
        fresh.write_text(json.dumps(entries), encoding="utf-8")
        os.replace(fresh, self.path)

    def reap_leftovers(self) -> None:
        import psutil

        try:
            entries = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        killed = []
        for entry in entries:
            try:
                process = psutil.Process(entry["pid"])
                if abs(process.create_time() - entry["started"]) > 1:
                    continue
                killed.append(process)
                logger.warning(
                    f"stopping leftover {entry['name']} (pid {entry['pid']})"
                )
                if sys.platform == "win32":
                    for child in process.children(recursive=True):
                        child.kill()
                    process.kill()
                else:
                    os.killpg(entry["pid"], signal.SIGKILL)
            except (psutil.Error, OSError, KeyError, TypeError):
                continue
        # Their ports are about to be checked, and reused if they are free.
        psutil.wait_procs(killed, timeout=10)
        self.path.unlink(missing_ok=True)


def send_posix_signal(sig: int) -> Callable[[ManagedProcess], None]:
    def stop(process: ManagedProcess) -> None:
        assert process.popen
        if sys.platform == "win32":
            process.popen.terminate()
        else:
            os.killpg(process.popen.pid, sig)

    return stop


def listening(port: int, timeout: float = 0.25) -> bool:
    """Whether something accepts connections on a loopback port.

    Asked before any client is pointed at a server that may not be up yet.
    Windows answers a connection to a closed port only after retrying it for
    two seconds, which a database driver or an HTTP client then sits through
    on every attempt: a start that asked nine services in turn noticed the
    last of them up to twenty seconds late. An open port accepts at once (the
    system does, however busy the program is), so a short timeout loses
    nothing."""
    try:
        socket.create_connection(("127.0.0.1", port), timeout=timeout).close()
        return True
    except OSError:
        return False


def wait_until(
    predicate: Callable[[], bool], timeout: float, interval: float = 0.5
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _rotate(path: Path) -> None:
    try:
        if path.stat().st_size > LOG_ROTATE_BYTES:
            os.replace(path, path.with_suffix(path.suffix + ".1"))
    except FileNotFoundError:
        pass
