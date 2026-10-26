"""Boot an assembled runtime, check that it serves, stop it, check it is gone.

    <runtime>/python/python smoke_test.py <runtime> [--timeout 600] [--quick]
                                          [--profile compact|balanced|isolated]
                                          [--data-dir DIR] [--keep] [--read-only]
                                          [--restart-budget SECONDS]

Run it with the bundle's own interpreter (it uses psutil, psycopg2 and redis
from the bundle). This is the shell's contract exercised without the shell:
start the runtime as the bundle's manifest says to, read JSON events from
stdout until `ready`, close stdin, expect exit code 0, no process it started
left running, and no file in the bundle changed.

Before anything is started, the backend in the bundle is checked against
what the service host depends on (runtime/tests/backend_contract.py).

While the first start is up, the stack itself is put to work: every service
answers, an agent runs to the right result, memory is inside the profile's
budget, the process hosting the executor is killed with a run in flight and
comes back to finish that run and take another. After every stop, each service host's log must show that
it was asked, cleaned up, and left by itself.

It starts the runtime four times on one data directory, because the owner
account's promises, and what becomes of agent runs when the app closes, are
about restarts:

  1. a fresh install: the first account to sign up is an admin at once, and
     a second sign-up is refused. A run loses its message (its executor is
     killed and the queue purged) and stays "running" in nobody's hands. It
     is stopped the plain way (stdin closes) with a run in flight;
  2. a restart: the owner is still an admin, and the frontend now refuses
     sign-ups with its own "not allowed" message; runs left "running" by the
     first start are settled, its stale lock is gone, the run that was in
     flight is finished, and the run that lost its message is FAILED about
     a minute after the app is up. The runtime is stopped the way "Stop
     and quit" stops it, with another run in the middle of a long step: that
     run is cancelled and TERMINATED before PostgreSQL stops;
  3. a restart with a password-reset file waiting, and nobody an admin: the
     old password stops working, the new one works, the owner is admin again.
     The stopped run has stayed stopped. It is stopped the same way
     again, this time with the executor frozen so that the run cannot be
     cancelled: the run is ended in the database after the wait;
  4. a restart: that run has stayed ended, and the executor skipped its
     message. It is stopped the same way once more, executor frozen, and
     stdin is closed three seconds later, as the shell does when its user
     quits a second time: the wait is cut short, and the run is ended all
     the same. This run has also lost its lock, as one does that the
     computer slept through: it is still counted, and still ended.

While a run is in flight the runtime must say so (`activity` events), which
is what the shell asks its question by.

Both of those stops also have an AutoPilot turn in progress. No model is
needed for one: from the second start on, the backend is told to answer
turns with upstream's own stand-in (`CHAT_TEST_MODE`, copilot/sdk/dummy.py),
which takes fourteen seconds over a reply when asked to (seven words, two
seconds each). That is not long, so a turn is started only when the stop
is next: the one to be cancelled immediately before the request to stop,
and the one whose executor is frozen is frozen the moment it has taken the
turn. Everything around the model is the real thing: the queue, the
executor, its lock, its cancel exchange. The turn that is cancelled must be
heard to cancel; the one whose executor is frozen must not be run again by
the next start, which must close its chat; and that chat must take the next
message, as the chat of the cancelled turn must.

Every start must come up on the address the first one had: sessions and OAuth
redirect URLs hang off it.

Every start is timed, by the runtime's own account of it (its `timing`
events: autogpt_desktop/timing.py), and the table is printed at the end. A
restart that takes longer than `--restart-budget` fails the run: see
RESTART_BUDGET_SECONDS.

`--quick` stops after the first.

The bundle is the installed app's, so a run may leave nothing in it: no file
or directory added, changed or removed (the Next server's caches are probed
on purpose), and nothing fetched. The runtime is started with every proxy
variable pointing at a closed port, which the Prisma CLI (the one part of the
stack that would download something, an engine, when it found none it liked)
inherits: a download fails the migration instead of quietly succeeding.
`--read-only` (macOS and Linux) goes further and takes write permission off
the whole bundle for the run, as a system-wide install or a mounted AppImage
has it, then also fails on any service log that mentions being refused.

Without `--data-dir` the data directory is a temporary one, removed when the
run ends, however it ends (a signal that asks it to end included); `--keep`
leaves it. One given with `--data-dir` is the caller's and is left alone.
Either way, the space-free links the runtime made outside the data directory
during the run are removed (autogpt_desktop/rabbitmq.py).
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import http.cookiejar
import io
import json
import logging
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psutil
import psycopg2

# The shell kills a runtime that takes a minute to stop (src/runtime.js); a
# stop that would not survive that with room to spare is a failure here.
STOP_BUDGET_SECONDS = 30
# How long the runtime waits for runs it was asked to stop
# (supervisor.CANCEL_WAIT_SECONDS). A stop that ends runs may take that much
# longer; one whose runs cancel must not come near it.
CANCEL_WAIT_SECONDS = 20

OWNER_EMAIL = "owner@smoke.test"
OWNER_PASSWORD = "first-password-of-the-owner"
NEW_OWNER_PASSWORD = "second-password-of-the-owner"
# Cheap, read-only, and behind `requires_admin_user`
# (backend/api/features/admin/store_admin_routes.py).
ADMIN_ROUTE = "/_agpt/api/store/admin/listings?page_size=1"
# CalculatorBlock (backend/blocks/maths.py): operation, a, b -> result.
CALCULATOR_BLOCK = "b1ab9b19-67a6-406d-abf5-2dba76d00c79"
# CountdownTimerBlock (backend/blocks/time_blocks.py): seconds, input_message
# -> output_message. Long enough to still be running when its executor dies.
COUNTDOWN_BLOCK = "d67a9c52-5e4e-11e2-bcfd-0800200c9a71"
IN_FLIGHT_SECONDS = 20
# One step that outlasts everything done here: a run that is in the middle
# of it for as long as it is left alone.
LONG_STEP_SECONDS = 1800
# After the app is up, a run that nothing will finish is failed
# (runs.ORPHAN_SWEEP_AFTER_SECONDS): not before, and not much later.
ORPHAN_NOT_BEFORE_SECONDS = 45
# After a start is up: any run it was going to resume is running by now.
RESUMED_BY_SECONDS = 75
ORPHAN_SECONDS = 150
# backend/copilot/config.py `test_mode` and backend/copilot/sdk/dummy.py: no
# model is called, and a message with this word in it is answered slowly.
STAND_IN_MODEL = "CHAT_TEST_MODE=true"
# How often the runtime counts what is in progress (supervisor.Stack.watch).
ACTIVITY_POLL_SECONDS = 2
SLOW_TURN = "__test_slow_response__"
AGENT_RUN_SECONDS = 180
RESTART_SECONDS = 240
# Memory is read this long after ready, once start-up work has drained. On a
# fast machine that is before the skills catalog publish begins
# (supervisor.py: a minute); on a slow one the publisher may be running, so
# it is listed and left out of the total.
SETTLE_SECONDS = 45
CATALOG = "skills-catalog"
CATALOG_NOT_BEFORE_SECONDS = 55
CATALOG_SECONDS = 420
STALE_LOCK = "exec_lock:left-by-the-smoke-test"
# From starting the runtime to its `ready`, for a start that finds its data
# there (a first start also creates the database, and is not held to this).
# Measured on Windows x64, 24 cores, 2026-10-03: 22 seconds with 6 GB of
# memory free, every time; 25 to 36 with eight service hosts (the isolated
# profile) and half a gigabyte free. A start that makes each of its waits one
# after the other takes 40 to 54 on the same machine. The default is more
# than five times the first figure, because the machines that run this in CI
# have three or four cores, less memory than the stack wants, and neighbours,
# and none of them has been timed yet: it catches a start that has come to
# wait a minute for something, not one that lost ten seconds. On a quiet
# machine, hold it tighter: `--restart-budget 35`.
RESTART_BUDGET_SECONDS = 120
PHASES = ("config", "database", "cache", "queue", "migrations", "services", "frontend")

# Nothing listens on the discard port. What the runtime starts with these in
# its environment cannot reach the network through a proxy-aware client; its
# own services are on loopback and exempt.
CLOSED_PROXY = "http://127.0.0.1:9"
NO_WAY_OUT = {
    "HTTPS_PROXY": CLOSED_PROXY,
    "HTTP_PROXY": CLOSED_PROXY,
    "NO_PROXY": "127.0.0.1,localhost",
}
if sys.platform != "win32":  # where the two spellings are two variables
    NO_WAY_OUT.update({name.lower(): value for name, value in NO_WAY_OUT.items()})
# The same run on every machine: where Claude Code is signed in, the runtime
# would ask the user's CLI and start one more service host for AutoPilot
# (autogpt_desktop/claude_code.py). build/autopilot_turn.py tests that.
NO_SIGN_IN = {"AUTOGPT_CLAUDE_CODE": "off"}
# A public image, small, and of a width the Next configuration allows.
OPTIMISED_IMAGE = "/_next/image?url=%2Fplaceholder.png&w=64&q=75"
# What a service says when it is refused a write (--read-only).
REFUSED_WRITE = re.compile(
    r"EACCES|EROFS|EPERM|Read-only file system|Permission denied|Failed to write image to cache"
)

# The whole stack, idle: memory unique to its processes (USS; on macOS the
# physical footprint), in MB. Measured on Windows x64, 2026-10-03: balanced
# and compact (three service hosts) 2,900-3,000; isolated (eight) 5,900. The
# budget is that plus room for an OS that shares less, and for upstream
# growing a little between syncs; it is there to catch a fourth interpreter
# or a doubled import, not a few megabytes.
MEMORY_BUDGET_MB = {"compact": 3800, "balanced": 3800, "isolated": 7200}

CONTRACT_SCRIPT = (
    Path(__file__).resolve().parents[1] / "runtime" / "tests" / "backend_contract.py"
)


class Install:
    """What one start leaves for the next ones to check against."""

    def __init__(self, quick: bool) -> None:
        self.quick = quick
        self.address: str | None = None
        self.owner: Browser | None = None
        self.profile: str | None = None
        self.runtime: psutil.Process | None = None
        self.ready_at = 0.0
        self.ready_wall = 0.0
        # Agent runs of the first start, by what the next start must have
        # made of them.
        self.left_running: dict[str, str] = {}
        # A run still going when the first start was stopped.
        self.in_flight: str | None = None
        self.quit_mid_run = False
        # What the runtime says, for as long as it runs.
        self.said: Said | None = None
        # Set by a start's checks: its stop is the one that ends this run.
        self.stopping: Stopping | None = None
        # Service hosts that were frozen, and so could not answer a stop.
        self.frozen: set[str] = set()
        # The chat session whose AutoPilot turn each stop ended, and how
        # many messages it has been sent over all the starts.
        self.chat: str | None = None
        self.messages_sent = 0
        # Whether AutoPilot took a turn during this start.
        self.turn_taken = False
        # Runs of earlier starts, by path: what each must be now, and why.
        self.ended: dict[str, tuple[str, str]] = {}
        # A run whose message was lost, for the next start to fail.
        self.orphan: str | None = None
        # --read-only: puts the bundle's permissions back (see `unwritable`).
        self.give_back: Callable[[], None] | None = None
        # The runtime's last `timing` event of the start that is running.
        self.timing: dict | None = None
        self.starts: list[Start] = []


class Start:
    """How long one start took: to `ready` as seen from outside, and the
    runtime's own phases."""

    def __init__(self, name: str, to_ready: float | None, timing: dict | None) -> None:
        self.name = name
        self.to_ready = to_ready
        self.phases: dict[str, float] = (timing or {}).get("phases", {})

    @property
    def restart(self) -> bool:
        return self.name != fresh_install.__name__


Checks = Callable[[str, Path, Install], list[str]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime", type=Path)
    parser.add_argument("--timeout", type=int, default=600, help="seconds, per start")
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="use this (empty) data directory, e.g. one with a space in its path",
    )
    parser.add_argument(
        "--keep", action="store_true", help="do not remove the temporary data directory"
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="macOS and Linux: run with the bundle unwritable, as an installed app has it",
    )
    parser.add_argument("--quick", action="store_true", help="one start, no restarts")
    parser.add_argument(
        "--restart-budget",
        type=float,
        default=RESTART_BUDGET_SECONDS,
        help="seconds a restart may take to become ready (default: %(default)s)",
    )
    parser.add_argument(
        "--profile",
        default=os.environ.get("AUTOGPT_DESKTOP_PROFILE"),
        help="set AUTOGPT_DESKTOP_PROFILE in the data directory's settings.env",
    )
    args = parser.parse_args()
    # Service logs carry emoji, which a Windows console or pipe cannot print.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(errors="replace")
    runtime = args.runtime.resolve()
    sys.path.append(str(runtime))  # for autogpt_desktop.apps, as the bundle has it
    refused = cannot_run_read_only() if args.read_only else None
    if refused:
        print(refused)
        return 2
    end_on_signals()
    data = args.data_dir or Path(tempfile.mkdtemp(prefix="autogpt-smoke-"))
    aliases = Aliases(runtime, data)
    try:
        with (
            unwritable(runtime) if args.read_only else contextlib.nullcontext()
        ) as give_back:
            return smoke(runtime, data, args, give_back)
    finally:  # a failed check, an exception, Ctrl+C, a signal: the run is over
        aliases.remove_new()
        if not args.data_dir and not args.keep:
            remove_data(data)


END_SIGNALS = [
    getattr(signal, name)
    for name in ("SIGINT", "SIGTERM", "SIGHUP")
    if hasattr(signal, name)
]


def end_on_signals() -> None:
    """Being told to end (a CI job cancelled, `kill`) is a Ctrl+C: Python's
    own answer to SIGTERM is to die at once, with the stack still running,
    the data directory left and, with --read-only, the bundle unwritable."""

    def interrupt(number: int, frame: object) -> None:
        raise KeyboardInterrupt(signal.Signals(number).name)

    for number in END_SIGNALS:
        if number == signal.SIGINT:  # which already does this
            continue
        # A signal that is being ignored stays ignored: `nohup` runs a
        # program that way so that it outlives the terminal it was started
        # from, and a handler here would end the run when that one closes.
        if signal.getsignal(number) is signal.SIG_IGN:
            continue
        signal.signal(number, interrupt)


@contextlib.contextmanager
def deaf_to_signals():
    """For the one piece of clearing up that must not be cut short by a
    second Ctrl+C or the signal that follows the first."""
    previous = {}
    for number in END_SIGNALS:
        with contextlib.suppress(ValueError, OSError):  # not the main thread
            previous[number] = signal.signal(number, signal.SIG_IGN)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def smoke(
    runtime: Path,
    data: Path,
    args: argparse.Namespace,
    give_back: Callable[[], None] | None = None,
) -> int:
    if args.profile:
        choose_profile(data, args.profile)
    stale = left_by_earlier_runs(runtime)
    bundle_before = snapshot(runtime)
    starts: list[Checks] = [fresh_install]
    if not args.quick:
        starts += [
            restart,
            restart_with_a_password_reset,
            restart_after_runs_were_stopped,
        ]

    started = time.monotonic()
    failures = backend_contract(runtime)
    install = Install(args.quick)
    install.give_back = give_back
    for number, checks in enumerate(starts, start=1):
        if failures:
            break
        print(
            f"\n== start {number} of {len(starts)}: {checks.__name__.replace('_', ' ')}"
        )
        failures += run(runtime, data, args.timeout, checks, install)
    failures += changed_files(bundle_before, snapshot(runtime))
    failures += stale
    print_starts(install.starts)
    failures += slow_restarts(install.starts, args.restart_budget)
    if args.read_only:
        failures += refused_writes(data)

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"  - {failure}")
        print_logs(data)
        return 1
    print(f"\nOK in {time.monotonic() - started:.0f}s")
    return 0


def run(
    runtime: Path, data: Path, timeout: int, checks: Checks, install: Install
) -> list[str]:
    """One start and stop of the runtime, with `checks` made while it is up."""
    started = time.monotonic()
    process = subprocess.Popen(
        shell_command(runtime),
        cwd=runtime,
        env={
            **os.environ,
            **NO_WAY_OUT,
            **NO_SIGN_IN,
            "AUTOGPT_DESKTOP_DATA_DIR": str(data),
        },
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    failures: list[str] = []
    install.runtime = psutil.Process(process.pid)
    family = Family(install.runtime, data, runtime)
    try:
        install.timing = None
        install.stopping, install.frozen, install.turn_taken = None, set(), False
        install.said = Said(process, started, install)
        url = install.said.wait_for_ready(timeout)
        install.ready_at, install.ready_wall = time.monotonic(), time.time()
        to_ready = install.ready_at - started if url else None
        install.starts.append(Start(checks.__name__, to_ready, install.timing))
        failures += [] if url else ["the runtime never reported ready"]
        install.quit_mid_run = False
        if url:
            failures += checked(checks, url, data, install)
        failures += (
            stop_the_runs(process, runtime, data, install)
            if install.stopping
            else stop(process)
        )
        if url:
            failures += stopped_gracefully(data, install)
    finally:
        if process.poll() is None:  # the smoke test itself is being ended
            process.kill()
            if install.give_back:  # first: what follows is slow, and may be cut short
                install.give_back()
        failures += left_running(family)
    return failures


class Family:
    """Every process the runtime started, directly or not, noted while it
    ran: once the runtime is gone, what it left behind has no parent to be
    found through. Told apart from anything else that runs from the bundle
    at the same time (a developer's unit tests use its interpreter)."""

    def __init__(self, runtime: psutil.Process, data: Path, bundle: Path) -> None:
        self.runtime = runtime
        self.born = runtime.create_time()
        self.data = data
        self.bundle = bundle
        self.seen: set[psutil.Process] = set()
        self.done = threading.Event()
        self.watcher = threading.Thread(target=self.watch, name="family", daemon=True)
        self.watcher.start()

    def watch(self) -> None:
        while not self.done.wait(1):
            self.look()

    def look(self) -> None:
        with contextlib.suppress(psutil.Error):
            self.seen.update(self.runtime.children(recursive=True))
        for detached in nobodys_children(self.data, self.runtime):
            with contextlib.suppress(psutil.Error):
                # With the command prompt pg_ctl started it through.
                launcher = [
                    p for p in detached.parents()[:1] if born_under(p, self.runtime)
                ]
                self.seen.update([detached, *launcher, *detached.children()])

    def still_running(self) -> list[psutil.Process]:
        self.done.set()
        self.watcher.join(10)
        living = [process for process in self.seen if is_alive(process)]
        return [*living, *self.strays(set(living))]

    def strays(self, known: set[psutil.Process]) -> list[psutil.Process]:
        """What left the family tree before it was looked at: a process that
        makes itself a daemon (epmd does, left to itself) is nobody's child
        within milliseconds. It runs from the bundle, was started after the
        runtime, and is of this run by what it was started with: the data
        directory, or a link to it, in its environment, on its command line
        or as its working directory, which is true of every process of the
        stack. Who its parent is now says nothing (an orphan is handed to
        whatever the system has for that), and a developer's unit tests, also
        running from the bundle, have none of the three."""
        marks = [str(self.data), *(str(link) for link in links_to(self.data))]
        found = []
        for process in psutil.process_iter():
            with contextlib.suppress(psutil.Error, OSError):
                if process in known or process.pid == os.getpid():
                    continue
                if process.create_time() < self.born or not self.runs_from_bundle(
                    process
                ):
                    continue
                if mentions(process, marks):
                    found.append(process)
        return found

    def runs_from_bundle(self, process: psutil.Process) -> bool:
        executable = process.exe()
        return bool(executable) and Path(executable).resolve().is_relative_to(
            self.bundle.resolve()
        )


def mentions(process: psutil.Process, marks: list[str]) -> bool:
    """Whether a process was started with any of `marks` in its environment,
    on its command line, or as the directory it works in."""
    said: list[str] = []
    for ask in (
        lambda: process.environ().values(),
        process.cmdline,
        lambda: [process.cwd()],
    ):
        with contextlib.suppress(psutil.Error, OSError):
            said += ask()
    return any(mark in value for mark in marks for value in said)


def is_alive(process: psutil.Process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def left_running(family: Family) -> list[str]:
    """What outlived the runtime, ended here so that the machine is not left
    dirty."""
    time.sleep(2)
    leftovers = family.still_running()
    if not leftovers:
        return []
    names = set()
    for leftover in leftovers:
        with contextlib.suppress(psutil.Error):
            names.add(leftover.name())
            leftover.kill()
    psutil.wait_procs(leftovers, timeout=10)
    return [f"processes left running: {sorted(names)}"]


def remove_data(data: Path) -> None:
    """The temporary data directory."""
    for _ in range(10):  # a virus scanner may still hold a file just closed
        with contextlib.suppress(OSError):
            if sys.version_info >= (3, 12):
                shutil.rmtree(data, onexc=_remove_read_only)
            else:
                shutil.rmtree(
                    data,
                    onerror=lambda remove, path, info: _remove_read_only(
                        remove, path, info[1]
                    ),
                )
        if not data.exists():
            return
        time.sleep(1)
    print(f"could not remove {data}")


def _remove_read_only(remove: Callable, path: str, error: BaseException) -> None:
    """Windows refuses to delete a read-only file, and RabbitMQ's cookie is
    one (written for its owner to read, and nobody to change)."""
    os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    remove(path)


class Aliases:
    """What a run keeps outside the data directory: the space-free links
    RabbitMQ is given to the data directory and to the bundle when their
    paths have a space in them (autogpt_desktop/rabbitmq.py; symlinks, or
    junctions on Windows). Those that appear during the run and lead to
    either are this run's, and go with it; so do the folders made to hold
    them. A link that was there before, or that leads anywhere else, is
    another install's or another run's."""

    def __init__(self, runtime: Path, data: Path) -> None:
        self.targets = [runtime, data]
        self.roots = alias_roots()
        self.new_folders = [
            folder
            for root in self.roots
            for folder in (root, root.parent)
            if not folder.exists()
        ]
        self.before = set(self.links())

    def links(self) -> list[Path]:
        return [link for root in self.roots for link in links_in(root)]

    def remove_new(self) -> None:
        for link in set(self.links()) - self.before:
            if any(leads_into(link, target) for target in self.targets):
                remove_link(link)
        for folder in self.new_folders:
            with contextlib.suppress(OSError):
                folder.rmdir()  # only when nothing else is in it


def alias_roots() -> list[Path]:
    from autogpt_desktop import rabbitmq

    return rabbitmq.alias_roots()


def links_in(root: Path) -> list[Path]:
    try:
        return [entry for entry in root.iterdir() if is_link(entry)]
    except OSError:
        return []


def links_to(directory: Path) -> list[Path]:
    """The links, in any of the places the runtime keeps them, that lead
    into `directory`."""
    return [
        link
        for root in alias_roots()
        for link in links_in(root)
        if leads_into(link, directory)
    ]


def is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    return getattr(status, "st_reparse_tag", 0) == 0xA0000003


def leads_into(link: Path, directory: Path) -> bool:
    """By where the link really goes, not by how either path is spelled
    (/var and /private/var on macOS, an 8.3 name on Windows)."""
    destination = Path(os.path.realpath(link))
    return destination.is_relative_to(Path(os.path.realpath(directory)))


def remove_link(link: Path) -> None:
    with contextlib.suppress(OSError):
        if link.is_symlink():
            link.unlink()
        else:
            link.rmdir()  # a junction


def cannot_run_read_only() -> str | None:
    if sys.platform == "win32":
        return "--read-only is for macOS and Linux; Windows installs per user, writable"
    if os.geteuid() == 0:
        return "--read-only proves nothing as root, whom permissions do not stop"
    return None


@contextlib.contextmanager
def unwritable(runtime: Path):
    """Write permission taken off every file and directory of the bundle,
    and put back as it was. Yields the putting back, for a run that is being
    ended to do first of all; done once, and not interruptible: a bundle
    left unwritable fails the next build until someone repairs it by hand."""
    modes = {}
    for path in [runtime, *runtime.rglob("*")]:
        if not path.is_symlink():
            modes[path] = stat.S_IMODE(path.lstat().st_mode)
    given_back = False

    def give_back() -> None:
        nonlocal given_back
        if given_back:
            return
        with deaf_to_signals():
            # Top down: a directory must be writable before what is in it can be.
            for path, mode in modes.items():
                with contextlib.suppress(OSError):
                    path.chmod(mode)
            given_back = True

    try:
        for path, mode in modes.items():
            path.chmod(mode & ~0o222)
        yield give_back
    finally:
        give_back()


def refused_writes(data: Path) -> list[str]:
    """Lines of the logs (the services', in any folder under logs/, and the
    runtime's own) that speak of a refused write."""
    found = []
    logs = data / "logs"
    for log in sorted(logs.rglob("*.log")):
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        refused = [line for line in lines if REFUSED_WRITE.search(line)]
        if refused:
            found.append(
                f"{log.relative_to(logs).as_posix()} has {len(refused)} line(s) about a refused "
                f"write with the bundle read-only, the first: {refused[0][:300]}"
            )
    return found


def left_by_earlier_runs(runtime: Path) -> list[str]:
    """What a run of an older runtime wrote into this bundle. The snapshot
    is taken after it, so nothing else would notice, and it would be packaged
    with the app."""
    engines = runtime / "prisma" / "node_modules" / "@prisma" / "engines"
    found = [
        *[path for path in [runtime / "frontend" / ".next" / "cache"] if path.exists()],
        *engines.glob("*.node"),
    ]
    if not found:
        return []
    listed = ", ".join(str(path.relative_to(runtime)) for path in found)
    print(f"== left in the bundle by an earlier run: {listed}")
    return [
        f"the bundle already held what an earlier run wrote into it ({listed}); "
        "`build_runtime.py --only seal` removes it"
    ]


def print_starts(starts: list[Start]) -> None:
    """Seconds: to `ready` from outside, then each phase by the runtime's
    own clock. Phases overlap; they do not add up."""
    if not starts:
        return
    print("\n== how long each start took, in seconds")
    print(
        f"  {'start':30} {'to ready':>8} " + " ".join(f"{name:>10}" for name in PHASES)
    )
    for start in starts:
        to_ready = "never" if start.to_ready is None else f"{start.to_ready:.1f}"
        phases = " ".join(
            f"{start.phases[name]:10.1f}" if name in start.phases else f"{'-':>10}"
            for name in PHASES
        )
        print(f"  {start.name.replace('_', ' '):30} {to_ready:>8} {phases}")


def slow_restarts(starts: list[Start], budget: float) -> list[str]:
    """A restart has nothing to create and nothing to migrate: what it takes
    is what every start of the app costs its user."""
    return [
        f"the start `{start.name.replace('_', ' ')}` took {start.to_ready:.0f}s to become "
        f"ready, and a restart's budget is {budget:g}s (--restart-budget). Its phases, in "
        f"seconds: {phases_of(start)}"
        for start in starts
        if start.restart and start.to_ready is not None and start.to_ready > budget
    ]


def phases_of(start: Start) -> str:
    listed = ", ".join(f"{name} {spent:.1f}" for name, spent in start.phases.items())
    return listed or "the runtime reported none"


def checked(checks: Checks, url: str, data: Path, install: Install) -> list[str]:
    """A check that blows up is a failure, and the runtime is still stopped."""
    try:
        return checks(url, data, install)
    except Exception as exc:
        return [f"{checks.__name__} raised {type(exc).__name__}: {exc}"]


def stop(process: subprocess.Popen[str]) -> list[str]:
    assert process.stdin
    asked = time.monotonic()
    process.stdin.close()
    try:
        code = process.wait(STOP_BUDGET_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        return [
            f"the runtime did not stop within {STOP_BUDGET_SECONDS}s of stdin closing"
        ]
    print(f"  stopped in {time.monotonic() - asked:.1f}s")
    return [f"the runtime exited with code {code}"] if code != 0 else []


@dataclass
class Stopping:
    """A run in the middle of a long step, which the stop is to end."""

    run: str
    host: str
    # An AutoPilot turn in progress as well (its session), and its host.
    # None until the stop itself starts one, as it does when the turn is to
    # be cancelled: see `turn_in_progress`.
    turn: str | None
    turn_host: str
    # Who sends that turn's message.
    owner: Browser
    # Whether their executors can be told: not when they were frozen.
    cancels: bool = True
    # Asked a second time, a few seconds into the wait, by closing stdin.
    hurried: bool = False


class RunWatcher:
    """A run's status and its lock, asked many times a second for as long
    as the database and the cache answer: what they last said is how the
    runtime left them."""

    def __init__(self, data: Path, run: str) -> None:
        ports = json.loads((data / "config" / "ports.json").read_text(encoding="utf-8"))
        self.run = run.rpartition("/")[2]
        self.database = database(data)
        self.cache = valkey(data, ports)
        self.status: str | None = None
        self.locked: int | None = None
        # One thread each: a server that has gone takes seconds to say so.
        self.threads = [
            threading.Thread(target=watch, name="run-watch", daemon=True)
            for watch in (self.watch_status, self.watch_lock)
        ]
        for thread in self.threads:
            thread.start()

    def watch_status(self) -> None:
        cursor = self.database.cursor()
        with contextlib.suppress(Exception):  # until PostgreSQL has stopped
            while True:
                cursor.execute(
                    'SELECT "executionStatus"::text FROM platform."AgentGraphExecution" WHERE id = %s',
                    (self.run,),
                )
                self.status = cursor.fetchone()[0]
                time.sleep(0.03)

    def watch_lock(self) -> None:
        with contextlib.suppress(Exception):  # until the cache has stopped
            while True:
                self.locked = int(self.cache.exists(f"exec_lock:{self.run}"))
                time.sleep(0.03)

    def last(self) -> tuple[str | None, int | None]:
        for thread in self.threads:
            thread.join(30)
        return self.status, self.locked


def stop_the_runs(
    process: subprocess.Popen[str], runtime: Path, data: Path, install: Install
) -> list[str]:
    """Stop the runtime the way the shell does when its user chose to stop
    what is running (src/runtime.js): the request on stdin, which stays
    open. The run must be durably TERMINATED before any later startup can
    reconcile it, and the runtime must have told the shell how long this may take."""
    assert process.stdin and install.said and install.stopping
    stopping, report = install.stopping, Report()
    watcher = RunWatcher(data, stopping.run)
    if stopping.cancels:
        # Now, and not a check earlier: the turn lasts fourteen seconds.
        stopping.turn = turn_in_progress(data, install, stopping.owner, report)
    turn_id = None
    if stopping.turn:
        from autogpt_desktop import runs

        ports = json.loads((data / "config" / "ports.json").read_text(encoding="utf-8"))
        value = valkey(data, ports).hget(
            f"{runs.SESSION_META_PREFIX}{{{stopping.turn}}}", "turn_id"
        )
        turn_id = value.decode() if value else None
        report.expect(
            "the stopped AutoPilot turn has an original identity", bool(turn_id), True
        )
    heard = len(install.said.events)
    asked = time.monotonic()
    process.stdin.write(json.dumps({"command": "stop", "drain": True}) + "\n")
    process.stdin.flush()
    if stopping.hurried:
        time.sleep(3)
        process.stdin.close()
    try:
        code = process.wait(CANCEL_WAIT_SECONDS + STOP_BUDGET_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        return [
            f"the runtime did not stop within {CANCEL_WAIT_SECONDS + STOP_BUDGET_SECONDS}s of being asked to stop the runs"
        ]
    took = time.monotonic() - asked
    print(f"  stopped, runs and all, in {took:.1f}s")
    report.expect("the runtime's exit code after stopping the runs", code, 0)
    asked_for = [
        event.get("grace_seconds")
        for event in install.said.events[heard:]
        if event.get("step") == "stopping"
    ]
    enough = (
        bool(asked_for) and asked_for[0] >= CANCEL_WAIT_SECONDS + STOP_BUDGET_SECONDS
    )
    report.expect(
        "the shell was asked for the time that takes", enough, True, shown=asked_for
    )
    status, locked = watcher.last()
    report.note("the stopped run, when PostgreSQL last answered the watcher", status)
    status, unfinished = persisted_run(runtime, data, stopping.run)
    report.expect(
        "the stopped run, persisted before any startup reconciliation",
        status,
        "TERMINATED",
    )
    report.expect("its persisted node executions still marked running", unfinished, 0)
    check_stop_records(data, stopping, turn_id, report)
    run = stopping.run.rpartition("/")[2]
    log = (data / "logs" / f"{stopping.host}.log").read_text(
        encoding="utf-8", errors="replace"
    )
    if stopping.cancels:
        report.expect(
            "its executor was asked to cancel it",
            f"Received cancel for {run}" in log,
            True,
        )
        report.expect(
            "...and did, well inside the wait",
            took < CANCEL_WAIT_SECONDS,
            True,
            shown=f"{took:.1f}s",
        )
        report.expect("its lock, when the cache last answered", locked, 0)
    elif stopping.hurried:
        report.expect(
            "asked a second time, the runtime waited no longer",
            took < CANCEL_WAIT_SECONDS,
            True,
            shown=f"{took:.1f}s",
        )
    else:
        waited = took >= CANCEL_WAIT_SECONDS
        report.expect(
            "a run that could not be cancelled was waited for, and then ended",
            waited,
            True,
            shown=f"{took:.1f}s",
        )
        report.note("its lock, when the cache last answered", locked)
    install.ended[stopping.run] = ("TERMINATED", "stopped as the app closed")
    if stopping.turn:
        said = (data / "logs" / f"{stopping.turn_host}.log").read_text(
            encoding="utf-8", errors="replace"
        )
        if stopping.cancels:
            heard = f"Received cancel for {stopping.turn}" in said
            report.expect(
                "the AutoPilot turn's executor was asked to cancel it", heard, True
            )
    return report.failures


def persisted_run(runtime: Path, data: Path, run: str) -> tuple[str, int]:
    """Read a stopped cluster's copy without starting any AutoGPT service.
    The supervisor may commit less than one polling interval before it
    stops PostgreSQL; the watcher's last answer cannot prove that commit."""
    from autogpt_desktop import ports, postgres
    from autogpt_desktop.layout import Bundle, DataDir
    from autogpt_desktop.maintenance import DataLock
    from autogpt_desktop.process import wait_until

    with DataLock(data), tempfile.TemporaryDirectory(
        prefix="autogpt-stop-oracle-"
    ) as temporary:
        source = data / "postgres"
        if (source / "postmaster.pid").exists():
            raise RuntimeError(
                "Cannot inspect persisted runs while PostgreSQL is running"
            )
        if any(is_link(item) for item in [source, *source.rglob("*")]):
            raise RuntimeError("The smoke database must not contain external links")
        copied = DataDir(Path(temporary))
        copied.prepare()
        shutil.copytree(source, copied.postgres)
        port = ports.allocate(copied.ports_file, ("postgres",))["postgres"]
        server = postgres.process(
            Bundle(runtime),
            copied,
            port,
            {
                "default_transaction_read_only": "on",
                "shared_buffers": "16MB",
                "max_connections": "20",
            },
        )
        connection = None

        def connected() -> bool:
            nonlocal connection
            try:
                connection = psycopg2.connect(
                    host="127.0.0.1",
                    port=port,
                    user="postgres",
                    password=secrets(data)["POSTGRES_PASSWORD"],
                    dbname="postgres",
                    connect_timeout=2,
                    options="-c default_transaction_read_only=on -c statement_timeout=10000",
                )
                return True
            except psycopg2.OperationalError:
                return False

        try:
            server.start()
            if not wait_until(connected, 30, interval=0.1):
                raise RuntimeError(
                    "The isolated persisted-run database did not become ready"
                )
            assert connection
            with connection.cursor() as cursor:
                cursor.execute(
                    'SELECT e."executionStatus"::text, '
                    '(SELECT count(*) FROM platform."AgentNodeExecution" n '
                    'WHERE n."agentGraphExecutionId" = e.id AND n."executionStatus" = \'RUNNING\') '
                    'FROM platform."AgentGraphExecution" e WHERE e.id = %s',
                    (run.rpartition("/")[2],),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RuntimeError(
                        "The stopped run is missing from the persisted database"
                    )
                return row[0], row[1]
        finally:
            if connection:
                connection.close()
            server.stop()
            if server.exit_code() is None:
                raise RuntimeError("The isolated persisted-run database did not stop")


def check_stop_records(
    data: Path, stopping: Stopping, turn_id: str | None, report: Report
) -> None:
    stopped = data / "run" / "stopped-runs.json"
    ended = data / "run" / "ended-turns.json"
    if not stopping.turn:
        report.expect("the graph-only stop record is used up", stopped.exists(), False)
        return
    report.expect(
        "the stopped turn's identity is retained for startup reconciliation",
        stopped.exists() or ended.exists(),
        True,
    )
    if stopped.exists():
        report.expect(
            "the pending stop names exactly the original run and turn",
            json.loads(stopped.read_text()),
            {
                "graph_runs": [stopping.run.rpartition("/")[2]],
                "copilot_turns": [stopping.turn],
                "copilot_turn_ids": {stopping.turn: turn_id},
            },
        )
    if ended.exists():
        report.expect(
            "the pending chat closure names exactly the original turn",
            json.loads(ended.read_text()),
            [
                {"session_id": stopping.turn, "turn_id": turn_id},
            ],
        )


def choose_profile(data: Path, profile: str) -> None:
    settings = data / "config" / "settings.env"
    settings.parent.mkdir(parents=True, exist_ok=True)
    with open(settings, "a", encoding="utf-8") as stream:
        stream.write(f"AUTOGPT_DESKTOP_PROFILE={profile}\n")


def backend_contract(runtime: Path) -> list[str]:
    """The bundle's backend still has the shape the service host depends on.
    Checked in an interpreter of its own, as a service would import it."""
    print("== the backend against the service host's contract")
    result = subprocess.run(
        [shell_command(runtime)[0], "-B", str(CONTRACT_SCRIPT), str(runtime)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdin=subprocess.DEVNULL,
    )
    lines = result.stdout.splitlines()  # among the backend's own logging
    broken = [line for line in lines if line.startswith("BROKEN:")]
    print("  " + "\n  ".join([*broken, *lines[-1:]]))
    if result.returncode == 0:
        return []
    report = "\n".join(broken) or (result.stdout + result.stderr).strip()[-2000:]
    return [f"the backend broke the service host's contract:\n{report}"]


def stopped_gracefully(data: Path, install: Install) -> list[str]:
    """Each service host was asked, ran its services' cleanup and left by
    itself. On Windows that is the named event doing what a signal cannot;
    before the host, a Windows service was simply ended.

    Three cleanups are allowed not to finish inside the stop: the executor's
    while a run is in flight (it waits for the run, and a quit does not),
    the scheduler's while one of its jobs is, and the copilot executor's
    once it has taken a turn (it then starts every idle worker of its pool
    in order to clean it up, after its consumers have stopped and its turns
    are settled). The services' own last words ("Cleanup done",
    "Terminated") are upstream's, logged when run() returns after cleanup;
    they are required of the services that do."""
    from autogpt_desktop import apps

    report = Report()
    may_overrun = {"scheduler"} | ({"executor"} if install.quit_mid_run else set())
    may_overrun |= {"copilot-executor"} if install.turn_taken else set()
    last_words = {
        "database-manager",
        "batch-executor",
        "notification",
        *apps.API_SERVERS,
    }
    for group in apps.layout(install.profile != "isolated"):
        if group.name in install.frozen:
            continue  # it could not answer, and was ended
        log = (data / "logs" / f"{group.name}.log").read_text(
            encoding="utf-8", errors="replace"
        )
        _, asked, after = log.rpartition(f"[host {group.name}] stopping: asked to stop")
        report.expect(f"{group.name} was asked to stop", bool(asked), True)
        left = f"[host {group.name}] stopped with code 0" in after
        report.expect(f"{group.name} left by itself", left, True)
        cleaned = [
            name
            for name in group.services
            if re.search(rf"\] {re.escape(name)} \([^)]*\) cleaned up in", after)
        ]
        required = [name for name in group.services if name not in may_overrun]
        finished = [name for name in cleaned if name in required]
        report.expect(
            f"{group.name}: cleanup finished for", finished, required, shown=cleaned
        )
        if last_words & set(group.services):
            own_words = "Cleanup done" in after and "Terminated" in after
            report.expect(
                f'{group.name}: "Cleanup done" and "Terminated" logged', own_words, True
            )
    return report.failures


Snapshot = dict[str, tuple[object, ...]]


def snapshot(runtime: Path) -> Snapshot:
    """Every entry of the bundle: a file by size and modification time; a
    directory by its modification time, which moves when anything is created
    in it or deleted from it, so a file that came and went during the run
    shows; a link by where it points; anything else (a socket, a pipe) by
    what it is."""
    entries: Snapshot = {os.curdir + os.sep: ("directory", runtime.lstat().st_mtime_ns)}
    for path in runtime.rglob("*"):
        status = path.lstat()
        name = str(path.relative_to(runtime))
        if is_link(path):
            entries[name] = ("link", os.readlink(path))
        elif stat.S_ISREG(status.st_mode):
            entries[name] = ("file", status.st_size, status.st_mtime_ns)
        elif stat.S_ISDIR(status.st_mode):
            entries[name + os.sep] = ("directory", status.st_mtime_ns)
        else:
            entries[name] = ("other", stat.S_IFMT(status.st_mode))
    return entries


def changed_files(before: Snapshot, after: Snapshot) -> list[str]:
    """Running the app must leave its bundle exactly as it was: installed, the
    bundle may be read-only, and on macOS a changed file breaks the code
    signature."""
    changed = sorted(
        name
        for name in before.keys() | after.keys()
        if before.get(name) != after.get(name)
    )
    if not changed:
        return []
    listed = ", ".join(changed[:5]) + (" ..." if len(changed) > 5 else "")
    return [
        f"the run changed {len(changed)} file(s) or directories in the bundle: {listed}"
    ]


def shell_command(runtime: Path) -> list[str]:
    """What the shell runs: the bundle's manifest.json (src/paths.js)."""
    manifest = json.loads((runtime / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest.get(sys.platform, manifest["default"])
    return [str(runtime / entry["command"]), *entry["args"]]


class Said:
    """What the runtime says on stdout, read for as long as it runs."""

    def __init__(
        self, process: subprocess.Popen[str], started: float, install: Install
    ) -> None:
        self.process = process
        self.started = started
        self.install = install
        self.events: list[dict] = []
        self.url: str | None = None
        self.settled = threading.Event()
        threading.Thread(target=self.read, name="runtime-events", daemon=True).start()

    def read(self) -> None:
        assert self.process.stdout
        try:
            for line in self.process.stdout:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    self.hear(event)
        except ValueError:
            pass  # the pipe was closed under the reader
        finally:
            self.settled.set()

    def hear(self, event: dict) -> None:
        self.events.append(event)
        elapsed = time.monotonic() - self.started
        if event.get("event") == "timing":
            self.install.timing = event
            print(
                f"{elapsed:6.1f}s timing: {event.get('phase')} {event.get('seconds')} s"
            )
            return
        if event.get("event") == "activity":
            print(f"{elapsed:6.1f}s activity: {self.activity()}")
            return
        print(
            f"{elapsed:6.1f}s {event.get('event')}: {event.get('message') or event.get('url')}"
        )
        if event.get("step") == "profile":
            named = re.search(r"Using the (\w+) profile", str(event.get("message")))
            self.install.profile = named.group(1) if named else None
        if event.get("event") == "ready":
            self.url = event["url"]
            self.settled.set()
        if event.get("event") == "error" and event.get("fatal"):
            self.settled.set()

    def wait_for_ready(self, timeout: int) -> str | None:
        self.settled.wait(timeout)
        return self.url

    def activity(self) -> tuple[int, int]:
        """(agent runs, AutoPilot turns) in progress, as the runtime last
        counted them. It says nothing until there is something."""
        counts = [event for event in self.events if event.get("event") == "activity"]
        return (
            (counts[-1]["graph_runs"], counts[-1]["copilot_turns"])
            if counts
            else (0, 0)
        )


def fresh_install(url: str, data: Path, install: Install) -> list[str]:
    report = Report()
    install.address = url
    report.failures += probe(url)
    # The app's address is predictable, so a page open in the user's browser
    # must not be able to claim the owner account with a blind request.
    status, _ = Browser(url, origin="https://example.com").sign_up(
        "page@smoke.test", "password-of-a-web-page"
    )
    report.expect("a sign-up sent by a page on another site is refused", status, 403)
    with contextlib.closing(database(data)) as connection:
        report.expect(
            "accounts before the first sign-up", count_accounts(connection), 0
        )

    owner = Browser(url)
    status, _ = owner.sign_up(OWNER_EMAIL, OWNER_PASSWORD)
    report.expect("the first sign-up", status, 200)
    report.expect(
        "the first account's role, in its first session", owner.role(), "admin"
    )
    report.expect(
        "an admin route without a token", Browser(url).get(ADMIN_ROUTE)[0], 401
    )
    report.expect("an admin route with the owner's token", owner.admin_route(), 200)

    status, _ = Browser(url).sign_up("second@smoke.test", "password-of-a-second-person")
    report.expect(
        "a second sign-up is refused", status in (403, 422), True, shown=status
    )
    with contextlib.closing(database(data)) as connection:
        report.expect(
            "accounts after the refused sign-up", count_accounts(connection), 1
        )
        report.expect(
            "role of an account inserted as postgres (as a migration would)",
            insert_as_postgres(connection),
            None,
        )
    bundled_tools(install, report)
    work_the_stack(data, install, owner, report)
    return report.failures


def bundled_tools(install: Install, report: Report) -> None:
    """ffmpeg, as a backend service finds it: by name, the way the backend's
    own subprocess calls and yt-dlp look for it, and through imageio-ffmpeg."""
    assert install.runtime
    host = next(iter(hosts(install).values()))
    env = host.environ()
    bundle = Path(host.exe()).resolve().parents[1 if sys.platform == "win32" else 2]
    found = shutil.which("ffmpeg", path=env.get("PATH"))
    inside = bool(found) and Path(found or "").resolve().is_relative_to(
        bundle / "tools" / "bin"
    )
    report.expect(
        "a service finds `ffmpeg` on its PATH, in the bundle", inside, True, shown=found
    )
    if not found:
        return
    banner = subprocess.run(
        [found, "-version"],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    first = (banner.stdout.splitlines() or [banner.stderr.strip()])[0]
    report.expect(
        "...and it runs", first.startswith("ffmpeg version"), True, shown=first[:70]
    )
    asked = subprocess.run(
        [
            host.exe(),
            "-B",
            "-c",
            "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())",
        ],
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    same = asked.returncode == 0 and os.path.samefile(asked.stdout.strip(), found)
    shown = asked.stdout.strip() or asked.stderr.strip()[-300:]
    report.expect("imageio-ffmpeg uses the same one", same, True, shown=shown)


def work_the_stack(
    data: Path, install: Install, owner: Browser, report: Report
) -> None:
    """The service hosts under real use: see the module docstring."""
    from autogpt_desktop import apps

    assert install.runtime
    ports = json.loads((data / "config" / "ports.json").read_text(encoding="utf-8"))
    groups = apps.layout(install.profile != "isolated")
    expected = sorted(group.name for group in groups)
    report.expect(
        f"service hosts of the {install.profile} profile",
        sorted(hosts(install)),
        expected,
    )
    health = apps.health_urls(ports)
    for name, address in health.items():
        report.expect(
            f"{name} answers at {address.split('/', 3)[3] or '/'}",
            fetch_status(address),
            200,
        )
    with contextlib.closing(database(data)) as connection:
        report.expect(
            "PostgreSQL's connection limit",
            setting(connection, "max_connections"),
            "50",
        )
        report.note("PostgreSQL connections in use", connections(connection))

    first = run_agent(owner, report, "an agent run", 6, 7)

    settle = install.ready_at + SETTLE_SECONDS - time.monotonic()
    time.sleep(max(0.0, settle))
    catalog_log = data / "logs" / "skills-catalog.log"
    if time.monotonic() - install.ready_at < CATALOG_NOT_BEFORE_SECONDS:
        report.expect(
            "the skills catalog publish has not started yet",
            catalog_log.exists(),
            False,
        )
    used = memory_table(install.runtime, nobodys_children(data, install.runtime))
    budget = MEMORY_BUDGET_MB.get(str(install.profile), 0)
    report.expect(
        f"memory within the {install.profile} budget of {budget} MB",
        used <= budget,
        True,
        shown=f"{used:.0f} MB",
    )
    open_files(install, report)

    # The executor dies with a run in flight. Its lock on the run outlives
    # it; unless that is cleared before the new executor starts, the run's
    # redelivered message is dropped and the run stays "running" for ever.
    executor_host = next(group for group in groups if "executor" in group.services)
    cache = valkey(data, ports)
    report.note(
        "keys in the cache, all of which a count of the locks walks", cache.dbsize()
    )
    in_flight = start_run(
        owner, report, "a run in flight", countdown(IN_FLIGHT_SECONDS)
    )
    if in_flight:
        running = wait_for(
            lambda: run_status(owner, in_flight) == "RUNNING", AGENT_RUN_SECONDS
        )
        report.expect("a run is in flight when its executor is killed", running, True)
        counted(
            install, report, "the runtime says one agent run is in progress", (1, 0)
        )
    cache.set(STALE_LOCK, "an executor that is gone", ex=300)
    restart_host(
        install,
        executor_host.name,
        [health[name] for name in executor_host.services],
        report,
    )
    report.expect(
        "the dead executor's lock was cleared before the restart",
        cache.exists(STALE_LOCK),
        0,
    )
    if in_flight:
        what = "the run that was in flight, picked up by the new executor,"
        finish_run(
            owner, report, what, in_flight, "output_message", "the timer ran out"
        )
        counted(install, report, "...and none once it has ended", (0, 0))
    second = run_agent(owner, report, "an agent run after the restart", 5, 9)

    if not install.quick:
        use_the_stand_in_model(data)
        skills_catalog(data, install, report)
        lose_a_run(data, install, owner, report)
        leave_runs_running(data, install, first, second)
        cache.set(STALE_LOCK, "an executor that is gone", ex=300)
        # And a real one: the app is about to quit with this run in flight.
        install.in_flight = start_run(
            owner, report, "a run in flight at quit", countdown(IN_FLIGHT_SECONDS)
        )
        if install.in_flight:
            running = wait_for(
                lambda: run_status(owner, install.in_flight or "") == "RUNNING",
                AGENT_RUN_SECONDS,
            )
            report.expect("a run is in flight when the app quits", running, True)
            install.quit_mid_run = running


def open_files(install: Install, report: Report) -> None:
    """What each service host holds open, for the record: a merged host has
    several services' files and sockets under one per-process limit (the
    runtime raises the soft limit on POSIX; process.raise_file_limit)."""
    for name, host in sorted(hosts(install).items()):
        with contextlib.suppress(psutil.Error):
            if sys.platform == "win32":
                report.note(f"{name}: open handles", host.num_handles())
            else:
                report.note(f"{name}: open file descriptors", host.num_fds())
    if sys.platform != "win32":
        import resource

        report.note(
            "the smoke test's own open-file limit",
            resource.getrlimit(resource.RLIMIT_NOFILE),
        )


def hosts(install: Install) -> dict[str, psutil.Process]:
    """The runtime's service hosts, by the name the supervisor gave each."""
    assert install.runtime
    found = {}
    for child in install.runtime.children():
        with contextlib.suppress(psutil.Error):
            command = child.cmdline()
            if "--name" in command and "servicehost" in " ".join(command):
                found[command[command.index("--name") + 1]] = child
    return found


def calculator(a: int, b: int) -> dict:
    return {
        "block_id": CALCULATOR_BLOCK,
        "input_default": {"operation": "Multiply", "a": a, "b": b},
    }


def countdown(seconds: int) -> dict:
    return {
        "block_id": COUNTDOWN_BLOCK,
        "input_default": {"seconds": seconds, "input_message": "the timer ran out"},
    }


def run_agent(owner: Browser, report: Report, what: str, a: int, b: int) -> str | None:
    """A graph of one calculator block, run through the REST API
    (rest -> RabbitMQ -> executor -> database-manager), must give a * b.
    Returns the run's id."""
    run = start_run(owner, report, what, calculator(a, b))
    if run:
        finish_run(owner, report, what, run, "result", float(a * b))
    return run.rpartition("/")[2] if run else None


def start_run(owner: Browser, report: Report, what: str, node: dict) -> str | None:
    """Create a graph of one block and run it. Returns the path the run is
    read back from."""
    node = {**node, "metadata": {"position": {"x": 0, "y": 0}}}
    graph = {
        "name": f"Smoke: {what}",
        "description": "One block.",
        "nodes": [node],
        "links": [],
    }
    status, created = owner.api("/graphs", {"graph": graph})
    if status != 200:
        report.expect(
            f"{what}: the graph is created", status, 200, shown=f"{status} {created}"
        )
        return None
    path = f"/graphs/{created['id']}"
    start = {"inputs": {}, "credentials_inputs": {}}
    status, run = owner.api(f"{path}/execute/{created['version']}", start)
    if status != 200:
        report.expect(
            f"{what}: the run is accepted", status, 200, shown=f"{status} {run}"
        )
        return None
    return f"{path}/executions/{run['id']}"


def run_status(owner: Browser, run: str) -> object:
    return owner.api(run)[1].get("status")


def finish_run(
    owner: Browser, report: Report, what: str, run: str, output: str, expected: object
) -> None:
    started = time.monotonic()
    wait_for(
        lambda: run_status(owner, run) not in ("QUEUED", "RUNNING", "INCOMPLETE", None),
        AGENT_RUN_SECONDS,
    )
    _, execution = owner.api(run)
    took = f"{execution.get('status')} after {time.monotonic() - started:.0f}s"
    report.expect(f"{what} ends", execution.get("status"), "COMPLETED", shown=took)
    results = [
        node.get("output_data", {}).get(output)
        for node in execution.get("node_executions", [])
    ]
    report.expect(f"{what}: its {output}", results, [[expected]])


def counted(
    install: Install, report: Report, what: str, expected: tuple[int, int]
) -> None:
    """The runtime counts what is running every two seconds."""
    assert install.said
    said = install.said
    wait_for(lambda: said.activity() == expected, 15)
    report.expect(what, said.activity(), expected)


def executor_host(install: Install, service: str = "executor") -> str:
    """The process that hosts the executor, in this profile."""
    from autogpt_desktop import apps

    groups = apps.layout(install.profile != "isolated")
    return next(group.name for group in groups if service in group.services)


def use_the_stand_in_model(data: Path) -> None:
    """From the next start on: see SLOW_TURN."""
    with open(data / "config" / "settings.env", "a", encoding="utf-8") as settings:
        settings.write(f"{STAND_IN_MODEL}\n")


def turn_in_progress(
    data: Path,
    install: Install,
    owner: Browser,
    report: Report,
    freeze: Callable[[], None] | None = None,
) -> str | None:
    """Send AutoPilot a message it takes fourteen seconds to answer, and
    wait until the turn is counted beside the one run in progress. Returns
    its session: one chat for every start, so that each message after the
    first also shows that a chat whose turn was ended by a stop takes the
    next message as any chat does.

    `freeze` is called the moment the executor has taken the turn (its lock
    is there), before the count is waited for: a turn that is to be found
    unfinished must not be given the time to finish."""
    from autogpt_desktop import runs

    if not install.chat:
        status, session = owner.api("/chat/sessions", {})
        report.expect("a chat session is created", status, 200)
        if status != 200:
            return None
        install.chat = str(session.get("id"))
    install.turn_taken = True
    install.messages_sent += 1
    # Not the same words twice: a message repeated before it was answered is
    # taken for the same message sent again, and is not queued.
    message = f"{SLOW_TURN} (message {install.messages_sent})"
    threading.Thread(
        target=send_message, args=(owner, install.chat, message), daemon=True
    ).start()
    what = (
        "a chat whose last turn a stop ended takes the next message"
        if install.messages_sent > 1
        else "an AutoPilot turn"
    )
    if freeze:
        ports = json.loads((data / "config" / "ports.json").read_text(encoding="utf-8"))
        cache, lock = valkey(data, ports), runs.lock_key(
            "copilot-executor", install.chat
        )
        taken = wait_often_for(lambda: bool(cache.exists(lock)), 60)
        freeze()
        report.expect(
            f"{what}: its executor takes the turn, and is frozen with it", taken, True
        )
    counted(install, report, f"{what}: counted as in progress beside the run", (1, 1))
    return install.chat


def wait_often_for(done: Callable[[], bool], timeout: float) -> bool:
    """`wait_for`, asking fifty times a second."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if done():
            return True
        time.sleep(0.02)
    return done()


def send_message(owner: Browser, chat: str, message: str) -> None:
    """Post a message and read the reply for as long as it comes: what the
    window does. The app may stop under it."""
    _, minted = owner.get("/api/auth/token")
    request = urllib.request.Request(
        f"{owner.url}/_agpt/api/chat/sessions/{chat}/stream",
        data=json.dumps({"message": message, "is_user_message": True}).encode(),
        headers={
            "Origin": owner.url,
            "Content-Type": "application/json",
            "Authorization": f"Bearer {minted.get('token')}",
        },
    )
    with contextlib.suppress(OSError, http.client.HTTPException), owner.opener.open(
        request, timeout=120
    ) as reply:
        for _ in reply:
            pass


def stopped_turns_are_not_run_again(
    data: Path, install: Install, report: Report
) -> None:
    """A turn that a stop ended was taken once and never again: cancelled,
    its message was settled; ended with its executor frozen, its message
    was taken off the queue before this start's executor could be handed it.
    Startup consumes either saved stop intent or a pending chat closure,
    idempotently closing only the original turn."""
    if not install.chat:
        return
    listed = data / "run" / "ended-turns.json"
    closed = "closed the chat(s) of 1 AutoPilot turn(s)"
    log = data / "logs" / f"{executor_host(install, 'copilot-executor')}.log"
    stopped = data / "run" / "stopped-runs.json"
    done = wait_for(lambda: not listed.exists() and not stopped.exists(), 90)
    report.expect(
        "the saved stop intent and pending chat closure are used up on restart",
        done,
        True,
    )
    said = log.read_text(encoding="utf-8", errors="replace").count(closed)
    report.expect(
        "the copilot host settled each stopped turn exactly once",
        said,
        install.messages_sent,
    )
    from autogpt_desktop import runs

    ports = json.loads((data / "config" / "ports.json").read_text(encoding="utf-8"))
    with contextlib.closing(broker(data, ports)) as connection:
        channel = connection.channel()

        def queue():
            return channel.queue_declare(runs.COPILOT_TURN_QUEUE, passive=True).method

        consuming = wait_for(lambda: queue().consumer_count >= 1, 60)
        report.expect(
            "the AutoPilot executor is taking turns from its queue", consuming, True
        )
        time.sleep(5)  # a message left there would have been handed to it by now
        report.expect("turns waiting on the queue", queue().message_count, 0)
    said = log.read_text(encoding="utf-8", errors="replace")
    taken = said.count(f"Acquired cluster lock for {install.chat}")
    report.expect(
        "turns taken up in that chat, over every start: one a message",
        taken,
        install.messages_sent,
    )


def in_a_long_step(
    owner: Browser, report: Report, install: Install, what: str
) -> str | None:
    """Start a run whose one step outlasts the test, and wait until it is
    in the middle of it and counted."""
    run = start_run(owner, report, what, countdown(LONG_STEP_SECONDS))
    if not run:
        return None
    running = wait_for(lambda: run_status(owner, run) == "RUNNING", AGENT_RUN_SECONDS)
    report.expect(f"{what} is in the middle of its step", running, True)
    counted(install, report, "...and counted as in progress", (1, 0))
    return run


def lose_a_run(data: Path, install: Install, owner: Browser, report: Report) -> None:
    """A run that nothing will ever finish: its executor dies, and its
    message, back on the queue, is purged before a new executor takes it. It
    stays "running"; the next start must fail it, a minute after it is up.
    Meanwhile it is in nobody's hands, and must not be counted as in
    progress for being "running" in the database."""
    from autogpt_desktop import apps, runs

    run = in_a_long_step(
        owner, report, install, "a run that is about to lose its message"
    )
    if not run:
        return
    ports = json.loads((data / "config" / "ports.json").read_text(encoding="utf-8"))
    name = executor_host(install)
    health = apps.health_urls(ports)
    answering = [
        health[service]
        for group in apps.layout(install.profile != "isolated")
        if group.name == name
        for service in group.services
    ]
    victim = hosts(install)[name]
    killed = time.monotonic()
    victim.kill()
    with contextlib.closing(broker(data, ports)) as connection:
        channel = connection.channel()

        def waiting() -> int:
            return channel.queue_declare(
                runs.GRAPH_RUN_QUEUE, passive=True
            ).method.message_count

        wait_for(lambda: waiting() >= 1, 30)
        purged = channel.queue_purge(runs.GRAPH_RUN_QUEUE).method.message_count
    report.expect(
        "its message, back on the queue when the executor died, was purged", purged, 1
    )
    host_is_back(install, name, victim, killed, answering, report)
    time.sleep(5)  # were the message still there, the new executor would have it by now
    report.expect(
        "the run that lost its message is still marked",
        run_status(owner, run),
        "RUNNING",
    )
    counted(install, report, "...and nobody holds it", (0, 0))
    install.orphan = run


def broker(data: Path, ports: dict[str, int]):
    import pika

    logging.getLogger("pika").setLevel(logging.CRITICAL)
    secret = secrets(data)
    credentials = pika.PlainCredentials(
        secret["RABBITMQ_DEFAULT_USER"], secret["RABBITMQ_DEFAULT_PASS"]
    )
    return pika.BlockingConnection(
        pika.ConnectionParameters(
            host="127.0.0.1",
            port=ports["rabbitmq"],
            credentials=credentials,
            socket_timeout=10,
        )
    )


def orphan_is_failed(install: Install, owner: Browser, report: Report) -> None:
    """The run that lost its message during the last start (`lose_a_run`).
    That start was stopped the plain way: a stop of the runs would have
    ended this one with the rest."""
    run, install.orphan = install.orphan, None
    if not run:
        return
    failed = wait_for(lambda: run_status(owner, run) == "FAILED", ORPHAN_SECONDS)
    after = time.monotonic() - install.ready_at
    report.expect(
        "the run that lost its message is failed",
        failed,
        True,
        shown=run_status(owner, run),
    )
    in_time = ORPHAN_NOT_BEFORE_SECONDS <= after <= ORPHAN_SECONDS
    report.expect(
        "...about a minute after the app is up", in_time, True, shown=f"{after:.0f}s"
    )
    install.ended[run] = ("FAILED", "left with nothing to resume it")


def ended_runs_stay_ended(
    data: Path, install: Install, owner: Browser, report: Report
) -> None:
    """Runs that earlier starts ended, or were stopped with: not resumed."""
    for run, (expected, why) in install.ended.items():
        report.expect(
            f"a run {why} is still {expected}", run_status(owner, run), expected
        )
    if not install.ended:
        return
    with contextlib.closing(database(data)) as connection:
        cursor = connection.cursor()
        cursor.execute(
            'SELECT count(*) FROM platform."AgentNodeExecution" WHERE "executionStatus" = \'RUNNING\' '
            'AND "agentGraphExecutionId" = ANY(%s)',
            ([run.rpartition("/")[2] for run in install.ended],),
        )
        report.expect(
            "steps of those runs still marked running", cursor.fetchone()[0], 0
        )
    ports = json.loads((data / "config" / "ports.json").read_text(encoding="utf-8"))
    locks = [key.decode() for key in valkey(data, ports).scan_iter(match="exec_lock:*")]
    report.expect("locks on runs, with none in progress", locks, [])


def stop_with_a_run_in_a_long_step(
    data: Path,
    install: Install,
    owner: Browser,
    report: Report,
    frozen: bool = False,
    hurried: bool = False,
    unlocked: bool = False,
) -> None:
    """Leave a run in the middle of a long step, and an AutoPilot turn in
    progress, for the stop that ends this start (`stop_the_runs`). `frozen`:
    with their executors unable to hear the request to cancel, or anything
    else, as one stuck in a step that cannot be interrupted is. `hurried`:
    see Stopping. `unlocked`: see `lose_its_lock`.

    The turn is short (see the top of this file). One that is to be
    cancelled is started by the stop itself; one that is to be found
    unfinished is frozen as soon as its executor has it."""
    what = (
        "a run that cannot be cancelled"
        if frozen
        else "a run that is about to be stopped"
    )
    run = in_a_long_step(owner, report, install, what)
    if not run:
        return
    if unlocked:
        lose_its_lock(data, install, run, report)
    name, turn_host = executor_host(install), executor_host(install, "copilot-executor")
    turn = None
    if frozen:
        install.frozen = {name, turn_host}

        def freeze() -> None:
            for host in install.frozen:
                hosts(install)[host].suspend()

        turn = turn_in_progress(data, install, owner, report, freeze)
    install.stopping = Stopping(
        run, name, turn, turn_host, owner, cancels=not frozen, hurried=hurried
    )


def lose_its_lock(data: Path, install: Install, run: str, report: Report) -> None:
    """What a sleep of more than five minutes does to a run: its lock runs
    out, upstream never takes it again, and the run goes on. The runtime
    must still count it (it asks the database), and a stop of the runs must
    still end it."""
    from autogpt_desktop import runs

    ports = json.loads((data / "config" / "ports.json").read_text(encoding="utf-8"))
    removed = valkey(data, ports).delete(
        runs.lock_key("executor", run.rpartition("/")[2])
    )
    report.expect("the run's lock is taken away, as a long sleep does", removed, 1)
    time.sleep(
        3 * ACTIVITY_POLL_SECONDS
    )  # the runtime has looked again by now, more than once
    assert install.said
    report.expect(
        "...and the run is still counted as in progress",
        install.said.activity(),
        (1, 0),
    )


def restart_host(
    install: Install, name: str, health: list[str], report: Report
) -> None:
    """Kill a service host; the supervisor must bring it back, serving."""
    victim = hosts(install)[name]
    killed = time.monotonic()
    victim.kill()
    host_is_back(install, name, victim, killed, health, report)


def host_is_back(
    install: Install,
    name: str,
    victim: psutil.Process,
    killed: float,
    health: list[str],
    report: Report,
) -> None:
    def back() -> bool:
        replacement = hosts(install).get(name)
        return bool(replacement and replacement.pid != victim.pid)

    def serving() -> bool:
        return all(fetch_status(address) == 200 for address in health)

    restarted = wait_for(back, RESTART_SECONDS) and wait_for(serving, RESTART_SECONDS)
    took = f"{time.monotonic() - killed:.0f}s"
    report.expect(
        f"{name}, killed, is restarted and serving", restarted, True, shown=took
    )


def skills_catalog(data: Path, install: Install, report: Report) -> None:
    """One more backend interpreter, 700 MB while it runs: not during the
    start, and not at all once this bundle's catalog is in."""
    assert install.runtime
    log = data / "logs" / "skills-catalog.log"
    marker = data / "config" / "skills-catalog.published"
    report.expect(
        "the skills catalog publish starts", wait_for(log.exists, CATALOG_SECONDS), True
    )
    for child in install.runtime.children():
        with contextlib.suppress(psutil.Error):
            if "publish_skills_catalog" in " ".join(child.cmdline()):
                after = child.create_time() - install.ready_wall
                on_time = after >= CATALOG_NOT_BEFORE_SECONDS
                report.expect(
                    "...once the app has been up a while",
                    on_time,
                    True,
                    shown=f"{after:.0f}s",
                )
    published = wait_for(marker.exists, CATALOG_SECONDS)
    # It downloads the catalog from GitHub; a machine without a network has
    # no catalog and loses nothing else.
    report.note("the skills catalog was published", published)


def leave_runs_running(
    data: Path, install: Install, first: str | None, second: str | None
) -> None:
    """What a crash leaves: runs marked running that nothing is working on.
    The next start fails the one that is days old and ends the one the user
    stopped; it must not resume either."""
    if not (first and second):
        return
    with contextlib.closing(database(data)) as connection:
        cursor = connection.cursor()
        cursor.execute(
            'UPDATE platform."AgentGraphExecution" SET "executionStatus" = \'RUNNING\', '
            '"createdAt" = "createdAt" - interval \'2 days\' WHERE id = %s',
            (first,),
        )
        cursor.execute(
            'UPDATE platform."AgentNodeExecution" SET "executionStatus" = \'RUNNING\' '
            'WHERE "agentGraphExecutionId" = %s',
            (first,),
        )
        cursor.execute(
            'UPDATE platform."AgentGraphExecution" SET "executionStatus" = \'RUNNING\' WHERE id = %s',
            (second,),
        )
    (data / "run" / "stopped-runs.json").write_text(
        json.dumps([second]), encoding="utf-8"
    )
    install.left_running = {first: "FAILED", second: "TERMINATED"}


def settled_runs(data: Path, install: Install, report: Report, owner: Browser) -> None:
    """What the restart made of the runs the first start left behind."""
    if install.in_flight:
        what = "the run in flight when the app quit, picked up after the restart,"
        finish_run(
            owner,
            report,
            what,
            install.in_flight,
            "output_message",
            "the timer ran out",
        )
    ports = json.loads((data / "config" / "ports.json").read_text(encoding="utf-8"))
    with contextlib.closing(database(data)) as connection:
        cursor = connection.cursor()
        for run, expected in install.left_running.items():
            cursor.execute(
                'SELECT "executionStatus"::text FROM platform."AgentGraphExecution" WHERE id = %s',
                (run,),
            )
            report.expect(
                f"a run left running is {expected} after a restart",
                cursor.fetchone()[0],
                expected,
            )
        cursor.execute(
            'SELECT count(*) FROM platform."AgentNodeExecution" WHERE "executionStatus" = \'RUNNING\''
        )
        report.expect("node executions still marked running", cursor.fetchone()[0], 0)
    report.expect(
        "the record of stopped runs is used up",
        (data / "run" / "stopped-runs.json").exists(),
        False,
    )
    report.expect(
        "a lock from before the restart", valkey(data, ports).exists(STALE_LOCK), 0
    )


def nobodys_children(data: Path, runtime: psutil.Process) -> list[psutil.Process]:
    """Part of the stack, but not below the runtime in the process tree:
    PostgreSQL on Windows, which pg_ctl starts and then leaves
    (autogpt_desktop/postgres.py). Found the way the runtime finds it."""
    try:
        pid = int((data / "postgres" / "postmaster.pid").read_text().splitlines()[0])
        server = psutil.Process(pid)
        # A file left by an earlier start names a process that is gone, or
        # by now somebody else's.
        ours = born_under(server, runtime) and server.name().lower().startswith(
            "postgres"
        )
        return [server] if ours and runtime not in server.parents() else []
    except (OSError, ValueError, IndexError, psutil.Error):
        return []


def born_under(process: psutil.Process, runtime: psutil.Process) -> bool:
    return process.create_time() >= runtime.create_time()


def memory_table(runtime: psutil.Process, detached: list[psutil.Process]) -> float:
    """Print what each supervised process and its children hold; return the
    total in MB, without the skills-catalog publisher (a one-shot that is not
    part of the idle app). USS is memory unique to a process, the honest number on
    Windows and Linux; on macOS it under-reports, and the physical footprint
    (what Activity Monitor shows) is read from `top` instead. "Committed" is
    what the process has asked for, which paging cannot hide."""
    footprints = macos_footprints()
    megabyte = 1024 * 1024
    rows: dict[str, list[float]] = {}
    for child in [runtime, *runtime.children(), *detached]:
        members = (
            [child] if child == runtime else [child, *child.children(recursive=True)]
        )
        row = rows.setdefault(label(child, runtime), [0, 0, 0, 0])
        for member in members:
            rss, own, committed = usage(member, footprints)
            row[0] += 1
            row[1] += rss / megabyte
            row[2] += own / megabyte
            row[3] += committed / megabyte
    print(
        f"  {'process':18} {'count':>5} {'RSS MB':>8} {'own MB':>8} {'committed MB':>13}"
    )
    total = [0.0, 0.0, 0.0, 0.0]
    for name, row in sorted(rows.items(), key=lambda item: -item[1][2]):
        counted = name != CATALOG
        note = "" if counted else "  (running now; not in the total)"
        print(
            f"  {name:18} {row[0]:5.0f} {row[1]:8.0f} {row[2]:8.0f} {row[3]:13.0f}{note}"
        )
        if counted:
            total = [sum(pair) for pair in zip(total, row, strict=True)]
    print(
        f"  {'TOTAL':18} {total[0]:5.0f} {total[1]:8.0f} {total[2]:8.0f} {total[3]:13.0f}"
    )
    machine = psutil.virtual_memory()
    print(
        f"  machine: {machine.total / megabyte / 1024:.0f} GB, "
        f"{machine.available / megabyte / 1024:.1f} GB available, {psutil.cpu_count()} cores"
    )
    return total[2]


def label(process: psutil.Process, runtime: psutil.Process) -> str:
    if process == runtime:
        return "(runtime)"
    try:
        command = process.cmdline()
        name = process.name().lower()
    except psutil.Error:
        return "?"
    if "--name" in command:
        return command[command.index("--name") + 1]
    if "publish_skills_catalog" in " ".join(command):
        return CATALOG
    if any(part.endswith("server.js") for part in command):
        return "frontend"
    for marker in ("postgres", "valkey", "epmd", "erl", "beam", "cmd", "sh"):
        if name.startswith(marker):
            return "rabbitmq" if marker in ("erl", "beam", "cmd", "sh") else marker
    return name


def usage(process: psutil.Process, footprints: dict[int, int]) -> tuple[int, int, int]:
    """(RSS, memory of its own, committed) in bytes."""
    try:
        info = process.memory_info()
        committed = getattr(info, "private", getattr(info, "vms", 0))
        if process.pid in footprints:
            return info.rss, footprints[process.pid], committed
        return info.rss, getattr(process.memory_full_info(), "uss", info.rss), committed
    except psutil.Error:
        return 0, 0, 0


def macos_footprints() -> dict[int, int]:
    if sys.platform != "darwin":
        return {}
    output = subprocess.run(
        ["top", "-l", "1", "-stats", "pid,mem"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    units = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3}
    footprints = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            value = parts[1].rstrip("+-")
            if value and value[-1] in units:
                footprints[int(parts[0])] = int(float(value[:-1]) * units[value[-1]])
    return footprints


def wait_for(done: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if done():
            return True
        time.sleep(1)
    return done()


def restart(url: str, data: Path, install: Install) -> list[str]:
    report = Report()
    report.expect("the app's address after a restart", url, install.address)
    owner = install.owner = Browser(url)
    report.expect(
        "the owner signs in", owner.sign_in(OWNER_EMAIL, OWNER_PASSWORD)[0], 200
    )
    report.expect("the owner's role after a restart", owner.role(), "admin")
    report.expect("an admin route with the owner's token", owner.admin_route(), 200)
    orphan_is_failed(install, owner, report)
    settled_runs(data, install, report, owner)

    status, body = Browser(url).sign_up(
        "second@smoke.test", "password-of-a-second-person"
    )
    message = str(body.get("message"))
    report.expect("a sign-up is refused by the frontend's own gate", status, 403)
    # What the sign-up page recognises (frontend/src/app/api/auth/utils.ts).
    report.expect(
        "...with its message", "not allowed" in message.lower(), True, shown=message
    )
    stop_with_a_run_in_a_long_step(data, install, owner, report)
    with contextlib.closing(database(data)) as connection:
        report.expect(
            "accounts after the refused sign-up", count_accounts(connection), 1
        )
        # For the next start: an install from before the owner step has an
        # account and no admin, and the owner has forgotten the password.
        connection.cursor().execute(
            'UPDATE platform."UserAuthIdentity" SET role = NULL'
        )
    write_password_reset(data, NEW_OWNER_PASSWORD)
    return report.failures


def restart_with_a_password_reset(url: str, data: Path, install: Install) -> list[str]:
    report = Report()
    report.expect("the app's address after a second restart", url, install.address)
    report.expect(
        "the password file is gone",
        (data / "config" / "reset-password").exists(),
        False,
    )
    if install.owner:
        # The reset deletes the owner's sessions. A browser that holds one
        # keeps working from its cookie until the frontend's five-minute
        # cache of it runs out (cookieCache in auth.ts); the README says so.
        stale = install.owner
        report.note(
            "a session from before the reset, from its cookie alone", stale.role()
        )
        report.expect(
            "...is gone when looked up in the database", stale.role(cached=False), None
        )
    stale = Browser(url).sign_in(OWNER_EMAIL, OWNER_PASSWORD)[0]
    report.expect("the old password is refused", stale, 401)
    owner = Browser(url)
    report.expect(
        "the new password works", owner.sign_in(OWNER_EMAIL, NEW_OWNER_PASSWORD)[0], 200
    )
    report.expect("the oldest account was made admin again", owner.role(), "admin")
    report.expect("an admin route with the owner's token", owner.admin_route(), 200)
    ended_runs_stay_ended(data, install, owner, report)
    stopped_turns_are_not_run_again(data, install, report)
    # A minute and more after the start: a run that was going to resume would have.
    time.sleep(max(0.0, install.ready_at + RESUMED_BY_SECONDS - time.monotonic()))
    ended_runs_stay_ended(data, install, owner, report)
    stop_with_a_run_in_a_long_step(data, install, owner, report, frozen=True)
    return report.failures


def restart_after_runs_were_stopped(
    url: str, data: Path, install: Install
) -> list[str]:
    """After a stop that ended a run its executor could not be told to
    cancel: the run's message was still on the queue."""
    report = Report()
    owner = Browser(url)
    report.expect(
        "the owner signs in", owner.sign_in(OWNER_EMAIL, NEW_OWNER_PASSWORD)[0], 200
    )
    ended_runs_stay_ended(data, install, owner, report)
    stopped = [
        run.rpartition("/")[2]
        for run, (status, _) in install.ended.items()
        if status == "TERMINATED"
    ]
    log = data / "logs" / f"{executor_host(install)}.log"

    def skipped() -> bool:
        said = log.read_text(encoding="utf-8", errors="replace")
        return any(f"Skipped graph execution {run}" in said for run in stopped)

    report.expect(
        "the executor skipped the message of the run that was ended",
        wait_for(skipped, 60),
        True,
    )
    stopped_turns_are_not_run_again(data, install, report)
    ended_runs_stay_ended(data, install, owner, report)
    run_agent(owner, report, "an agent run after all that", 3, 14)
    counted(install, report, "nothing is counted as in progress", (0, 0))
    stop_with_a_run_in_a_long_step(
        data, install, owner, report, frozen=True, hurried=True, unlocked=True
    )
    return report.failures


class Report:
    def __init__(self) -> None:
        self.failures: list[str] = []

    def expect(
        self, what: str, actual: object, expected: object, shown: object = None
    ) -> None:
        print(f"  {what} -> {actual if shown is None else shown}")
        if actual != expected:
            self.failures.append(f"{what}: got {actual!r}, expected {expected!r}")

    def note(self, what: str, observed: object) -> None:
        """Printed for the record; not a pass or a fail."""
        print(f"  {what} -> {observed} (not checked)")


class Browser:
    """What a signed-in window is to the app: a cookie jar, and requests that
    name the app as their origin (Better Auth refuses cookie-bearing POSTs
    from anywhere else)."""

    def __init__(self, url: str, origin: str | None = None) -> None:
        self.url = url
        self.origin = origin or url
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )

    def sign_up(self, email: str, password: str) -> tuple[int, dict]:
        account = {"name": "Smoke Test", "email": email, "password": password}
        return self.post("/api/auth/sign-up/email", account)

    def sign_in(self, email: str, password: str) -> tuple[int, dict]:
        return self.post(
            "/api/auth/sign-in/email", {"email": email, "password": password}
        )

    def role(self, cached: bool = True) -> object:
        query = "" if cached else "?disableCookieCache=true"
        _, session = self.get(f"/api/auth/get-session{query}")
        user = session.get("user")
        return user.get("role") if isinstance(user, dict) else None

    def admin_route(self) -> int:
        """The backend only believes the token the frontend mints."""
        _, minted = self.get("/api/auth/token")
        return self.get(ADMIN_ROUTE, token=str(minted.get("token")))[0]

    def api(self, path: str, body: dict | None = None) -> tuple[int, dict]:
        """The platform's REST API, as the signed-in window calls it."""
        _, minted = self.get("/api/auth/token")
        return self.send(f"/_agpt/api{path}", body, str(minted.get("token")))

    def get(self, path: str, token: str | None = None) -> tuple[int, dict]:
        return self.send(path, None, token)

    def post(self, path: str, body: dict) -> tuple[int, dict]:
        return self.send(path, body, None)

    def send(self, path: str, body: dict | None, token: str | None) -> tuple[int, dict]:
        headers = {"Origin": self.origin, "Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=payload, headers=headers)
        try:
            with self.opener.open(request, timeout=60) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        try:
            answer = json.loads(raw)
        except ValueError:
            answer = None
        return status, answer if isinstance(answer, dict) else {}


def secrets(data: Path) -> dict[str, str]:
    """What the runtime generated for this data directory."""
    lines = (data / "config" / "runtime.env").read_text(encoding="utf-8").splitlines()
    return dict(
        line.split("=", 1) for line in lines if "=" in line and not line.startswith("#")
    )


def database(data: Path):
    """A connection as postgres, the way the runtime's own bootstrap makes
    one: from the port and the generated password in the data directory."""
    connection = psycopg2.connect(
        host="127.0.0.1",
        port=json.loads((data / "config" / "ports.json").read_text(encoding="utf-8"))[
            "postgres"
        ],
        user="postgres",
        password=secrets(data)["POSTGRES_PASSWORD"],
        dbname="postgres",
        connect_timeout=10,
    )
    connection.autocommit = True
    return connection


def valkey(data: Path, ports: dict[str, int]):
    import redis

    return redis.Redis(
        host="127.0.0.1",
        port=ports["valkey"],
        password=secrets(data)["REDIS_PASSWORD"],
        socket_timeout=5,
        socket_connect_timeout=3,
    )


def setting(connection, name: str) -> str:
    cursor = connection.cursor()
    cursor.execute("SELECT current_setting(%s)", (name,))
    return cursor.fetchone()[0]


def connections(connection) -> int:
    cursor = connection.cursor()
    cursor.execute(
        "SELECT count(*) FROM pg_stat_activity WHERE backend_type = 'client backend'"
    )
    return cursor.fetchone()[0]


def count_accounts(connection) -> int:
    cursor = connection.cursor()
    cursor.execute('SELECT count(*) FROM platform."UserAuthIdentity"')
    return cursor.fetchone()[0]


def insert_as_postgres(connection) -> object:
    """Upstream migrations insert accounts as postgres. The owner trigger
    must let them through, as ordinary users; one that raised would leave the
    migration unfinished and the app unable to start. Rolled back."""
    cursor = connection.cursor()
    cursor.execute("BEGIN")
    try:
        cursor.execute(
            'INSERT INTO platform."UserAuthIdentity" '
            '(id, name, email, "emailVerified", "createdAt", "updatedAt") '
            "VALUES ('smoke-migrated', 'Migrated', 'migrated@smoke.test', true, now(), now()) "
            "RETURNING role"
        )
        return cursor.fetchone()[0]
    except psycopg2.Error as exc:
        return f"refused: {exc}"
    finally:
        cursor.execute("ROLLBACK")


def write_password_reset(data: Path, password: str) -> None:
    """What the shell's "Reset owner password" does (src/owner.js)."""
    path = data / "config" / "reset-password"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(password + "\n")


def probe(url: str) -> list[str]:
    checks = {
        "/healthz": 200,
        "/_agpt/health": 200,
        "/_agpt/docs": 404,
        "/login": 200,
        "/_agpt/api/store/agents?page_size=1": 200,
    }
    failures = []
    for path, expected in checks.items():
        status = fetch_status(url + path)
        print(f"  {path} -> {status}")
        if status != expected:
            failures.append(f"{path} returned {status}, expected {expected}")
    return failures + caches_put_to_work(url)


def caches_put_to_work(url: str) -> list[str]:
    """What the Next server would keep on disk beside its own code, asked
    for: an optimised image (twice: the second answer comes from the cache,
    wherever that is) and a page that fetches from the backend while it
    renders. The check that the bundle is unchanged is what these are for."""
    report = Report()
    for attempt in ("first", "second"):
        status, kind, _ = fetch(url + OPTIMISED_IMAGE, {"Accept": "image/webp"})
        # Anything but an image Next encoded itself would not have been cached.
        report.expect(
            f"an optimised image, {attempt} request",
            (status, kind),
            (200, "image/webp"),
        )
    status, _, body = fetch(url + "/sitemap.xml", {})
    listed = b"/marketplace" in body
    report.expect(
        "the sitemap, which fetches while rendering", (status, listed), (200, True)
    )
    return report.failures


def fetch(url: str, headers: dict[str, str]) -> tuple[int | str, str, bytes]:
    """(status, content type, body); the status is the error's text when
    there was no answer."""
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=60
        ) as response:
            return response.status, response.headers.get_content_type(), response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get_content_type(), exc.read()
    except (urllib.error.URLError, OSError) as exc:
        return str(exc), "", b""


def fetch_status(url: str) -> int | str:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, OSError) as exc:
        return str(exc)


def print_logs(data: Path) -> None:
    for log in sorted((data / "logs").glob("*.log")):
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
        print(f"\n--- {log.name} (last {len(tail)} lines)")
        print("\n".join(tail))


if __name__ == "__main__":
    sys.exit(main())
