"""Start the shim again after an exit that asks for it, where the service
manager doesn't.

launchd and systemd start the shim again when it exits asking to be started
again (75, EX_TEMPFAIL: a keychain that isn't up yet, audit storage that is
full) or crashes. launchd relaunches on any non-zero exit (KeepAlive with
SuccessfulExit false), and the systemd unit restarts on failure after five
seconds, except after 77 and 78 (RestartPreventExitStatus).

Task Scheduler doesn't. Its RestartOnFailure applies when the task's action
can't be started; a program that starts and then exits, with whatever code,
is a run that completed. Under the Windows task a shim that exited 75, or
crashed after logon, stayed down until the next logon.

So the task runs `autogpt-shim start --supervise`, which does what the other
two service managers do: it runs the daemon as a child process, and when
the child exits it decides by the exit code, as the systemd unit does.

- 0, 77 and 78 end it: a stop that was asked for (a revoked session and a
  protocol mismatch exit 0 too), a sign-in that was refused, and a setting
  or a missing package that fails the same way every time.
- Any other exit starts the daemon again after a pause: 75, and a crash.

The daemon runs in a process of its own each time, and not again in this
one, because what a restart is for is per process: the keyring package
picks its backend once per process, so only a new one finds a keychain that
came up later. A crash that takes the interpreter down is seen from outside
it, too.

A person who runs `autogpt-shim start` in a terminal gets none of this: the
flag is in the task `install` renders, and nowhere else.
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from typing import Protocol

logger = logging.getLogger(__name__)

# The option that turns the supervisor on, which the Windows task's
# arguments carry (install._task_scheduler_values).
SUPERVISE_OPTION = "--supervise"

# Exit codes after which the daemon isn't started again, as in the systemd
# unit (`Restart=on-failure` with `RestartPreventExitStatus=77 78`).
STOP_CODES = frozenset({0, 77, 78})
# The pause before the first try, systemd's RestartSec. It doubles after
# each run that ended sooner than SETTLED_SECONDS, up to the longest, so a
# daemon that can't stay up isn't started in a tight loop; one that ran
# longer starts the pauses over.
FIRST_PAUSE_SECONDS = 5.0
LONGEST_PAUSE_SECONDS = 60.0
SETTLED_SECONDS = 60.0
# How often a pause between two runs looks whether a stop was asked for.
STOP_POLL_SECONDS = 0.2


class Stop(Protocol):
    """A stop that can be asked for, looked at and waited for:
    threading.Event's part in the loop, which a test passes, and _Stop's."""

    def is_set(self) -> bool: ...

    def set(self) -> None: ...

    def wait(self, timeout: float) -> bool: ...


class _Stop:
    """The supervisor's stop, with no lock of its own.

    The stop's signal handler sets it, and Python runs a handler on the main
    thread, between two of its instructions, wherever that thread is.
    threading.Event.set takes the lock that Event.wait holds around its own
    bookkeeping, so a signal that landed there would wait for a lock its own
    thread held, for good. Setting an attribute takes none, and `wait`
    looks at it every STOP_POLL_SECONDS."""

    def __init__(self) -> None:
        self._asked = False

    def is_set(self) -> bool:
        return self._asked

    def set(self) -> None:
        self._asked = True

    def wait(self, timeout: float) -> bool:
        """Wait until the stop is asked for, `timeout` seconds at most, and
        say whether it was."""
        give_up = time.monotonic() + timeout
        while not self._asked:
            remaining = give_up - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(remaining, STOP_POLL_SECONDS))
        return self._asked


def supervise(
    run: Callable[[], int],
    stop: Stop,
    *,
    wait: Callable[[float], bool] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run the daemon (`run()`, which returns its exit code) until it ends
    with a code that means stop, or `stop` is set. Returns the last exit
    code.

    `wait(seconds)` pauses between runs and says whether a stop was asked
    for meanwhile: `stop.wait` unless a test passes its own."""
    wait = stop.wait if wait is None else wait
    pause = FIRST_PAUSE_SECONDS
    while True:
        started = clock()
        code = run()
        if code in STOP_CODES or stop.is_set():
            return code
        if clock() - started >= SETTLED_SECONDS:
            pause = FIRST_PAUSE_SECONDS
        logger.warning("The local executor exited with %d; starting it again in %g s", code, pause)
        if wait(pause):
            return code
        pause = min(pause * 2, LONGEST_PAUSE_SECONDS)


def daemon_command(arguments: Sequence[str]) -> list[str]:
    """The command that runs the daemon itself for a supervisor started with
    `arguments` (its own, without the program): the same ones without the
    option that asked for the supervisor, run by this interpreter.

    However the option was written. argparse takes any beginning of a long
    option that no other option shares, `--supervis` say, and a daemon
    handed that would take it the same way: it would be a supervisor too,
    start another, and no daemon would ever run. On a command line that
    turned the supervisor on, an argument that is a beginning of the
    option is the option: argparse reads nothing that starts with `--` as
    another option's value."""
    return [
        sys.executable,
        "-m",
        "autogpt_local_executor",
        *(argument for argument in arguments if not _is_the_option(argument)),
    ]


def _is_the_option(argument: str) -> bool:
    """Whether `argument` is SUPERVISE_OPTION, whole or cut short: `--s` and
    longer. `--` alone ends the options, and is no option."""
    return len(argument) > 2 and SUPERVISE_OPTION.startswith(argument)


class _Child:
    """The daemon's process, while one runs, for a stop to end.

    With no lock. The stop's signal handler ends the daemon (end), and
    Python runs a handler on the thread that runs the daemon (run), between
    two of its instructions. A lock that `run` held while it made the
    process was one the handler then waited for on that same thread, for
    good: the supervisor never stopped, and the daemon ran on. Everything
    here is that one thread's, so there is nothing for a lock to keep
    apart."""

    def __init__(self) -> None:
        self._process: subprocess.Popen[bytes] | None = None

    def run(self, command: Sequence[str], stop: Stop) -> int:
        """Run `command` to its exit and return its exit code. No input: a
        daemon under a supervisor has nobody to ask."""
        if stop.is_set():
            return 0
        process = self._process = subprocess.Popen(list(command), stdin=subprocess.DEVNULL)
        try:
            if stop.is_set():
                # Asked for while the process was being made, when there was
                # none yet for the stop to end.
                self.end()
            while True:
                try:
                    return process.wait()
                except KeyboardInterrupt:
                    # Ctrl+C in a terminal reaches the daemon as well, which
                    # stops itself: it is waited for, and not started again.
                    stop.set()
        finally:
            self._process = None

    def end(self) -> None:
        """Ask the running daemon to stop (SIGTERM, which it handles; on
        Windows the process is ended)."""
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()


def _stop_asked(stop: Stop, child: _Child) -> None:
    """What the stop's signal handler does: no daemon is started from here
    on, and the one that runs is told to stop. Nothing in it waits for a
    lock (_Stop, _Child)."""
    stop.set()
    child.end()


def run_supervised(arguments: Sequence[str]) -> int:
    """`autogpt-shim start --supervise`: run the daemon, and again after
    each exit that asks for it, until it or a stop ends the loop. Returns
    the daemon's last exit code."""
    stop = _Stop()
    child = _Child()
    command = daemon_command(arguments)

    def stopped(_signal: int, _frame: object) -> None:
        _stop_asked(stop, child)

    # A service manager's stop, where there is one: the daemon is told, and
    # isn't started again. (Task Scheduler ends the task's whole job
    # instead, the daemon with it.)
    previous = signal.signal(signal.SIGTERM, stopped)
    try:
        return supervise(lambda: child.run(command, stop), stop)
    finally:
        signal.signal(signal.SIGTERM, previous)


__all__ = [
    "FIRST_PAUSE_SECONDS",
    "LONGEST_PAUSE_SECONDS",
    "SETTLED_SECONDS",
    "STOP_CODES",
    "STOP_POLL_SECONDS",
    "SUPERVISE_OPTION",
    "Stop",
    "daemon_command",
    "run_supervised",
    "supervise",
]
