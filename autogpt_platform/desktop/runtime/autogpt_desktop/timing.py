"""Where the time of a start goes.

Each phase is reported as it ends, in the log and to the shell, and the start
ends with one line that has them all:

    started in 41 s (database 2.1, cache 0.6, queue 9.8, migrations 4.0,
    services 27.5, frontend 3.2)

Phases overlap (the queue boots while the migrations run), so they do not add
up to the total. A phase is the time between its work being started and its
result being seen: `queue` is from launching RabbitMQ to its first answer.
Each server is asked by a thread that does nothing else meanwhile
(supervisor.py: the cache and the broker on theirs, the frontend on its own),
so a phase is that server's time and not the wait for whatever the start was
busy with before it asked. `services` begins when the hosts are launched and
ends when what the window talks to answers, which is after the go.
build/smoke_test.py prints these for every start and holds a restart to a
budget.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator

from autogpt_desktop import events

# The order they are listed in; a phase that did not happen is left out.
PHASES = ("config", "database", "cache", "queue", "migrations", "services", "frontend")


class Phases:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.began = clock()
        self.seconds: dict[str, float] = {}
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Reported only when the work succeeds: a phase that failed is in
        the error, not in the table."""
        began = self.clock()
        yield
        self.record(name, self.clock() - began)

    def record(self, name: str, seconds: float) -> None:
        with self._lock:
            if name in self.seconds:
                return  # a service that was started again: its first answer counts
            self.seconds[name] = seconds
        events.timing(name, seconds)

    def since_start(self) -> float:
        return self.clock() - self.began

    def finish(self) -> None:
        with self._lock:
            seconds = dict(self.seconds)
        total, listed = self.since_start(), ordered(seconds)
        events.started(summary(total, listed), total, listed)


def ordered(seconds: dict[str, float]) -> dict[str, float]:
    known = [name for name in PHASES if name in seconds]
    return {
        name: seconds[name] for name in [*known, *sorted(seconds.keys() - set(known))]
    }


def summary(total: float, seconds: dict[str, float]) -> str:
    listed = ", ".join(f"{name} {spent:.1f}" for name, spent in seconds.items())
    return f"started in {total:.0f} s ({listed})"
