"""Do-not-merge timing probe for the AutoPilot latency work (draft #15739).

A route starts a list of marks for its request; deeper code adds named marks;
the route publishes them as a status. Outside a started request every mark is
a no-op."""

import contextvars
import time

_marks: contextvars.ContextVar[list | None] = contextvars.ContextVar(
    "perf_probe_marks", default=None
)


def start() -> None:
    _marks.set([])


def mark(name: str) -> None:
    marks = _marks.get()
    if marks is not None:
        marks.append((name, time.perf_counter()))


def take() -> list:
    return list(_marks.get() or [])
