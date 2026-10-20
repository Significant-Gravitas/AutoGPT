"""Do-not-merge timing probe for the turn-setup gather's legs (draft #15739).

The SDK service sets LEG_MARKS to a dict before its setup gather; every leg
and the memory recall's embedder / reranker / FalkorDB calls record into it,
and the service publishes it as one ``perf-legs:`` status after the gather.
Outside that window every call is a no-op."""

import contextvars
import time
from collections.abc import Awaitable
from typing import TypeVar

T = TypeVar("T")

LEG_MARKS: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "perf_leg_marks", default=None
)


def mark(name: str) -> None:
    marks = LEG_MARKS.get()
    if marks is not None:
        marks.setdefault(name, time.time())


async def timed(name: str, awaitable: Awaitable[T]) -> T:
    try:
        return await awaitable
    finally:
        mark(name)


def span_start(name: str) -> float:
    now = time.time()
    marks = LEG_MARKS.get()
    if marks is not None:
        marks.setdefault(f"{name}_start", now)
        marks[f"{name}_n"] = marks.get(f"{name}_n", 0) + 1
    return now


def span_end(name: str, started: float) -> None:
    marks = LEG_MARKS.get()
    if marks is None:
        return
    now = time.time()
    marks[f"{name}_end"] = max(marks.get(f"{name}_end", 0.0), now)
    marks[f"{name}_sum"] = marks.get(f"{name}_sum", 0.0) + (now - started)


def note(name: str, value: int | str) -> None:
    marks = LEG_MARKS.get()
    if marks is not None:
        marks[f"{name}_v"] = value


def note_in_set(name: str, value: str) -> None:
    marks = LEG_MARKS.get()
    if marks is not None:
        marks.setdefault(f"{name}_set", set()).add(value)


def call_span(name: str, started: float) -> None:
    """One call's start and end; rendered as start+duration in start order."""
    marks = LEG_MARKS.get()
    if marks is not None:
        marks.setdefault(f"{name}_calls", []).append((started, time.time()))


def render(marks: dict, since: float) -> str:
    parts = []
    for key, value in marks.items():
        if key.endswith("_n"):
            parts.append(f"{key}={int(value)}")
        elif key.endswith("_sum"):
            parts.append(f"{key}={int(value * 1000)}")
        elif key.endswith("_v"):
            parts.append(f"{key[:-2]}={value}")
        elif key.endswith("_set"):
            parts.append(f"{key[:-4]}={'/'.join(sorted(value))}")
        elif key.endswith("_calls"):
            spans = "|".join(
                f"{int((start - since) * 1000)}+{int((end - start) * 1000)}"
                for start, end in sorted(value)
            )
            parts.append(f"{key[:-6]}={spans}")
        else:
            parts.append(f"{key}={int((value - since) * 1000)}")
    return "perf-legs:" + ",".join(parts)


def wrap_async_method(obj, attr: str, span: str) -> None:
    original = getattr(obj, attr, None)
    if original is None:
        return

    async def wrapper(*args, **kwargs):
        started = span_start(span)
        try:
            return await original(*args, **kwargs)
        finally:
            span_end(span, started)

    setattr(obj, attr, wrapper)
