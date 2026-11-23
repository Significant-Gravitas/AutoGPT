"""A repeating task's rhythm: read the crons asked for, say them in words, and
work out where a task goes when one of its runs ends.

The crons are the routines' dialect (5 fields, read in the user's time zone),
so ``scheduled_chats.cadence`` counts and describes them. A run never overlaps
the next: while one is going the task is not waiting, so the sweep never
picks it, and when it ends the task waits for the next fire time still ahead,
skipping any that passed meanwhile.
"""

import hashlib
from datetime import datetime, timedelta

import prisma.models
from prisma.enums import TaskStatus
from pydantic import BaseModel

from backend.copilot.scheduled_chats.cadence import describe, next_fire, shortest_gap
from backend.data.task_schedule_meta import (
    RunEnd,
    TaskRepeat,
    TaskScheduleMeta,
    read_schedule_meta,
)

MAX_REPEAT_CRONS = 24
# Tighter than this, each run's cost adds up fast: Otto says so.
COSTLY_GAP = timedelta(minutes=15)
# This many failed runs in a row pause the task.
AUTO_PAUSE_AFTER = 3
AUTO_PAUSED_TEXT = f"Paused: the last {AUTO_PAUSE_AFTER} runs failed"
_SPREAD_MINUTE = "H"
_WEEKDAYS = ("1-5", "mon-fri", "MON-FRI")

RUN_END_BY_STATUS: dict[TaskStatus, RunEnd] = {
    TaskStatus.DONE: "done",
    TaskStatus.FAILED: "failed",
    TaskStatus.CANCELLED: "cancelled",
}
STATUS_BY_RUN_END: dict[RunEnd, TaskStatus] = {
    end: status for status, end in RUN_END_BY_STATUS.items()
}


class RhythmPlan(BaseModel):
    """A rhythm read from what was asked, and when it first runs."""

    repeat: TaskRepeat
    first_at: datetime
    is_costly: bool


def plan_rhythm(
    crons: list[str], timezone: str, now: datetime, seed: str
) -> RhythmPlan:
    """Read the crons the model asked for. Raises ValueError, written for the
    model, when they are malformed. An ``H`` minute is picked from ``seed``
    and kept, as routines pick it."""
    cleaned = [_spread(" ".join(cron.split()), seed) for cron in crons if cron.strip()]
    if not cleaned:
        raise ValueError("repeat_crons needs at least one cron.")
    if len(cleaned) > MAX_REPEAT_CRONS:
        raise ValueError(f"repeat_crons takes at most {MAX_REPEAT_CRONS} crons.")
    bad = [cron for cron in cleaned if len(cron.split()) != 5]
    if bad:
        raise ValueError(
            f"Each cron has 5 fields (minute hour day month weekday): {bad[0]!r}. "
            "The shortest rhythm is every minute."
        )
    first_at = next_fire(cleaned, timezone, now)
    if first_at is None:
        raise ValueError("Those crons do not parse, or never run within a year.")
    gap = shortest_gap(cleaned, timezone, now)
    return RhythmPlan(
        repeat=TaskRepeat(
            crons=cleaned,
            timezone=timezone,
            label=describe_rhythm(cleaned, timezone, now),
        ),
        first_at=first_at,
        is_costly=gap is not None and gap < COSTLY_GAP,
    )


def describe_rhythm(crons: list[str], timezone: str, now: datetime) -> str:
    """How often it runs, in words: "Every 3 minutes", "Every day at 08:00",
    "Weekdays at 08:00"."""
    words = describe(crons, timezone, now)
    weekdays = _weekdays_at(crons)
    if weekdays and (words is None or words.endswith("times a week")):
        return weekdays
    return words or "On a schedule"


def next_run_at(repeat: TaskRepeat, now: datetime) -> datetime | None:
    """The next fire time still ahead of ``now``; none for a task that
    watches, whose events set its start (``next_watch_run_at``)."""
    if repeat.watch is not None:
        return None
    return next_fire(repeat.crons, repeat.timezone, now)


def next_watch_run_at(meta: TaskScheduleMeta, now: datetime) -> datetime | None:
    """A watching task starts again at once while events wait, else waits."""
    return now if meta.events else None


def is_repeating(row: prisma.models.Task) -> bool:
    return read_schedule_meta(row.metadata).repeat is not None


class Rearm(BaseModel):
    """A repeating task back to waiting once a run ended."""

    start_at: datetime | None
    meta: TaskScheduleMeta
    # Set when this run paused it; the line its activity gets.
    paused_because: str | None = None


def rearmed(row: prisma.models.Task, ended: RunEnd, now: datetime) -> Rearm | None:
    """Where a repeating task goes once a run ended: back to waiting (TODO),
    for the next fire time ahead (a watching task: at once while events wait,
    else until the next event), with the run counted. A run that failed for
    the ``AUTO_PAUSE_AFTER``th time in a row pauses it. None for a task that
    does not repeat, or whose rhythm never runs again: it settles as a one-off
    does."""
    meta = read_schedule_meta(row.metadata)
    repeat = meta.repeat
    if repeat is None:
        return None
    failed = {
        "failed": repeat.failed_in_a_row + 1,
        "done": 0,
        "cancelled": repeat.failed_in_a_row,
    }[ended]
    auto_paused = not repeat.paused and failed >= AUTO_PAUSE_AFTER
    paused = repeat.paused or auto_paused
    watching = repeat.watch is not None
    if paused:
        next_at = None
    elif watching:
        next_at = next_watch_run_at(meta, now)
    else:
        next_at = next_run_at(repeat, now)
    if not paused and not watching and next_at is None:
        return None
    repeat = repeat.model_copy(
        update={"paused": paused, "failed_in_a_row": failed, "last_run": ended}
    )
    return Rearm(
        start_at=next_at,
        meta=meta.model_copy(
            update={"repeat": repeat, "due_at": next_at, "missed": False}
        ),
        paused_because=AUTO_PAUSED_TEXT if auto_paused else None,
    )


def _spread(cron: str, seed: str) -> str:
    fields = cron.split()
    if len(fields) != 5 or fields[0] != _SPREAD_MINUTE:
        return cron
    digest = hashlib.sha256(seed.encode()).digest()
    return " ".join([str(digest[0] % 60), *fields[1:]])


def _weekdays_at(crons: list[str]) -> str | None:
    """ "Weekdays at 08:00" for one cron that runs at one time Monday to
    Friday; None otherwise."""
    if len(crons) != 1:
        return None
    fields = crons[0].split()
    if len(fields) != 5:
        return None
    minute, hour, day, month, weekday = fields
    if not (minute.isdigit() and hour.isdigit()) or (day, month) != ("*", "*"):
        return None
    if weekday not in _WEEKDAYS:
        return None
    return f"Weekdays at {int(hour):02d}:{int(minute):02d}"
