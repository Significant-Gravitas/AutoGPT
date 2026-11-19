"""A repeating task's rhythm: reading the crons asked for, and where the task
goes when a run ends (back to waiting, paused after failures in a row)."""

import json
from datetime import UTC, datetime, timedelta

import prisma.models
import pytest
from prisma.enums import ResourceVisibility, TaskCreator, TaskStatus

from backend.data.task_rhythm import (
    AUTO_PAUSED_TEXT,
    describe_rhythm,
    plan_rhythm,
    rearmed,
)
from backend.data.task_schedule_meta import TaskRepeat, read_schedule_meta

_NOW = datetime(2026, 11, 17, 7, 59, 30, tzinfo=UTC)
_EVERY_3_MIN = TaskRepeat(crons=["*/3 * * * *"], label="Every 3 minutes")
_DAILY_8 = TaskRepeat(crons=["0 8 * * *"], label="Every day at 08:00")


def _row(repeat: TaskRepeat | None, **overrides) -> prisma.models.Task:
    schedule = {"repeat": repeat.model_dump(mode="json")} if repeat else {}
    fields = {
        "id": "task-1",
        "createdAt": _NOW,
        "updatedAt": _NOW,
        "userId": "user-1",
        "title": "Research AutoGPT",
        "status": TaskStatus.IN_PROGRESS,
        "createdBy": TaskCreator.OTTO,
        "isArchived": False,
        "metadata": json.dumps({"schedule": schedule, "other": 1}),
        "visibility": ResourceVisibility.PRIVATE,
        "attempt": 1,
        "isSuggested": False,
    }
    return prisma.models.Task(**(fields | overrides))


# ---------------------------------------------------------------- reading


def test_a_rhythm_first_runs_at_its_next_fire_time():
    plan = plan_rhythm(["0 8 * * *"], "UTC", _NOW, seed="s")

    assert plan.first_at == datetime(2026, 11, 17, 8, 0, tzinfo=UTC)
    assert plan.repeat.label == "Every day at 08:00"
    assert plan.repeat.timezone == "UTC"
    assert not plan.is_costly


def test_a_rhythm_tighter_than_15_minutes_is_costly():
    plan = plan_rhythm(["*/3 * * * *"], "UTC", _NOW, seed="s")

    assert plan.first_at == datetime(2026, 11, 17, 8, 0, tzinfo=UTC)
    assert plan.repeat.label == "Every 3 minutes"
    assert plan.is_costly


def test_an_hour_given_without_a_minute_keeps_one_picked_minute():
    first = plan_rhythm(["H 8 * * *"], "UTC", _NOW, seed="user:title")
    again = plan_rhythm(["H 8 * * *"], "UTC", _NOW, seed="user:title")

    assert first.repeat.crons == again.repeat.crons
    assert first.repeat.crons[0].split()[1:] == ["8", "*", "*", "*"]
    assert first.repeat.crons[0].split()[0].isdigit()


@pytest.mark.parametrize(
    "crons",
    [[], ["  "], ["* * * * * *"], ["not a cron"], ["0 8 31 2 *"]],
)
def test_a_malformed_rhythm_is_refused_in_words_for_the_model(crons):
    with pytest.raises(ValueError):
        plan_rhythm(crons, "UTC", _NOW, seed="s")


def test_weekdays_read_as_weekdays():
    assert describe_rhythm(["0 8 * * 1-5"], "UTC", _NOW) == "Weekdays at 08:00"


# ---------------------------------------------------------------- re-arming


def test_a_finished_run_puts_the_task_back_for_its_next_future_run():
    later = _NOW + timedelta(minutes=10)

    rearm = rearmed(_row(_EVERY_3_MIN), "done", later)

    assert rearm is not None
    # 08:09:30 -> the 08:12 run; the 08:00, 08:03, 08:06 and 08:09 ticks that
    # passed while the run went are skipped, never run twice at once.
    assert rearm.start_at == datetime(2026, 11, 17, 8, 12, tzinfo=UTC)
    assert rearm.meta.due_at == rearm.start_at
    assert rearm.meta.repeat is not None
    assert rearm.meta.repeat.last_run == "done"
    assert rearm.meta.repeat.failed_in_a_row == 0
    assert rearm.paused_because is None


def test_a_task_that_does_not_repeat_settles_as_before():
    assert rearmed(_row(None), "done", _NOW) is None


def test_failures_in_a_row_are_counted_and_a_success_forgets_them():
    failing = _DAILY_8.model_copy(update={"failed_in_a_row": 1})

    failed = rearmed(_row(failing), "failed", _NOW)
    stopped = rearmed(_row(failing), "cancelled", _NOW)
    done = rearmed(_row(failing), "done", _NOW)

    assert failed and failed.meta.repeat and failed.meta.repeat.failed_in_a_row == 2
    assert stopped and stopped.meta.repeat
    assert stopped.meta.repeat.failed_in_a_row == 1
    assert done and done.meta.repeat and done.meta.repeat.failed_in_a_row == 0


def test_the_third_failed_run_in_a_row_pauses_the_task():
    failing = _DAILY_8.model_copy(update={"failed_in_a_row": 2})

    rearm = rearmed(_row(failing), "failed", _NOW)

    assert rearm is not None
    assert rearm.start_at is None
    assert rearm.meta.repeat and rearm.meta.repeat.paused
    assert (
        rearm.paused_because == AUTO_PAUSED_TEXT == ("Paused: the last 3 runs failed")
    )


def test_a_task_paused_during_its_run_waits_paused_after_it():
    paused = _DAILY_8.model_copy(update={"paused": True})

    rearm = rearmed(_row(paused), "done", _NOW)

    assert rearm is not None and rearm.start_at is None
    assert rearm.meta.repeat and rearm.meta.repeat.paused
    assert rearm.paused_because is None


def test_the_rhythm_reads_back_from_the_task_metadata():
    meta = read_schedule_meta(_row(_DAILY_8).metadata)

    assert meta.repeat == _DAILY_8
