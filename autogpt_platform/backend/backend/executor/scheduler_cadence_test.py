"""A repeating follow-up gets a new cadence in place (SECRT-2879).

Against a real APScheduler with an in-memory job store, so the trigger, the
stored kwargs and the next run time are the scheduler's own.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from zoneinfo import ZoneInfo

import pytest
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.executor.scheduler import (
    CopilotTurnJobArgs,
    Jobstores,
    Scheduler,
    execute_copilot_turn,
)
from backend.util.exceptions import NotAuthorizedError, NotFoundError

_CHICAGO = "America/Chicago"
_STORE = Jobstores.EXECUTION.value


@contextmanager
def _scheduler() -> Iterator[Scheduler]:
    aps = BackgroundScheduler()
    aps.add_jobstore(MemoryJobStore(), _STORE)
    aps.start(paused=True)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.scheduler = aps
    try:
        yield scheduler
    finally:
        aps.shutdown(wait=False)


def _add(
    scheduler: Scheduler,
    *,
    routine_id: str | None = None,
    user_timezone: str | None = _CHICAGO,
    cron: str = "* * * * *",
) -> None:
    """Stored as the jobstore holds it, so a cron from before today's checks
    is possible too."""
    args = CopilotTurnJobArgs(
        schedule_id="job-1",
        user_id="user-1",
        message="Check Discord for new messages.",
        cron=cron,
        user_timezone=user_timezone,
        routine_id=routine_id,
    )
    scheduler.scheduler.add_job(
        execute_copilot_turn,
        kwargs=args.model_dump(mode="json"),
        trigger=CronTrigger.from_crontab(cron, timezone=_CHICAGO),
        jobstore=_STORE,
        id="job-1",
    )


def test_a_follow_up_runs_on_its_new_cadence_under_the_same_id():
    with _scheduler() as scheduler:
        _add(scheduler)

        info = scheduler.update_copilot_turn_cron("job-1", "user-1", "0 9 * * mon-fri")

        job = scheduler.scheduler.get_job("job-1", jobstore=_STORE)
        assert info.id == "job-1" and info.cron == "0 9 * * mon-fri"
        assert job.kwargs["cron"] == "0 9 * * mon-fri"
        next_run = job.next_run_time.astimezone(ZoneInfo(_CHICAGO))
        assert (next_run.hour, next_run.minute) == (9, 0)
        assert next_run.weekday() < 5


def test_a_follow_up_saved_without_its_zone_keeps_the_triggers():
    """Older follow-ups have no user_timezone; their trigger still has the
    zone, and the dialog showed its times in it. 09:00 there, not UTC."""
    with _scheduler() as scheduler:
        _add(scheduler, user_timezone=None)

        scheduler.update_copilot_turn_cron("job-1", "user-1", "0 9 * * *")

        job = scheduler.scheduler.get_job("job-1", jobstore=_STORE)
        next_run = job.next_run_time.astimezone(ZoneInfo(_CHICAGO))
        assert (next_run.hour, next_run.minute) == (9, 0)


def test_a_paused_follow_up_stays_paused():
    with _scheduler() as scheduler:
        _add(scheduler)
        scheduler.scheduler.pause_job("job-1", jobstore=_STORE)

        scheduler.update_copilot_turn_cron("job-1", "user-1", "0 9 * * *")

        job = scheduler.scheduler.get_job("job-1", jobstore=_STORE)
        assert job.next_run_time is None
        assert job.kwargs["cron"] == "0 9 * * *"


def test_a_routines_job_is_left_to_its_routine():
    with _scheduler() as scheduler:
        _add(scheduler, routine_id="routine-1")

        with pytest.raises(NotFoundError):
            scheduler.update_copilot_turn_cron("job-1", "user-1", "0 9 * * *")


def test_a_cadence_that_never_fires_is_refused_and_the_old_one_kept():
    with _scheduler() as scheduler:
        _add(scheduler)

        with pytest.raises(ValueError, match="never fires"):
            scheduler.update_copilot_turn_cron("job-1", "user-1", "0 0 30 2 *")

        job = scheduler.scheduler.get_job("job-1", jobstore=_STORE)
        assert job.kwargs["cron"] == "* * * * *"


def test_resuming_a_stored_schedule_that_never_fires_is_refused():
    with _scheduler() as scheduler:
        _add(scheduler, cron="0 0 30 2 *")

        with pytest.raises(ValueError, match="never fires"):
            scheduler.resume_execution_schedule("job-1", "user-1")

        job = scheduler.scheduler.get_job("job-1", jobstore=_STORE)
        assert job.next_run_time is None


def test_a_paused_schedule_that_fires_resumes():
    with _scheduler() as scheduler:
        _add(scheduler)
        scheduler.scheduler.pause_job("job-1", jobstore=_STORE)

        assert scheduler.resume_execution_schedule("job-1", "user-1") is True
        job = scheduler.scheduler.get_job("job-1", jobstore=_STORE)
        assert job.next_run_time is not None


def test_nobody_else_can_change_it():
    with _scheduler() as scheduler:
        _add(scheduler)

        with pytest.raises(NotAuthorizedError):
            scheduler.update_copilot_turn_cron("job-1", "someone-else", "0 9 * * *")
