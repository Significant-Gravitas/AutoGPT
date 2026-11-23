"""The archive's mark on a graph schedule (#14577), against a real APScheduler.

Re-hiring an expert resumes the schedules its archive paused and no others, so
the scheduler has to tell those pauses apart from every other kind. The mark
lives in the job's own kwargs. These check who sets it, who clears it, and that
a job stored without it reads as unmarked. Races between the writers are in
``scheduler_write_race_test.py``; the archive and re-hire flows run end to end
in ``api/features/experts/rehire_schedules_test.py``.
"""

import uuid
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pytest
from apscheduler.triggers.base import BaseTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.executor.scheduler import GraphExecutionJobArgs, execute_graph
from backend.executor.scheduler_jobstore_test_helpers import (
    CRON,
    STORE,
    USER,
    add_graph_job,
    graph_args,
    pause,
    resume,
    sqlite_scheduler,
    stored,
)


def test_the_archive_marks_a_running_schedule_it_pauses(tmp_path):
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)

        assert pause(scheduler, for_expert_archive=True) is True

        after = stored(scheduler)
        assert after.next_run_time == "" and after.paused_by_expert_archive


def test_the_archive_never_marks_a_schedule_someone_else_paused(tmp_path):
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        assert pause(scheduler) is True

        assert pause(scheduler, for_expert_archive=True) is False

        after = stored(scheduler)
        assert after.next_run_time == "" and not after.paused_by_expert_archive


def test_a_users_pause_takes_a_schedule_over_from_the_archive(tmp_path):
    """Already paused, so the pause stops nothing, but the schedule is the
    user's to keep paused now: re-hiring leaves it."""
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        pause(scheduler, for_expert_archive=True)

        assert pause(scheduler) is False

        after = stored(scheduler)
        assert after.next_run_time == "" and not after.paused_by_expert_archive
        assert resume(scheduler, only_if_paused_by_expert_archive=True) is False
        assert stored(scheduler).next_run_time == ""


def test_rehire_resumes_a_schedule_the_archive_paused_and_clears_the_mark(
    tmp_path,
):
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        pause(scheduler, for_expert_archive=True)

        assert resume(scheduler, only_if_paused_by_expert_archive=True) is True

        after = stored(scheduler)
        assert after.next_run_time and not after.paused_by_expert_archive


def test_rehire_leaves_a_schedule_the_user_paused(tmp_path):
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        pause(scheduler)

        assert resume(scheduler, only_if_paused_by_expert_archive=True) is False

        assert stored(scheduler).next_run_time == ""


def test_any_resume_clears_the_mark(tmp_path):
    """A user resuming a schedule the archive paused has taken it over too: a
    later pause of theirs must not leave it marked for the next re-hire."""
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        pause(scheduler, for_expert_archive=True)

        assert resume(scheduler) is True

        after = stored(scheduler)
        assert after.next_run_time and not after.paused_by_expert_archive


# Midnight, a month from now: a trigger starting then fires at a time that
# doesn't depend on when the test runs.
_LATER = (datetime.now(timezone.utc) + timedelta(days=30)).replace(
    hour=0, minute=0, second=0, microsecond=0
)
_ENDED = datetime.now(timezone.utc) - timedelta(days=1)


@pytest.mark.parametrize(
    "make_trigger",
    [
        pytest.param(lambda: DateTrigger(run_date=_LATER), id="date"),
        pytest.param(
            lambda: CronTrigger(hour=9, minute=0, start_date=_LATER, timezone="UTC"),
            id="cron",
        ),
        pytest.param(
            lambda: CronTrigger(hour=9, minute=0, end_date=_ENDED, timezone="UTC"),
            id="cron-ended",
        ),
        pytest.param(
            lambda: IntervalTrigger(
                hours=1, start_date=_ENDED - timedelta(days=1), end_date=_ENDED
            ),
            id="interval-ended",
        ),
    ],
)
def test_a_resume_lands_where_apschedulers_resume_job_would(
    tmp_path, make_trigger: Callable[[], BaseTrigger]
):
    """Resume repeats ``resume_job``'s steps, to clear the mark in the same
    write, so it has to end where ``resume_job`` does: at the same next run
    time, or with the job removed when its trigger has no next fire time.
    An APScheduler upgrade that changes ``resume_job`` fails here."""
    with sqlite_scheduler(tmp_path) as scheduler:
        aps = scheduler.scheduler
        for job_id in ("ours", "apschedulers"):
            aps.add_job(
                execute_graph,
                kwargs=graph_args(job_id).model_dump(mode="json"),
                trigger=make_trigger(),
                jobstore=STORE,
                id=job_id,
                next_run_time=None,
            )

        assert scheduler.resume_execution_schedule("ours", USER) is True
        aps.resume_job("apschedulers", jobstore=STORE)

        ours = aps.get_job("ours", jobstore=STORE)
        theirs = aps.get_job("apschedulers", jobstore=STORE)
        assert (ours is None) == (theirs is None)
        if ours is not None and theirs is not None:
            assert ours.next_run_time == theirs.next_run_time


def test_a_job_stored_without_the_mark_reads_as_unmarked(tmp_path):
    """A job whose kwargs have no ``paused_by_expert_archive``, paused the
    plain way, reads as unmarked: its pause can't be told apart from the
    user's, so re-hiring leaves it paused."""
    with sqlite_scheduler(tmp_path) as scheduler:
        old_kwargs = graph_args().model_dump(mode="json")
        del old_kwargs["paused_by_expert_archive"]
        add_graph_job(scheduler, kwargs=old_kwargs)
        scheduler.scheduler.pause_job("job-1", jobstore=STORE)

        assert not stored(scheduler).paused_by_expert_archive
        assert resume(scheduler, only_if_paused_by_expert_archive=True) is False
        assert stored(scheduler).next_run_time == ""
        # The fire path reads the same kwargs.
        job = scheduler.scheduler.get_job("job-1", jobstore=STORE)
        assert GraphExecutionJobArgs(**job.kwargs).graph_id == "graph-1"


def test_an_edit_keeps_the_mark_and_rehire_resumes_the_edited_schedule(tmp_path):
    """Changing a schedule's timing says nothing about whether it should run,
    so the archive's pause still stands, and re-hiring brings the schedule
    back as the user left it: at the new time."""
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(scheduler)
        pause(scheduler, for_expert_archive=True)

        scheduler.update_graph_execution_schedule("job-1", USER, cron="0 7 * * *")

        edited = stored(scheduler)
        assert edited.next_run_time == "" and edited.paused_by_expert_archive
        assert resume(scheduler, only_if_paused_by_expert_archive=True) is True
        resumed = stored(scheduler)
        assert resumed.cron == "0 7 * * *" and resumed.next_run_time
        job = scheduler.scheduler.get_job("job-1", jobstore=STORE)
        assert (job.next_run_time.hour, job.next_run_time.minute) == (7, 0)


def test_a_create_sent_again_after_the_archive_paused_it_gets_that_schedule(
    tmp_path,
):
    """The mark is pause state, like the next run time, not part of what the
    create asked for: a retry whose first answer was lost is answered with the
    schedule, not refused as a different one under its key."""
    schedule_id = str(uuid.uuid4())
    args = graph_args(schedule_id).model_copy(update={"expert_id": None})
    with sqlite_scheduler(tmp_path) as scheduler:
        add_graph_job(
            scheduler, kwargs=args.model_dump(mode="json"), job_id=schedule_id
        )
        scheduler.pause_execution_schedule(schedule_id, USER, for_expert_archive=True)

        replayed = scheduler.add_graph_execution_schedule(
            user_id=USER,
            graph_id=args.graph_id,
            graph_version=args.graph_version,
            cron=CRON,
            input_data={},
            input_credentials={},
            name=args.agent_name,
            user_timezone="UTC",
            schedule_id=schedule_id,
        )

        assert replayed.id == schedule_id and replayed.paused_by_expert_archive
