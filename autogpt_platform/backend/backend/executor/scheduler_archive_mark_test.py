"""The archive's mark on a graph schedule (#14577), against a real APScheduler.

Re-hiring an expert resumes the schedules its archive paused and no others, so
the scheduler has to tell those pauses apart from every other kind. The mark
lives in the job's own kwargs. These check who sets it, who clears it, and that
a job stored before it existed reads as unmarked. The archive and re-hire flows
run end to end in ``api/features/experts/rehire_schedules_test.py``.
"""

import threading
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from backend.executor.scheduler import (
    GraphExecutionJobArgs,
    GraphExecutionJobInfo,
    Jobstores,
    Scheduler,
    _build_trigger,
    _job_to_info,
    execute_graph,
)

_STORE = Jobstores.EXECUTION.value
_USER = "user-1"
_CRON = "0 9 * * *"
# Only stops a hung test: a passing run never waits this long.
_HANG_SECONDS = 10


@contextmanager
def _scheduler(tmp_path: Path) -> Iterator[Scheduler]:
    aps = BackgroundScheduler()
    # A SQL store, as in production: every read is a copy of the stored job,
    # so a write built from an earlier read really can be stale.
    aps.add_jobstore(
        SQLAlchemyJobStore(url=f"sqlite:///{tmp_path}/jobs.sqlite"), _STORE
    )
    aps.start(paused=True)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.scheduler = aps
    try:
        yield scheduler
    finally:
        aps.shutdown(wait=False)


def _args(job_id: str = "job-1") -> GraphExecutionJobArgs:
    return GraphExecutionJobArgs(
        schedule_id=job_id,
        user_id=_USER,
        graph_id="graph-1",
        graph_version=1,
        agent_name="Morning report",
        cron=_CRON,
        input_data={},
        expert_id="expert-1",
    )


def _add(scheduler: Scheduler, kwargs: dict | None = None, job_id: str = "job-1"):
    """Stored the way ``_persist_schedule`` stores one, or with *kwargs*."""
    scheduler.scheduler.add_job(
        execute_graph,
        kwargs=_args(job_id).model_dump(mode="json") if kwargs is None else kwargs,
        trigger=_build_trigger(cron=_CRON, run_at=None, user_timezone="UTC"),
        jobstore=_STORE,
        id=job_id,
    )


def _stored(scheduler: Scheduler, job_id: str = "job-1") -> GraphExecutionJobInfo:
    info = _job_to_info(scheduler.scheduler.get_job(job_id, jobstore=_STORE))
    assert isinstance(info, GraphExecutionJobInfo)
    return info


def _pause(scheduler: Scheduler, **kwargs) -> bool:
    return scheduler.pause_execution_schedule("job-1", _USER, **kwargs)


def _resume(scheduler: Scheduler, **kwargs) -> bool:
    return scheduler.resume_execution_schedule("job-1", _USER, **kwargs)


def test_the_archive_marks_a_running_schedule_it_pauses(tmp_path):
    with _scheduler(tmp_path) as scheduler:
        _add(scheduler)

        assert _pause(scheduler, for_expert_archive=True) is True

        stored = _stored(scheduler)
        assert stored.next_run_time == "" and stored.paused_by_expert_archive


def test_the_archive_never_marks_a_schedule_someone_else_paused(tmp_path):
    with _scheduler(tmp_path) as scheduler:
        _add(scheduler)
        assert _pause(scheduler) is True

        assert _pause(scheduler, for_expert_archive=True) is False

        stored = _stored(scheduler)
        assert stored.next_run_time == "" and not stored.paused_by_expert_archive


def test_a_users_pause_takes_a_schedule_over_from_the_archive(tmp_path):
    """Already paused, so the pause changes nothing a listing shows, but the
    schedule is the user's to keep paused now: re-hiring leaves it."""
    with _scheduler(tmp_path) as scheduler:
        _add(scheduler)
        _pause(scheduler, for_expert_archive=True)

        assert _pause(scheduler) is False

        stored = _stored(scheduler)
        assert stored.next_run_time == "" and not stored.paused_by_expert_archive
        assert _resume(scheduler, only_if_paused_by_expert_archive=True) is False
        assert _stored(scheduler).next_run_time == ""


def test_rehire_resumes_a_schedule_the_archive_paused_and_clears_the_mark(
    tmp_path,
):
    with _scheduler(tmp_path) as scheduler:
        _add(scheduler)
        _pause(scheduler, for_expert_archive=True)

        assert _resume(scheduler, only_if_paused_by_expert_archive=True) is True

        stored = _stored(scheduler)
        assert stored.next_run_time and not stored.paused_by_expert_archive


def test_rehire_leaves_a_schedule_the_user_paused(tmp_path):
    with _scheduler(tmp_path) as scheduler:
        _add(scheduler)
        _pause(scheduler)

        assert _resume(scheduler, only_if_paused_by_expert_archive=True) is False

        assert _stored(scheduler).next_run_time == ""


def test_any_resume_clears_the_mark(tmp_path):
    """A user resuming a schedule the archive paused has taken it over too: a
    later pause of theirs must not leave it marked for the next re-hire."""
    with _scheduler(tmp_path) as scheduler:
        _add(scheduler)
        _pause(scheduler, for_expert_archive=True)

        assert _resume(scheduler) is True

        stored = _stored(scheduler)
        assert stored.next_run_time and not stored.paused_by_expert_archive


def test_a_schedule_stored_before_the_mark_existed_reads_as_unmarked(tmp_path):
    """Archiving before this change paused jobs with a plain pause and stored
    no mark at all. Such a pause can't be told apart from the user's, so
    re-hiring leaves it paused rather than risk resuming one they switched
    off: it stays listed as paused, for them to resume."""
    with _scheduler(tmp_path) as scheduler:
        old_kwargs = _args().model_dump(mode="json")
        del old_kwargs["paused_by_expert_archive"]
        _add(scheduler, kwargs=old_kwargs)
        scheduler.scheduler.pause_job("job-1", jobstore=_STORE)

        assert not _stored(scheduler).paused_by_expert_archive
        assert _resume(scheduler, only_if_paused_by_expert_archive=True) is False
        assert _stored(scheduler).next_run_time == ""
        # The fire path reads the same kwargs.
        job = scheduler.scheduler.get_job("job-1", jobstore=_STORE)
        assert GraphExecutionJobArgs(**job.kwargs).graph_id == "graph-1"


def test_an_edit_keeps_the_mark_and_rehire_resumes_the_edited_schedule(tmp_path):
    """Changing a schedule's timing says nothing about whether it should run,
    so the archive's pause still stands, and re-hiring brings the schedule
    back as the user left it: at the new time."""
    with _scheduler(tmp_path) as scheduler:
        _add(scheduler)
        _pause(scheduler, for_expert_archive=True)

        scheduler.update_graph_execution_schedule("job-1", _USER, cron="0 7 * * *")

        edited = _stored(scheduler)
        assert edited.next_run_time == "" and edited.paused_by_expert_archive
        assert _resume(scheduler, only_if_paused_by_expert_archive=True) is True
        resumed = _stored(scheduler)
        assert resumed.cron == "0 7 * * *" and resumed.next_run_time
        job = scheduler.scheduler.get_job("job-1", jobstore=_STORE)
        assert (job.next_run_time.hour, job.next_run_time.minute) == (7, 0)


def test_a_create_sent_again_after_the_archive_paused_it_gets_that_schedule(
    tmp_path,
):
    """The mark is pause state, like the next run time, not part of what the
    create asked for: a retry whose first answer was lost is answered with the
    schedule, not refused as a different one under its key."""
    schedule_id = str(uuid.uuid4())
    args = _args(schedule_id).model_copy(update={"expert_id": None})
    with _scheduler(tmp_path) as scheduler:
        _add(scheduler, kwargs=args.model_dump(mode="json"), job_id=schedule_id)
        scheduler.pause_execution_schedule(schedule_id, _USER, for_expert_archive=True)

        replayed = scheduler.add_graph_execution_schedule(
            user_id=_USER,
            graph_id=args.graph_id,
            graph_version=args.graph_version,
            cron=_CRON,
            input_data={},
            input_credentials={},
            name=args.agent_name,
            user_timezone="UTC",
            schedule_id=schedule_id,
        )

        assert replayed.id == schedule_id and replayed.paused_by_expert_archive


@pytest.mark.parametrize(
    ("archived", "flow"),
    [
        pytest.param(
            False,
            lambda scheduler: _pause(scheduler, for_expert_archive=True),
            id="archive-pauses",
        ),
        pytest.param(
            True,
            lambda scheduler: _resume(scheduler, only_if_paused_by_expert_archive=True),
            id="rehire-resumes",
        ),
    ],
)
def test_a_users_pause_racing_the_archive_flow_is_not_overwritten(
    tmp_path, archived: bool, flow: Callable[[Scheduler], bool]
):
    """The archive flow reads the job and writes it back in one step. A user's
    pause landing between the two would be overwritten: the archive would mark
    a schedule the user had just paused, or re-hire resume it. The user's pause
    is forced into that gap, and has to wait for the flow and land after it."""
    with _scheduler(tmp_path) as scheduler:
        _add(scheduler)
        if archived:
            _pause(scheduler, for_expert_archive=True)
        flow_has_read = threading.Event()
        flow_may_write = threading.Event()
        authorized_job = scheduler._authorized_job

        def read_then_hold(schedule_id: str, user_id: str, *, action: str):
            found = authorized_job(schedule_id, user_id, action=action)
            if threading.current_thread().name == "flow":
                flow_has_read.set()
                flow_may_write.wait(_HANG_SECONDS)
            return found

        with patch.object(scheduler, "_authorized_job", side_effect=read_then_hold):
            flow_thread = threading.Thread(target=flow, args=(scheduler,), name="flow")
            flow_thread.start()
            assert flow_has_read.wait(_HANG_SECONDS)
            user = threading.Thread(target=_pause, args=(scheduler,), name="user")
            user.start()
            # Without the lock the user's pause lands now; with it, it waits.
            user.join(1)
            flow_may_write.set()
            flow_thread.join(_HANG_SECONDS)
            user.join(_HANG_SECONDS)

        stored = _stored(scheduler)
        assert stored.next_run_time == "" and not stored.paused_by_expert_archive
