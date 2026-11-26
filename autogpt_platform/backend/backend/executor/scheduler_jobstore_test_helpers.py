"""A real APScheduler on a SQL job store, for scheduler tests that read and
write stored jobs (#14577).

SQL, not ``MemoryJobStore``: every read of a SQL store is a copy, as in
production, so a write built from an earlier read really can be stale. The
memory store hands back the stored job itself, which hides such races.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from backend.executor.scheduler import (
    CopilotTurnJobArgs,
    GraphExecutionJobArgs,
    GraphExecutionJobInfo,
    Jobstores,
    Scheduler,
    _build_trigger,
    _job_to_info,
    execute_copilot_turn,
    execute_graph,
)

STORE = Jobstores.EXECUTION.value
USER = "user-1"
CRON = "0 9 * * *"


@contextmanager
def sqlite_scheduler(tmp_path: Path) -> Iterator[Scheduler]:
    """The scheduler service's methods over a started, paused APScheduler:
    nothing fires, and no background pass takes the jobstore lock."""
    aps = BackgroundScheduler()
    aps.add_jobstore(SQLAlchemyJobStore(url=f"sqlite:///{tmp_path}/jobs.sqlite"), STORE)
    aps.start(paused=True)
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.scheduler = aps
    try:
        yield scheduler
    finally:
        aps.shutdown(wait=False)


def graph_args(job_id: str = "job-1") -> GraphExecutionJobArgs:
    return GraphExecutionJobArgs(
        schedule_id=job_id,
        user_id=USER,
        graph_id="graph-1",
        graph_version=1,
        agent_name="Morning report",
        cron=CRON,
        input_data={},
        expert_id="expert-1",
    )


def add_graph_job(
    scheduler: Scheduler, kwargs: dict | None = None, job_id: str = "job-1"
) -> None:
    """Stored the way ``_persist_schedule`` stores one, or with *kwargs*."""
    scheduler.scheduler.add_job(
        execute_graph,
        kwargs=graph_args(job_id).model_dump(mode="json") if kwargs is None else kwargs,
        trigger=_build_trigger(cron=CRON, run_at=None, user_timezone="UTC"),
        jobstore=STORE,
        id=job_id,
    )


def add_follow_up(scheduler: Scheduler, job_id: str = "job-1") -> None:
    """A repeating follow-up, as ``add_copilot_turn_schedule`` stores one."""
    args = CopilotTurnJobArgs(
        schedule_id=job_id,
        user_id=USER,
        message="Check the queue.",
        cron=CRON,
        user_timezone="UTC",
    )
    scheduler.scheduler.add_job(
        execute_copilot_turn,
        kwargs=args.model_dump(mode="json"),
        trigger=_build_trigger(cron=CRON, run_at=None, user_timezone="UTC"),
        jobstore=STORE,
        id=job_id,
    )


def stored(scheduler: Scheduler, job_id: str = "job-1") -> GraphExecutionJobInfo:
    job = scheduler.scheduler.get_job(job_id, jobstore=STORE)
    assert job is not None, f"job {job_id} is gone"
    info = _job_to_info(job)
    assert isinstance(info, GraphExecutionJobInfo)
    return info


def pause(scheduler: Scheduler, for_expert_archive: bool = False) -> bool:
    return scheduler.pause_execution_schedule(
        "job-1", USER, for_expert_archive=for_expert_archive
    )


def resume(
    scheduler: Scheduler, only_if_paused_by_expert_archive: bool = False
) -> bool:
    return scheduler.resume_execution_schedule(
        "job-1",
        USER,
        only_if_paused_by_expert_archive=only_if_paused_by_expert_archive,
    )
