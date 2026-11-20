import threading
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_MISSED
from apscheduler.job import Job
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from backend.executor import scheduler as scheduler_module
from backend.executor.jobstore import ResilientSQLAlchemyJobStore
from backend.executor.scheduler import (
    _MEMORY_CRON_MISFIRE_GRACE,
    _MEMORY_CRONS,
    SKILL_LEARNING_JOB_PREFIX,
    Scheduler,
    execute_copilot_turn,
    execute_graph,
)

_EXECUTION = "execution"
_SKILL_LEARNING_GRACE = _MEMORY_CRON_MISFIRE_GRACE[SKILL_LEARNING_JOB_PREFIX]


def test_old_user_schedules_run_late_while_memory_crons_and_pauses_keep_theirs(
    tmp_path, monkeypatch
):
    database_url = f"sqlite:///{(tmp_path / 'jobs.sqlite').as_posix()}"
    late = datetime.now(timezone.utc) - timedelta(seconds=3)
    _seed(
        database_url,
        _job("graph", execute_graph, next_run_time=late, misfire_grace_time=1),
        _job("turn", execute_copilot_turn, next_run_time=late, misfire_grace_time=1),
        _job("paused", execute_graph, next_run_time=None, misfire_grace_time=1),
        _job(
            "skill_learning",
            _MEMORY_CRONS[SKILL_LEARNING_JOB_PREFIX][0],
            next_run_time=datetime.now(timezone.utc) + timedelta(days=1),
            misfire_grace_time=_SKILL_LEARNING_GRACE,
            kwargs={"user_id": "user-1"},
        ),
    )
    monkeypatch.setattr(
        "backend.executor.scheduler.run_async", lambda coro, *_, **__: coro.close()
    )
    store = ResilientSQLAlchemyJobStore(url=database_url)
    restarted = BackgroundScheduler(
        jobstores={_EXECUTION: store},
        job_defaults={
            "coalesce": True,
            "max_instances": 1000,
            "misfire_grace_time": None,
        },
        timezone=timezone.utc,
    )
    ran: list[str] = []
    missed: list[str] = []
    both_ran = threading.Event()

    def record(event) -> None:
        (missed if event.code == EVENT_JOB_MISSED else ran).append(event.job_id)
        if {"graph", "turn"} <= set(ran):
            both_ran.set()

    restarted.add_listener(record, EVENT_JOB_EXECUTED | EVENT_JOB_MISSED)
    restarted.start(paused=True)
    service = Scheduler(register_system_tasks=False)
    service.scheduler = restarted
    service._execution_jobstore = store
    try:
        service._drop_stored_misfire_graces()
        restarted.resume()
        both_ran.wait(5)
    finally:
        restarted.shutdown()

    assert missed == []
    assert sorted(ran) == ["graph", "turn"]
    stored = _stored(database_url)
    assert {job_id: job.misfire_grace_time for job_id, job in stored.items()} == {
        "graph": None,
        "turn": None,
        "paused": None,
        "skill_learning": _SKILL_LEARNING_GRACE,
    }
    assert stored["paused"].next_run_time is None


def test_the_scheduler_drops_stored_graces_once_it_has_started(tmp_path, monkeypatch):
    database_url = f"sqlite:///{(tmp_path / 'jobs.sqlite').as_posix()}"
    _seed(
        database_url,
        _job(
            "graph",
            execute_graph,
            next_run_time=datetime.now(timezone.utc) + timedelta(days=1),
            misfire_grace_time=1,
        ),
    )
    for name in ("_event_loop", "_event_loop_thread", "_scheduler_instance"):
        monkeypatch.setattr(scheduler_module, name, getattr(scheduler_module, name))
    real_start = BackgroundScheduler.start
    service = Scheduler(register_system_tasks=True)
    with (
        patch.object(
            Scheduler, "_jobstore_database", return_value=(None, database_url)
        ),
        patch.object(Scheduler, "_start_schedule_index_backfill"),
        patch.object(
            BackgroundScheduler,
            "start",
            lambda self, *_, **__: real_start(self, paused=True),
        ),
        patch.object(scheduler_module, "load_dotenv"),
        # Blocks forever serving RPC; the start-up work before it is under test.
        patch("backend.util.service.AppService.run_service"),
    ):
        service.run_service()
    try:
        for thread in threading.enumerate():
            if thread.name == "DropStoredMisfireGraces":
                thread.join(10)
    finally:
        service.cleanup()
        if scheduler_module._event_loop is not None:
            scheduler_module._event_loop.close()

    assert _stored(database_url)["graph"].misfire_grace_time is None


def _job(job_id, func, *, next_run_time, misfire_grace_time, kwargs=None) -> dict:
    return {
        "id": job_id,
        "func": func,
        "kwargs": kwargs or {},
        # Next regular run two years out, so none falls inside the test.
        "trigger": CronTrigger(
            year=datetime.now(timezone.utc).year + 2,
            month=1,
            day=1,
            hour=0,
            minute=0,
            timezone=timezone.utc,
        ),
        "next_run_time": next_run_time,
        "misfire_grace_time": misfire_grace_time,
    }


def _seed(database_url: str, *jobs: dict) -> None:
    """Pickled into the store as rows saved by an older scheduler: with
    APScheduler's own options rather than the scheduler's job defaults."""
    store = ResilientSQLAlchemyJobStore(url=database_url)
    seeder = BackgroundScheduler(timezone=timezone.utc)
    store.start(seeder, _EXECUTION)
    try:
        for job in jobs:
            store.add_job(
                Job(
                    seeder,
                    executor="default",
                    args=(),
                    coalesce=True,
                    max_instances=1,
                    **job,
                )
            )
    finally:
        store.shutdown()


def _stored(database_url: str) -> dict[str, Job]:
    store = ResilientSQLAlchemyJobStore(url=database_url)
    store.start(BackgroundScheduler(timezone=timezone.utc), _EXECUTION)
    try:
        return {job.id: job for job in store.get_all_jobs()}
    finally:
        store.shutdown()
