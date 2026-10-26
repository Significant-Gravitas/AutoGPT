import threading
from datetime import datetime, timedelta, timezone

import pytest
from apscheduler.events import EVENT_JOB_EXECUTED
from apscheduler.job import Job
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.executor.jobstore import ResilientSQLAlchemyJobStore
from backend.executor.scheduler import Scheduler, refresh_near_expiry_credentials

JOB_ID = "refresh_near_expiry_credentials"


def _scheduler(database_url):
    return BackgroundScheduler(
        jobstores={"execution": ResilientSQLAlchemyJobStore(url=database_url)},
        job_defaults={"coalesce": True, "misfire_grace_time": None, "max_instances": 1},
        timezone=timezone.utc,
    )


@pytest.mark.parametrize("paused", [False, True])
def test_persisted_idle_sweep_catches_up_once_after_restart(
    tmp_path, monkeypatch, paused
):
    database_url = f"sqlite:///{(tmp_path / 'jobs.sqlite').as_posix()}"
    due = datetime.now(timezone.utc).replace(hour=3, minute=0, second=0, microsecond=0)
    due -= timedelta(days=2)
    first = ResilientSQLAlchemyJobStore(url=database_url)
    seed_scheduler = BackgroundScheduler(timezone=timezone.utc)
    first.start(seed_scheduler, "execution")
    try:
        first.add_job(
            Job(
                seed_scheduler,
                func=refresh_near_expiry_credentials,
                id=JOB_ID,
                trigger=CronTrigger(hour=3, minute=0, timezone=timezone.utc),
                next_run_time=None if paused else due,
                executor="default",
                args=(),
                kwargs={},
                coalesce=True,
                misfire_grace_time=None,
                max_instances=1,
            )
        )
    finally:
        first.shutdown()

    restarted = _scheduler(database_url)
    restarted.start(paused=True)
    service = Scheduler(register_system_tasks=False)
    service.scheduler = restarted
    executed = threading.Event()
    calls = []

    def complete(coro, timeout):
        coro.close()
        calls.append(timeout)

    monkeypatch.setattr("backend.executor.scheduler.run_async", complete)
    restarted.add_listener(lambda event: executed.set(), EVENT_JOB_EXECUTED)
    try:
        service._register_idle_credential_sweep()
        job = restarted.get_job(JOB_ID, jobstore="execution")
        assert job is not None
        assert job.next_run_time == (None if paused else due)
        assert len(restarted.get_jobs(jobstore="execution")) == 1
        if not paused:
            restarted.resume()
            assert executed.wait(5)
            assert calls == [None]
            updated = restarted.get_job(JOB_ID, jobstore="execution")
            assert updated is not None
            assert updated.next_run_time > datetime.now(timezone.utc)
    finally:
        restarted.shutdown()


@pytest.mark.parametrize("previous", ["missing", "interval", "different_timezone"])
def test_idle_sweep_installs_or_migrates_the_daily_job(tmp_path, previous):
    scheduler = _scheduler(f"sqlite:///{(tmp_path / 'jobs.sqlite').as_posix()}")
    scheduler.start(paused=True)
    service = Scheduler(register_system_tasks=False)
    service.scheduler = scheduler
    try:
        if previous != "missing":
            scheduler.add_job(
                refresh_near_expiry_credentials,
                id=JOB_ID,
                trigger=(
                    IntervalTrigger(seconds=60)
                    if previous == "interval"
                    else CronTrigger(hour=3, minute=0, timezone="America/Chicago")
                ),
                jobstore="execution",
            )
        service._register_idle_credential_sweep()
        job = scheduler.get_job(JOB_ID, jobstore="execution")
        assert job is not None
        expected = CronTrigger(hour=3, minute=0, timezone=timezone.utc)
        assert str(job.trigger) == str(expected)
        assert job.next_run_time == expected.get_next_fire_time(
            None, datetime.now(timezone.utc)
        )
        assert job.max_instances == 1
        assert job.coalesce is True
        assert job.misfire_grace_time is None
        assert len(scheduler.get_jobs(jobstore="execution")) == 1
    finally:
        scheduler.shutdown()
