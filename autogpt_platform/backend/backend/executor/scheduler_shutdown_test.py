"""Scheduler shutdown ordering (AUTOGPT-SERVER-8A8).

A real APScheduler with a real thread pool and a persistent SQLAlchemy jobstore,
like production (on SQLite here). A job is still running when
shutdown starts, and an interval job keeps falling due while shutdown waits for
it, so the scheduler's own loop wakes and queues up behind shutdown on the
jobstore lock. That is the window where the stock scheduler submitted to the
pool it had just shut down and advanced the job's next run without running it.
"""

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.base import STATE_STOPPED

from backend.executor.scheduler_shutdown import (
    DrainingBackgroundScheduler,
    shutdown_scheduler,
)

_TICK_S = 0.1
# Long enough for the scheduler loop to wake on several ticks and queue up
# behind the shutdown that holds the jobstore lock.
_HOLD_S = 1.0


class _Jobs:
    def __init__(self) -> None:
        self.long_started = threading.Event()
        self.release_long = threading.Event()
        self.long_finished = threading.Event()
        self.tick_runs: list[float] = []


# Persistent jobs are pickled by reference, so their functions and the state
# they touch live at module level.
_jobs = _Jobs()


def long_job() -> None:
    _jobs.long_started.set()
    _jobs.release_long.wait(timeout=30)
    _jobs.long_finished.set()


def tick() -> None:
    _jobs.tick_runs.append(time.monotonic())


@pytest.fixture
def jobs() -> _Jobs:
    global _jobs
    _jobs = _Jobs()
    yield _jobs
    _jobs.release_long.set()


def _jobstore_url(tmp_path: Path) -> str:
    return f"sqlite:///{tmp_path / 'jobs.sqlite'}"


def _start_scheduler(jobs: _Jobs, tmp_path: Path) -> DrainingBackgroundScheduler:
    scheduler = DrainingBackgroundScheduler(
        executors={"default": ThreadPoolExecutor(max_workers=4)},
        jobstores={"default": SQLAlchemyJobStore(url=_jobstore_url(tmp_path))},
        # The production defaults: missed runs collapse into one and are never
        # dropped for being late.
        job_defaults={"coalesce": True, "misfire_grace_time": None},
        timezone=timezone.utc,
    )
    scheduler.start()
    scheduler.add_job(long_job, id="long", next_run_time=datetime.now(timezone.utc))
    scheduler.add_job(tick, "interval", id="tick", seconds=_TICK_S)
    assert jobs.long_started.wait(timeout=5)
    return scheduler


def _wait_until_shutdown_holds_the_lock(scheduler: DrainingBackgroundScheduler):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if scheduler.state == STATE_STOPPED and not scheduler._jobstores_lock.acquire(
            blocking=False
        ):
            return
        if scheduler.state == STATE_STOPPED:
            scheduler._jobstores_lock.release()
        time.sleep(0.01)
    pytest.fail("shutdown never took the jobstore lock")


def test_shutdown_submits_nothing_new_and_waits_for_running_jobs(
    jobs: _Jobs, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    scheduler = _start_scheduler(jobs, tmp_path)

    caplog.set_level(logging.ERROR, logger="apscheduler")
    result: list[bool] = []
    shutdown = threading.Thread(
        target=lambda: result.append(shutdown_scheduler(scheduler, grace_seconds=10))
    )
    shutdown.start()
    _wait_until_shutdown_holds_the_lock(scheduler)
    stopped_at = time.monotonic()
    stopped_wall = datetime.now(timezone.utc)

    time.sleep(_HOLD_S)
    # Shutdown is still waiting on the job that was running when it began.
    assert shutdown.is_alive()
    assert not jobs.long_finished.is_set()

    jobs.release_long.set()
    shutdown.join(timeout=10)
    assert result == [True]
    assert jobs.long_finished.is_set()

    # Give a pass that queued behind shutdown the chance to run, as it did in
    # production once shutdown let go of the lock.
    time.sleep(_TICK_S * 5)

    assert not [r for r in caplog.records if "Error submitting job" in r.getMessage()]
    assert not [t for t in jobs.tick_runs if t > stopped_at]
    # The run that fell due during shutdown is still due in the jobstore, so the
    # next scheduler to start runs it instead of skipping past it.
    next_store = SQLAlchemyJobStore(url=_jobstore_url(tmp_path))
    next_store.start(scheduler, "default")
    stored = next_store.lookup_job("tick")
    next_store.shutdown()
    assert stored is not None and stored.next_run_time is not None
    # A pass that ran after shutdown would have moved it past the hold.
    assert stored.next_run_time < stopped_wall + timedelta(seconds=_HOLD_S / 2)


def test_shutdown_gives_up_on_a_stuck_job_after_the_grace_period(
    jobs: _Jobs, tmp_path: Path
):
    scheduler = _start_scheduler(jobs, tmp_path)

    started = time.monotonic()
    finished_in_time = shutdown_scheduler(scheduler, grace_seconds=0.5)
    elapsed = time.monotonic() - started

    assert finished_in_time is False
    assert 0.5 <= elapsed < 3
    assert not jobs.long_finished.is_set()
