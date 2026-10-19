"""The scheduled cloud-storage cleanups: voice recordings, and the rest."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from apscheduler.executors.base import run_job
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from pytest_mock import MockerFixture

from backend.api.features.onboarding_dump.retention import RecordingPurgeResult
from backend.api.features.onboarding_dump.storage import RECORDING_PURGE_INTERVAL

# Imported here, not in the test: the executor conftest patches a function
# ``db_manager`` wraps at import, and an import under that patch would wrap
# the mock.
from backend.data.db_manager import DatabaseManager, DatabaseManagerAsyncClient
from backend.executor import scheduler
from backend.util.service import EXPOSED_FLAG


def _run(coro):
    return asyncio.run(coro)


def test_the_generic_cleanup_leaves_recordings_to_their_own_job(
    mocker: MockerFixture,
):
    purge = mocker.patch.object(scheduler, "purge_expired_recordings", AsyncMock())
    generic = mocker.patch.object(
        scheduler, "cleanup_expired_files_async", AsyncMock(return_value=0)
    )
    mocker.patch.object(scheduler, "run_async", side_effect=_run)

    scheduler.cleanup_expired_files()

    generic.assert_awaited_once()
    purge.assert_not_called()


def test_the_purge_job_clears_rows_through_the_database_manager(
    mocker: MockerFixture,
):
    db = MagicMock()
    mocker.patch.object(scheduler, "get_database_manager_async_client", return_value=db)
    purge = mocker.patch.object(
        scheduler,
        "purge_expired_recordings",
        AsyncMock(return_value=RecordingPurgeResult()),
    )
    mocker.patch.object(scheduler, "run_async", side_effect=_run)

    scheduler.purge_expired_recordings_job()

    purge.assert_awaited_once_with(db.clear_brain_dump_audio_paths)


def test_a_failed_purge_is_reported_as_an_error(
    mocker: MockerFixture, caplog: pytest.LogCaptureFixture
):
    """The job raises rather than swallowing the failure, so APScheduler logs
    it at ERROR with the traceback, which Sentry's logging integration
    captures. A sweep that keeps failing has to be seen: nothing else deletes
    the recordings."""
    mocker.patch.object(scheduler, "get_database_manager_async_client")
    mocker.patch.object(
        scheduler,
        "purge_expired_recordings",
        AsyncMock(side_effect=RuntimeError("bucket down")),
    )
    mocker.patch.object(scheduler, "run_async", side_effect=_run)
    background = BackgroundScheduler(
        jobstores={"default": MemoryJobStore()}, timezone=timezone.utc
    )
    background.start(paused=True)
    job = background.add_job(
        scheduler.purge_expired_recordings_job, id=scheduler.RECORDING_PURGE_JOB_ID
    )
    try:
        with caplog.at_level(logging.ERROR, logger="apscheduler"):
            run_job(
                job,
                "default",
                [datetime.now(timezone.utc)],
                "apscheduler.executors.default",
            )
    finally:
        background.shutdown(wait=False)

    [record] = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], RuntimeError)


class TestRecordingPurgeRegistration:
    """ "Within 48 hours" holds only if the sweep really runs every
    ``RECORDING_PURGE_INTERVAL``. Re-adding an interval job on every start
    re-arms it at start + interval, so restarts would postpone the sweep;
    an unchanged job is left alone instead, and an overdue run fires on
    start."""

    @pytest.fixture
    def sched(self):
        background = BackgroundScheduler(
            jobstores={scheduler.Jobstores.EXECUTION.value: MemoryJobStore()},
            timezone=timezone.utc,
        )
        background.start(paused=True)
        sched = scheduler.Scheduler.__new__(scheduler.Scheduler)
        sched.scheduler = background
        yield sched
        background.shutdown(wait=False)

    def _job(self, sched):
        return sched.scheduler.get_job(
            scheduler.RECORDING_PURGE_JOB_ID,
            jobstore=scheduler.Jobstores.EXECUTION.value,
        )

    def test_registers_the_purge_at_its_interval(self, sched):
        sched._register_recording_purge()

        job = self._job(sched)
        assert job.func is scheduler.purge_expired_recordings_job
        assert isinstance(job.trigger, IntervalTrigger)
        assert job.trigger.interval == RECORDING_PURGE_INTERVAL

    def test_a_restart_does_not_postpone_an_overdue_purge(self, sched):
        sched._register_recording_purge()
        overdue = datetime.now(timezone.utc) - timedelta(minutes=30)
        sched.scheduler.modify_job(
            scheduler.RECORDING_PURGE_JOB_ID,
            jobstore=scheduler.Jobstores.EXECUTION.value,
            next_run_time=overdue,
        )

        sched._register_recording_purge()

        assert self._job(sched).next_run_time == overdue

    def test_a_job_at_another_interval_is_replaced(self, sched):
        sched.scheduler.add_job(
            scheduler.purge_expired_recordings_job,
            "interval",
            hours=6,
            id=scheduler.RECORDING_PURGE_JOB_ID,
            jobstore=scheduler.Jobstores.EXECUTION.value,
        )

        sched._register_recording_purge()

        assert self._job(sched).trigger.interval == RECORDING_PURGE_INTERVAL


def test_the_purge_reaches_the_database_manager_endpoint_it_calls():
    """The server serves each exposed method at ``/{attribute name}`` and
    the client POSTs to ``/{function __name__}``; if they differ the purge
    404s on every run."""
    served = {
        name
        for name, attr in vars(DatabaseManager).items()
        if getattr(attr, EXPOSED_FLAG, False)
    }
    called = DatabaseManagerAsyncClient.clear_brain_dump_audio_paths.__name__

    assert called == "clear_brain_dump_audio_paths"
    assert called in served
