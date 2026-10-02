"""Pause provenance under concurrent lifecycle and owner requests."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timezone
from pathlib import Path
from threading import Event, RLock
from types import TracebackType

import pytest
from apscheduler.job import Job
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.schedulers.background import BackgroundScheduler

from backend.executor.scheduler import GraphExecutionJobArgs, Jobstores, Scheduler


def _noop(**kwargs) -> None:
    """Provide a serializable job target that accepts schedule metadata."""


@pytest.fixture
def persistent_scheduler(tmp_path: Path):
    """Yield a paused scheduler whose SQL store reloads each job independently."""
    service = Scheduler(register_system_tasks=False)
    service.scheduler = BackgroundScheduler(
        jobstores={
            Jobstores.EXECUTION.value: SQLAlchemyJobStore(
                url=f"sqlite:///{tmp_path / 'schedules.sqlite'}"
            )
        },
        timezone=timezone.utc,
    )
    service.scheduler.start(paused=True)
    service.scheduler.add_job(
        _noop,
        trigger="cron",
        hour=9,
        id="schedule-1",
        jobstore=Jobstores.EXECUTION.value,
        kwargs=GraphExecutionJobArgs(
            schedule_id="schedule-1",
            user_id="owner",
            graph_id="graph-1",
            graph_version=1,
            cron="0 9 * * *",
            input_data={},
            expert_id="expert-1",
        ).model_dump(mode="json"),
    )
    yield service
    service.scheduler.shutdown(wait=False)


class _ObservedLock:
    """Expose contention so the test never relies on timing or sleeps."""

    def __init__(self, second_request_ready: Event):
        """Create a reentrant lock and its contention signal."""
        self._lock = RLock()
        self._second_request_ready = second_request_ready

    def __enter__(self) -> None:
        """Signal a blocked acquisition before waiting for the current owner."""
        if not self._lock.acquire(blocking=False):
            self._second_request_ready.set()
            self._lock.acquire()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release one acquisition when the critical section exits."""
        self._lock.release()


@pytest.mark.parametrize("first_by_archive", [False, True])
def test_concurrent_pause_preserves_the_first_pause_provenance(
    persistent_scheduler: Scheduler,
    monkeypatch: pytest.MonkeyPatch,
    first_by_archive: bool,
) -> None:
    """A competing pause must preserve the first writer's persisted provenance."""
    service = persistent_scheduler
    first_read = Event()
    release_first = Event()
    first_written = Event()
    second_request_ready = Event()
    authorized_job = service._authorized_job

    def coordinated_authorize(*args, **kwargs):
        """Order independent SQL snapshots around the first pause's write."""
        job, info = authorized_job(*args, **kwargs)
        if not first_read.is_set():
            first_read.set()
            assert release_first.wait(5)
        else:
            second_request_ready.set()
            assert first_written.wait(5)
        return job, info

    monkeypatch.setattr(service, "_authorized_job", coordinated_authorize)
    monkeypatch.setattr(
        service.scheduler, "_jobstores_lock", _ObservedLock(second_request_ready)
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            service.pause_execution_schedule,
            "schedule-1",
            "owner",
            first_by_archive,
        )
        try:
            assert first_read.wait(5)
            second = executor.submit(
                service.pause_execution_schedule,
                "schedule-1",
                "owner",
                not first_by_archive,
            )
            # The second request has either read its own SQL snapshot or is
            # blocked by the first request's read/check/write critical section.
            assert second_request_ready.wait(5)
            release_first.set()
            assert first.result(timeout=5) is True
            first_written.set()
            second_changed = second.result(timeout=5)
        finally:
            release_first.set()
            first_written.set()

    job = service.scheduler.get_job("schedule-1", jobstore=Jobstores.EXECUTION.value)
    assert job is not None and job.next_run_time is None
    assert job.kwargs["paused_by_expert_archive"] is first_by_archive
    assert second_changed is False


def test_failed_archive_pause_leaves_no_half_paused_persisted_job(
    persistent_scheduler: Scheduler, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed archive write must leave the schedule active and safely retryable."""
    service = persistent_scheduler
    jobstore = service.scheduler._lookup_jobstore(Jobstores.EXECUTION.value)
    update_job = jobstore.update_job

    def fail_archive_write(job: Job) -> None:
        """Reject the provenance write while allowing any preceding partial pause."""
        if job.kwargs["paused_by_expert_archive"]:
            raise RuntimeError("Failed to persist archive provenance")
        update_job(job)

    with monkeypatch.context() as patch:
        patch.setattr(jobstore, "update_job", fail_archive_write)
        with pytest.raises(RuntimeError, match="Failed to persist archive provenance"):
            service.pause_execution_schedule(
                "schedule-1", "owner", by_expert_archive=True
            )

    job = service.scheduler.get_job("schedule-1", jobstore=Jobstores.EXECUTION.value)
    assert job is not None and job.next_run_time is not None
    assert job.kwargs["paused_by_expert_archive"] is False

    assert service.pause_execution_schedule(
        "schedule-1", "owner", by_expert_archive=True
    )
    retried_job = service.scheduler.get_job(
        "schedule-1", jobstore=Jobstores.EXECUTION.value
    )
    assert retried_job is not None and retried_job.next_run_time is None
    assert retried_job.kwargs["paused_by_expert_archive"] is True
