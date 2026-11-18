"""Stopping the scheduler keeps a job that falls due as it stops, and runs it once.

Each test starts the real ``Scheduler.run_service`` APScheduler (its pool and
job defaults, two starts sharing one SQLite jobstore), stops it through
``Scheduler.cleanup`` as SIGTERM does, and forces its race in order through
the scheduler's jobstore lock.
"""

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from apscheduler.events import EVENT_SCHEDULER_SHUTDOWN
from apscheduler.job import Job
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.executor import scheduler as scheduler_module
from backend.executor.jobstore import ResilientSQLAlchemyJobStore
from backend.executor.scheduler import Jobstores, Scheduler

_EXECUTION = Jobstores.EXECUTION.value
# Only a broken run waits this long.
_WAIT_S = 30

_runs: list[str] = []
_ran = threading.Condition()


def _record_run(name: str) -> None:
    with _ran:
        _runs.append(name)
        _ran.notify_all()


@pytest.fixture(autouse=True)
def _fresh_runs(monkeypatch):
    _runs.clear()
    # run_service replaces these; the next test gets its own back.
    for name in ("_event_loop", "_event_loop_thread", "_scheduler_instance"):
        monkeypatch.setattr(scheduler_module, name, getattr(scheduler_module, name))


def test_a_job_due_as_the_scheduler_stops_runs_once_on_the_next_start(tmp_path):
    """AUTOGPT-SERVER-8A8. A job pass that woke as the stop began got the
    jobstore lock only once the pool was shut down. It handed the due job to
    the dead pool ("cannot schedule new futures after shutdown") and still
    moved the job's next run on, so the run was lost."""
    jobstore = f"sqlite:///{tmp_path / 'jobs.sqlite'}"
    due = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=1)

    with _started_service(jobstore) as service:
        lock = _WatchedLock()
        service.scheduler._jobstores_lock = lock
        service.scheduler.add_job(
            _record_run,
            IntervalTrigger(minutes=1, timezone=timezone.utc),
            args=["minutely"],
            id="minutely",
            next_run_time=due,
            jobstore=_EXECUTION,
        )
        shut_down = threading.Event()
        service.scheduler.add_listener(
            lambda _: shut_down.set(), EVENT_SCHEDULER_SHUTDOWN
        )
        scheduler_thread = service.scheduler._thread
        lock.hold(scheduler_thread)
        service.scheduler.resume()
        assert lock.wait_until_asked_by(scheduler_thread)

        stopper = threading.Thread(target=service.cleanup, name="Stopper")
        stopper.start()
        assert shut_down.wait(_WAIT_S)
        lock.let_go()
        stopper.join(_WAIT_S)
        assert not stopper.is_alive()

    assert _runs == []
    assert _stored_jobs(jobstore)["minutely"].next_run_time == due

    with _started_service(jobstore) as service:
        service.scheduler.resume()
        _wait_for(lambda: "minutely" in _runs)
        service.cleanup()

    assert _runs == ["minutely"]
    assert _stored_jobs(jobstore)["minutely"].next_run_time == due + timedelta(
        minutes=1
    )


def test_a_one_shot_handed_to_the_pool_as_the_stop_begins_runs_once(tmp_path):
    """Stock shutdown marks the scheduler stopped outside the jobstore lock.
    Landing between a job pass handing a one-shot to the pool and removing
    it, that made the removal look only at pending jobs: the run went ahead,
    the row stayed, and the next start ran the one-shot again."""
    jobstore = f"sqlite:///{tmp_path / 'jobs.sqlite'}"

    with _started_service(jobstore) as service:
        lock = _WatchedLock()
        service.scheduler._jobstores_lock = lock
        stopper = threading.Thread(target=service.cleanup, name="Stopper")
        removing = threading.Event()
        remove_job = service.scheduler.remove_job

        def remove_once_the_stop_asks_for_the_lock(job_id, jobstore=None):
            # The job pass, holding the jobstore lock; the pool has the run.
            if job_id == "one-shot":
                removing.set()
                lock.wait_until_asked_by(stopper)
            return remove_job(job_id, jobstore)

        service.scheduler.remove_job = remove_once_the_stop_asks_for_the_lock
        service.scheduler.add_job(
            _record_run,
            DateTrigger(datetime.now(timezone.utc), timezone=timezone.utc),
            args=["one-shot"],
            id="one-shot",
            jobstore=_EXECUTION,
        )
        service.scheduler.resume()
        assert removing.wait(_WAIT_S)
        stopper.start()
        stopper.join(_WAIT_S)
        assert not stopper.is_alive()

    with _started_service(jobstore) as service:
        # Due after the one-shot, so the pass that runs it takes a row left
        # behind too, and every run it starts ends before the stop returns.
        service.scheduler.add_job(
            _record_run,
            DateTrigger(datetime.now(timezone.utc), timezone=timezone.utc),
            args=["later"],
            id="later",
            jobstore=_EXECUTION,
        )
        service.scheduler.resume()
        _wait_for(lambda: "later" in _runs)
        service.cleanup()

    assert sorted(_runs) == ["later", "one-shot"]


class _WatchedLock:
    """The scheduler's jobstore lock, noting each thread that asks for it and
    able to keep one waiting at its ask: how a test orders a job pass
    against a stop."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._changed = threading.Condition()
        self._askers: set[threading.Thread] = set()
        self._held: threading.Thread | None = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        me = threading.current_thread()
        with self._changed:
            self._askers.add(me)
            self._changed.notify_all()
            self._changed.wait_for(lambda: self._held is not me, _WAIT_S)
        return self._lock.acquire(blocking, timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> bool:
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()

    def hold(self, thread: threading.Thread) -> None:
        with self._changed:
            self._held = thread

    def let_go(self) -> None:
        with self._changed:
            self._held = None
            self._changed.notify_all()

    def wait_until_asked_by(self, thread: threading.Thread) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: thread in self._askers, _WAIT_S)


@contextmanager
def _started_service(jobstore: str) -> Iterator[Scheduler]:
    """``Scheduler.run_service`` on *jobstore*, its APScheduler paused until
    the test resumes it, and stopped on the way out if the test did not."""
    real_start = BackgroundScheduler.start
    service = Scheduler(register_system_tasks=False)
    with (
        patch.object(Scheduler, "_jobstore_database", return_value=(None, jobstore)),
        patch.object(Scheduler, "_start_schedule_index_backfill"),
        patch.object(
            BackgroundScheduler,
            "start",
            lambda self, *_, **__: real_start(self, paused=True),
        ),
        patch.object(scheduler_module, "load_dotenv"),
        # Blocks forever serving RPC; the start-up before it is what's needed.
        patch("backend.util.service.AppService.run_service"),
    ):
        service.run_service()
    try:
        yield service
    finally:
        if service.scheduler.running:
            service.cleanup()
        loop = scheduler_module._event_loop
        if loop is not None and not loop.is_running():
            loop.close()


def _wait_for(condition: Callable[[], bool]) -> None:
    with _ran:
        assert _ran.wait_for(condition, _WAIT_S), f"runs so far: {_runs}"


def _stored_jobs(jobstore: str) -> dict[str, Job]:
    store = ResilientSQLAlchemyJobStore(url=jobstore)
    store.start(BackgroundScheduler(timezone=timezone.utc), _EXECUTION)
    try:
        return {job.id: job for job in store.get_all_jobs()}
    finally:
        store.shutdown()
