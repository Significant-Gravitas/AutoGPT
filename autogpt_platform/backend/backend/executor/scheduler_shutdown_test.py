"""Stopping the scheduler keeps a job that falls due as it stops, and runs it once.

The race tests start the real ``Scheduler.run_service`` APScheduler (its pool
and job defaults, two starts sharing one SQLite jobstore), stop it through
``Scheduler.cleanup`` as SIGTERM does, and force their race in order through
the scheduler's jobstore lock.
"""

import threading
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from apscheduler.events import EVENT_SCHEDULER_SHUTDOWN
from apscheduler.job import Job
from apscheduler.schedulers import (
    SchedulerAlreadyRunningError,
    SchedulerNotRunningError,
)
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger

from backend.executor import scheduler as scheduler_module
from backend.executor.jobstore import ResilientSQLAlchemyJobStore
from backend.executor.scheduler import Jobstores, Scheduler
from backend.executor.scheduler_shutdown import StopSafeBackgroundScheduler

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

    with _started_service(jobstore) as (service, lock, stopper):
        # Hourly: a second run falls due only if the test stalls for an hour.
        service.scheduler.add_job(
            _record_run,
            IntervalTrigger(hours=1, timezone=timezone.utc),
            args=["hourly"],
            id="hourly",
            next_run_time=due,
            jobstore=_EXECUTION,
        )
        shut_down = threading.Event()
        service.scheduler.add_listener(
            lambda _: shut_down.set(), EVENT_SCHEDULER_SHUTDOWN
        )
        scheduler_thread = service.scheduler._thread
        lock.hold(scheduler_thread)
        try:
            service.scheduler.resume()
            assert lock.wait_until_asked_by(scheduler_thread)
            stopper.start()
            assert shut_down.wait(_WAIT_S)
        finally:
            lock.let_go()

    assert _runs == []
    assert _stored_jobs(jobstore)["hourly"].next_run_time == due

    with _started_service(jobstore) as (service, _, _):
        service.scheduler.resume()
        _wait_for(lambda: "hourly" in _runs)
        service.cleanup()

    assert _runs == ["hourly"]
    assert _stored_jobs(jobstore)["hourly"].next_run_time == due + timedelta(hours=1)


def test_a_one_shot_handed_to_the_pool_as_the_stop_begins_runs_once(tmp_path):
    """Stock shutdown marks the scheduler stopped outside the jobstore lock.
    Landing between a job pass handing a one-shot to the pool and removing
    it, that made the removal look only at pending jobs: the run went ahead,
    the row stayed, and the next start ran the one-shot again."""
    jobstore = f"sqlite:///{tmp_path / 'jobs.sqlite'}"
    stop_asked_first: list[bool] = []

    with _started_service(jobstore) as (service, lock, stopper):
        removing = threading.Event()
        remove_job = service.scheduler.remove_job

        # A job pass removes a one-shot it handed to the pool through
        # self.remove_job, so this instance attribute is what it calls.
        def remove_once_the_stop_asks_for_the_lock(job_id, jobstore=None):
            # The job pass, holding the jobstore lock; the pool has the run.
            if job_id == "one-shot":
                removing.set()
                stop_asked_first.append(lock.wait_until_asked_by(stopper))
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

    assert stop_asked_first == [True]
    assert _runs == ["one-shot"]
    assert "one-shot" not in _stored_jobs(jobstore)

    with _started_service(jobstore) as (service, _, _):
        # Due after the one-shot, so the pass that runs it would take a row
        # left behind too, and every run it starts ends before the stop returns.
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


def test_only_a_start_that_goes_ahead_clears_a_stop():
    """A stop asked of a scheduler that wasn't running doesn't keep its first
    start from running jobs, and a start refused mid-stop keeps the stop."""
    scheduler = StopSafeBackgroundScheduler(timezone=timezone.utc)
    with pytest.raises(SchedulerNotRunningError):
        scheduler.shutdown()
    scheduler.add_job(_record_run, args=["first"], id="first")
    scheduler.start()
    try:
        _wait_for(lambda: "first" in _runs)
        # As between a stop's mark and APScheduler flipping its state.
        scheduler._stopping = True
        with pytest.raises(SchedulerAlreadyRunningError):
            scheduler.start()
        assert scheduler._stopping
    finally:
        scheduler.shutdown()


class _WatchedLock:
    """The scheduler's jobstore lock, noting each thread that asks for it and
    able to keep one waiting at its ask until ``let_go``: how a test orders a
    job pass against a stop."""

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
            self._changed.wait_for(lambda: self._held is not me)
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
            # So that waiting for its ask means one made since.
            self._askers.discard(thread)

    def let_go(self) -> None:
        with self._changed:
            self._held = None
            self._changed.notify_all()

    def wait_until_asked_by(self, thread: threading.Thread) -> bool:
        with self._changed:
            return self._changed.wait_for(lambda: thread in self._askers, _WAIT_S)


@contextmanager
def _started_service(
    jobstore: str,
) -> Iterator[tuple[Scheduler, _WatchedLock, threading.Thread]]:
    """``Scheduler.run_service`` on *jobstore*, its APScheduler paused until
    the test resumes it and its jobstore lock watched from before its thread
    starts, with a thread a test may start to stop it as SIGTERM does. On the
    way out the service is stopped, whether or not its start finished."""
    real_start = BackgroundScheduler.start
    lock = _WatchedLock()

    def start_paused(self: BackgroundScheduler, *_, **__) -> None:
        # The override takes this lock: an APScheduler without it fails here.
        assert self._jobstores_lock is not None
        self._jobstores_lock = lock
        real_start(self, paused=True)

    service = Scheduler(register_system_tasks=False)
    stop_errors: list[BaseException] = []

    def stop() -> None:
        try:
            service.cleanup()
        except BaseException as e:
            stop_errors.append(e)

    stopper = threading.Thread(target=stop, name="Stopper", daemon=True)
    try:
        with (
            patch.object(
                Scheduler, "_jobstore_database", return_value=(None, jobstore)
            ),
            patch.object(Scheduler, "_start_schedule_index_backfill"),
            patch.object(Scheduler, "_load_blocks_once_serving"),
            patch.object(BackgroundScheduler, "start", start_paused),
            patch.object(scheduler_module, "load_dotenv"),
            # Blocks forever serving RPC; the start-up before it is what's needed.
            patch("backend.util.service.AppService.run_service"),
        ):
            service.run_service()
        yield service, lock, stopper
    finally:
        _stop(service, stopper)
    if stop_errors:
        raise stop_errors[0]


def _stop(service: Scheduler, stopper: threading.Thread) -> None:
    """Stop *service* if its test didn't, then close what its start opened:
    the event loop, and the jobstores of a start that failed part-way."""
    if stopper.ident is not None:
        # Twice the longest wait inside the stop, once any held pass is let go.
        stopper.join(2 * _WAIT_S)
        if stopper.is_alive():
            pytest.fail("the scheduler's stop never finished")
    # Run last to first, each tried even if one after it fails; a failure
    # still raises.
    with ExitStack() as closing:
        closing.callback(_close_event_loop)
        # Already closed by a stop; these are for a start that failed part-way.
        for store in service._persistent_jobstores.values():
            closing.callback(store.shutdown)
        if (
            stopper.ident is None
            and service._scheduler is not None
            and service._scheduler.running
        ):
            closing.callback(service.cleanup)


def _close_event_loop() -> None:
    """Stop and close the event loop ``run_service`` started."""
    loop, thread = scheduler_module._event_loop, scheduler_module._event_loop_thread
    if loop is None or loop.is_closed():
        return
    if thread is not None and thread.is_alive():
        loop.call_soon_threadsafe(loop.stop)
        thread.join(_WAIT_S)
        if thread.is_alive():
            pytest.fail("the scheduler's event loop never stopped")
    if not loop.is_running():
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
