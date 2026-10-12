"""Which scheduled runs have taken effect, so shutdown can hand back the rest.

APScheduler advances a job (or removes a one-shot) when it submits the run,
before the run does anything. A run cancelled at shutdown before its effect
is therefore lost, and one whose effect had happened must not be replayed.

The ledger records every submitted run. A dispatch marks the moment its effect
commits with :func:`committing`. At shutdown :meth:`RunLedger.close` waits for
commits in progress, then marks every run that has not committed as abandoned:
it can no longer commit, and :meth:`RunLedger.give_back` returns it to the
jobstore (rolling the job's next run time back, or re-adding a one-shot) so
the next scheduler runs it once. Jobs that never mark a commit point (system
maintenance) count as run when they start.
"""

import contextlib
import logging
import threading
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any, Literal

from apscheduler.executors.base import run_job
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.job import Job
from apscheduler.schedulers.base import BaseScheduler
from apscheduler.triggers.date import DateTrigger

logger = logging.getLogger(__name__)

RunState = Literal["pending", "committing", "done", "abandoned"]

_bound = threading.local()


class RunAbandoned(Exception):
    """Shutdown handed this run back to the jobstore before it took effect."""


class ScheduledRun:
    def __init__(
        self, ledger: "RunLedger", job: Job, run_time: datetime, state: RunState
    ):
        self.ledger = ledger
        self.job_id: str = job.id
        self.jobstore: str | None = job._jobstore_alias
        self.job_state: dict[str, Any] = job.__getstate__()
        self.run_time = run_time
        self.state: RunState = state


class RunLedger:
    def __init__(self, commit_aware: Callable[[Job], bool]) -> None:
        self._commit_aware = commit_aware
        self._cond = threading.Condition()
        self._runs: set[ScheduledRun] = set()
        self._closed = False

    def open(self, job: Job, run_times: list[datetime]) -> ScheduledRun:
        state: RunState = "pending" if self._commit_aware(job) else "done"
        with self._cond:
            if self._closed and state == "pending":
                state = "abandoned"
            run = ScheduledRun(self, job, min(run_times), state)
            self._runs.add(run)
        return run

    def finish(self, run: ScheduledRun) -> None:
        with self._cond:
            if run.state != "abandoned":
                self._runs.discard(run)

    def close(self, commit_wait: float) -> list[ScheduledRun]:
        """Stop commits; returns the runs that hadn't committed so far.

        Runs submitted after this are abandoned as they are opened.

        Commits already in progress get ``commit_wait`` to finish. One that
        doesn't is counted as run, the only way a run can be lost here.
        """
        with self._cond:
            self._closed = True
            self._cond.wait_for(
                lambda: all(r.state != "committing" for r in self._runs), commit_wait
            )
            for run in self._runs:
                if run.state == "pending":
                    run.state = "abandoned"
            return [r for r in self._runs if r.state == "abandoned"]

    def give_back(self, scheduler: BaseScheduler, lock_wait: float) -> bool:
        """Return abandoned runs to the jobstore; False if it can't be done."""
        lock = scheduler._jobstores_lock
        if not lock.acquire(timeout=lock_wait):
            logger.error("Could not hand abandoned runs back: jobstore is busy")
            return False
        try:
            with self._cond:
                runs = [r for r in self._runs if r.state == "abandoned"]
            for run in runs:
                _give_back_run(scheduler, run)
            return True
        finally:
            lock.release()

    def _begin_commit(self, run: ScheduledRun) -> None:
        with self._cond:
            if self._closed or run.state != "pending":
                raise RunAbandoned(f"Run of job {run.job_id} was handed back")
            run.state = "committing"

    def _end_commit(self, run: ScheduledRun) -> None:
        with self._cond:
            run.state = "done"
            self._cond.notify_all()


def current_run() -> ScheduledRun | None:
    """The scheduled run this worker thread is executing, if any."""
    return getattr(_bound, "run", None)


@contextlib.contextmanager
def committing(run: ScheduledRun | None) -> Iterator[None]:
    """Mark the block that makes a scheduled run take effect.

    Raises :class:`RunAbandoned` instead of entering when shutdown has handed
    the run back. Keep the block to the effect itself: shutdown waits for it.
    """
    if run is None:
        yield
        return
    run.ledger._begin_commit(run)
    try:
        yield
    finally:
        run.ledger._end_commit(run)


class LedgerThreadPoolExecutor(ThreadPoolExecutor):
    """APScheduler's thread pool executor, recording each run in a ledger."""

    def __init__(self, max_workers: int, ledger: RunLedger) -> None:
        super().__init__(max_workers=max_workers)
        self._ledger = ledger

    def _do_submit_job(self, job: Job, run_times: list[datetime]) -> None:
        run = self._ledger.open(job, run_times)

        def callback(f) -> None:
            exc = f.exception()
            if exc:
                self._run_job_error(job.id, exc, exc.__traceback__)
            else:
                self._run_job_success(job.id, f.result())

        f = self._pool.submit(
            _run_bound, run, job, job._jobstore_alias, run_times, self._logger.name
        )
        f.add_done_callback(callback)


def _run_bound(run: ScheduledRun, *args: Any) -> Any:
    _bound.run = run
    try:
        return run_job(*args)
    finally:
        _bound.run = None
        run.ledger.finish(run)


def _give_back_run(scheduler: BaseScheduler, run: ScheduledRun) -> None:
    store = scheduler._lookup_jobstore(run.jobstore)
    current = store.lookup_job(run.job_id)
    if current is None:
        # APScheduler removed the one-shot when it submitted the run. A
        # recurring job that's gone was deleted, and stays deleted.
        if not isinstance(run.job_state["trigger"], DateTrigger):
            return
        job = Job.__new__(Job)
        job.__setstate__({**run.job_state, "next_run_time": run.run_time})
        store.add_job(job)
    elif current.next_run_time is not None and current.next_run_time > run.run_time:
        current._modify(next_run_time=run.run_time)
        store.update_job(current)
    else:
        return
    logger.warning(f"Handed run of job {run.job_id} at {run.run_time} back")
