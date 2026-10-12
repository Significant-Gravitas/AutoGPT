"""Stop the scheduler on SIGTERM without losing, repeating or hanging on runs.

Stock APScheduler gets this wrong in several ways:

- ``shutdown`` sets the state to stopped *outside* the jobstore lock. A job
  pass that is mid-way, holding that lock, then finds the scheduler stopped
  when it removes a one-shot job it has just submitted. ``remove_job`` only
  searches the pending list once stopped, so the row stays due and the next
  scheduler runs the job a second time.
- ``shutdown(wait=True)`` holds the lock while it waits for running jobs. A
  pass that woke during that wait carries on afterwards without re-checking
  the state: it submits to the pool just shut down ("cannot schedule new
  futures after shutdown", AUTOGPT-SERVER-8A8) and still advances the job's
  next run time, so that run is skipped.
- The wait is unbounded, and its worker threads keep the process alive after
  it returns: our jobs block in ``future.result()`` on the scheduler's event
  loop for up to the 300s operation timeout.
- Cutting a run short loses it: APScheduler advanced the job when it
  submitted the run, before the run did anything (see ``scheduler_runs``).
"""

import asyncio
import logging
import os
import threading
import time

from apscheduler.events import EVENT_SCHEDULER_SHUTDOWN, SchedulerEvent
from apscheduler.schedulers import SchedulerNotRunningError
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import STATE_STOPPED

from backend.executor.scheduler_runs import RunLedger

logger = logging.getLogger(__name__)

# After the grace period: how long a commit in progress gets to finish, how
# long handing runs back may wait for the jobstore, and how long workers get
# to unwind once their coroutines are cancelled.
_COMMIT_WAIT_S = 5.0
_GIVE_BACK_WAIT_S = 5.0
_CANCEL_GRACE_S = 5.0


class DrainingBackgroundScheduler(BackgroundScheduler):
    """A BackgroundScheduler that only ever stops under its jobstore lock.

    Draining, stopping and the check every job pass makes all happen under the
    jobstore lock (an RLock, so the base pass can take it again). A pass
    therefore either runs to completion first, removing its one-shot jobs
    while the scheduler is still running, or sees the drain and leaves every
    due job untouched. Untouched jobs keep their next run time, and the next
    scheduler to start runs them: ``coalesce=True`` makes that one run per job
    however many were missed, and ``misfire_grace_time=None`` means none is
    dropped as too late.
    """

    _draining = False

    def __init__(self, *args, ledger: RunLedger | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ledger = ledger

    def drain(self, timeout: float) -> bool:
        """Stop job passes; returns whether no pass was left in progress.

        Waits up to ``timeout`` for a pass in progress to finish, and sets the
        flag either way. Stopping still waits for that pass (see ``shutdown``).
        """
        finished = self._jobstores_lock.acquire(timeout=timeout)
        self._draining = True
        if finished:
            self._jobstores_lock.release()
        return finished

    def _process_jobs(self) -> float | None:
        with self._jobstores_lock:
            if self._draining or self.state == STATE_STOPPED:
                return None
            return super()._process_jobs()

    def shutdown(self, wait: bool = True) -> None:
        """Stop under the jobstore lock, then wait for running jobs outside it.

        Replaces the stock sequence (Base, Blocking and Background schedulers'
        ``shutdown``), which flips the state outside the lock and then holds
        the lock for the whole wait.
        """
        with self._jobstores_lock:
            if self.state == STATE_STOPPED:
                raise SchedulerNotRunningError
            self._draining = True
            self.state = STATE_STOPPED
        with self._executors_lock:
            for executor in self._executors.values():
                executor.shutdown(wait)
        with self._jobstores_lock:
            for jobstore in self._jobstores.values():
                jobstore.shutdown()
        self._logger.info("Scheduler has been shut down")
        self._dispatch_event(SchedulerEvent(EVENT_SCHEDULER_SHUTDOWN))
        self._event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None


def stop_scheduler(
    scheduler: DrainingBackgroundScheduler,
    job_loop: asyncio.AbstractEventLoop | None,
    grace_seconds: float,
) -> bool:
    """Drain, then give running jobs ``grace_seconds`` to finish.

    One budget covers both the drain and the wait for running jobs. Runs still
    going after it are handed back to the jobstore if they haven't taken
    effect, then their coroutines on ``job_loop`` are cancelled, which unblocks
    the worker waiting on each. Returns whether every worker finished; if not,
    the caller must not wait for the process to exit on its own (see
    :func:`exit_if_jobs_hung`).
    """
    deadline = time.monotonic() + grace_seconds
    if not scheduler.drain(timeout=grace_seconds):
        logger.warning("A job pass was still running when the grace period ended")
    finished = threading.Event()

    def _shutdown() -> None:
        try:
            scheduler.shutdown(wait=True)
        except Exception:
            logger.exception("Scheduler shutdown failed")
        finally:
            finished.set()

    threading.Thread(target=_shutdown, name="SchedulerShutdown", daemon=True).start()
    if finished.wait(max(0.0, deadline - time.monotonic())):
        logger.info("Scheduler stopped; running jobs finished")
        return True

    logger.warning(
        f"Scheduler jobs still running after the {grace_seconds}s shutdown grace "
        "period; handing back the ones that haven't taken effect, then cancelling"
    )
    if scheduler.ledger is not None and not _hand_back(scheduler.ledger, scheduler):
        # Cancelling now would lose these runs; let the caller exit with them
        # still running instead.
        return False
    if job_loop is not None and not job_loop.is_closed():

        job_loop.call_soon_threadsafe(_cancel_all_tasks, job_loop)
    if finished.wait(_CANCEL_GRACE_S):
        return True
    logger.error("Scheduler jobs did not stop after being cancelled")
    return False


def _hand_back(ledger: RunLedger, scheduler: DrainingBackgroundScheduler) -> bool:
    try:
        ledger.close(commit_wait=_COMMIT_WAIT_S)
        # Always, even with nothing abandoned yet: taking the jobstore lock
        # waits out a pass still in progress, and any run it submits from now
        # on is abandoned too.
        return ledger.give_back(scheduler, _GIVE_BACK_WAIT_S)
    except Exception:
        logger.exception("Could not hand abandoned runs back")
        return False


def exit_if_jobs_hung(stopped: bool) -> None:
    """End the process now if job threads are still running.

    Pool workers are non-daemon threads, so the interpreter would otherwise
    wait for them at exit, past any grace period, until Kubernetes kills the
    pod.
    """
    if stopped:
        return
    logger.error("Exiting with scheduler jobs still running")
    logging.shutdown()
    os._exit(1)


def _cancel_all_tasks(loop: asyncio.AbstractEventLoop) -> None:
    # Safe because cleanup runs after the RPC server has stopped: by now the
    # only coroutines left on the scheduler's loop are job dispatches.
    for task in asyncio.all_tasks(loop):
        task.cancel()
