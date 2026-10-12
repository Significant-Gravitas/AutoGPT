"""Stop the scheduler on SIGTERM without losing, repeating or hanging on jobs.

Stock APScheduler gets three things wrong here:

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
"""

import asyncio
import logging
import os
import threading
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import STATE_STOPPED

logger = logging.getLogger(__name__)

# After cancelling job coroutines, how long workers get to unwind.
_CANCEL_GRACE_S = 5.0


class DrainingBackgroundScheduler(BackgroundScheduler):
    """A BackgroundScheduler that can be drained before it is shut down.

    Draining and the drained check both happen under the jobstore lock (an
    RLock, so the base pass can take it again). A pass therefore either runs
    to completion first, removing its one-shot jobs while the scheduler is
    still running, or sees the drain and leaves every due job untouched.
    Untouched jobs keep their next run time, and the next scheduler to start
    runs them: ``coalesce=True`` makes that one run per job however many were
    missed, and ``misfire_grace_time=None`` means none is dropped as too late.
    """

    _draining = False

    def drain(self, timeout: float) -> bool:
        """Stop job passes; returns whether no pass was left in progress.

        Waits up to ``timeout`` for a pass in progress to finish. A pass stuck
        on a slow jobstore must not hold shutdown past its grace period, so
        the flag is set either way; a pass that outlives the wait is the one
        case that can still race the stop.
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


def stop_scheduler(
    scheduler: DrainingBackgroundScheduler,
    job_loop: asyncio.AbstractEventLoop | None,
    grace_seconds: float,
) -> bool:
    """Drain, then give running jobs ``grace_seconds`` to finish.

    Jobs still running after that have their coroutines on ``job_loop``
    cancelled, which unblocks the worker waiting on each. Returns whether every
    worker finished; if not, the caller must not wait for the process to exit
    on its own (see :func:`exit_if_jobs_hung`).
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
        "period; cancelling them"
    )
    if job_loop is not None and not job_loop.is_closed():
        job_loop.call_soon_threadsafe(_cancel_all_tasks, job_loop)
    if finished.wait(_CANCEL_GRACE_S):
        return True
    logger.error("Scheduler jobs did not stop after being cancelled")
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
    for task in asyncio.all_tasks(loop):
        task.cancel()
