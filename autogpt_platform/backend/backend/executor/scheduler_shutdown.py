"""Shut the scheduler down without losing the runs that fall due meanwhile.

APScheduler's ``shutdown(wait=True)`` sets the state to stopped, then holds
the jobstore lock while it waits for running jobs and shuts the executor pool.
A job-processing pass that woke up during that wait is blocked on the same
lock. When it gets the lock it carries on without re-checking the state: it
submits to the pool that was just shut down ("RuntimeError: cannot schedule new
futures after shutdown", AUTOGPT-SERVER-8A8), then still advances the job's next
run time in the jobstore, so that run is lost for good.
"""

import logging
import threading
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.base import STATE_STOPPED

logger = logging.getLogger(__name__)


class DrainingBackgroundScheduler(BackgroundScheduler):
    """A BackgroundScheduler whose job passes stop once shutdown begins.

    The state check runs under the jobstore lock (an RLock, so the base pass
    can take it again). ``shutdown`` flips the state before it takes that lock,
    so a pass either finishes submitting before the pool is shut, or sees the
    stopped state and leaves every due job untouched in the jobstore. Those jobs
    keep their next run time, and the next scheduler to start runs them:
    ``coalesce=True`` makes it one run per job however many were missed, and
    ``misfire_grace_time=None`` means none is dropped for being late.
    """

    def _process_jobs(self) -> float | None:
        with self._jobstores_lock:
            if self.state == STATE_STOPPED:
                return None
            return super()._process_jobs()


def shutdown_scheduler(scheduler: BackgroundScheduler, grace_seconds: float) -> bool:
    """Stop scheduling, then wait up to ``grace_seconds`` for running jobs.

    Returns whether every running job finished in time. Waiting is bounded so a
    stuck job can't hold the pod until Kubernetes kills it; a job abandoned
    here already had its next run time advanced when it started.
    """
    finished = threading.Event()
    started_at = time.monotonic()

    def _shutdown() -> None:
        try:
            scheduler.shutdown(wait=True)
        except Exception:
            logger.exception("Scheduler shutdown failed")
        finally:
            finished.set()

    threading.Thread(target=_shutdown, name="SchedulerShutdown", daemon=True).start()
    if finished.wait(grace_seconds):
        elapsed = time.monotonic() - started_at
        logger.info(f"Scheduler stopped; running jobs finished in {elapsed:.1f}s")
        return True
    logger.warning(
        f"Scheduler jobs still running after the {grace_seconds}s shutdown grace "
        "period; exiting without them"
    )
    return False
