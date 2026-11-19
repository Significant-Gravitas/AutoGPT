from apscheduler.schedulers.background import BackgroundScheduler


class StopSafeBackgroundScheduler(BackgroundScheduler):
    """A BackgroundScheduler whose stop never meets a job pass half-way.

    Stock ``shutdown`` marks the scheduler stopped outside the jobstore lock,
    and job passes never check for it. A pass that gets the lock after the
    pool is shut down (its thread woke for a job due during the stop, or for
    the stop's own last wake-up) hands every due job to the dead pool
    ("cannot schedule new futures after shutdown", AUTOGPT-SERVER-8A8) and
    still moves the job's next run on, so the run is lost. A stop that lands
    in the middle of a pass makes the pass's removal of a one-shot it has just
    handed over look only at pending jobs: the row stays, and the next
    scheduler runs the one-shot again.

    Here the stop is marked under the jobstore lock, which waits out a pass
    in progress, and every pass checks the mark under that lock. A pass
    either finishes before the stop, its one-shots removed, or touches no job.
    The next scheduler runs each job left due once (``coalesce``, and no
    misfire grace).

    Overrides APScheduler's private ``_process_jobs`` and takes its
    ``_jobstores_lock``, as of APScheduler 3.11. ``scheduler_shutdown_test``
    fails if an upgrade renames either; one that changes how they work needs
    this class rechecked.
    """

    _stopping = False

    def start(self, *args, **kwargs) -> None:
        # A start while running raises, and must leave a stop's mark alone.
        if not self.running:
            self._stopping = False
        super().start(*args, **kwargs)

    def shutdown(self, wait: bool = True) -> None:
        with self._jobstores_lock:
            self._stopping = True
        super().shutdown(wait)

    def _process_jobs(self) -> float | None:
        # Held through the pass's own event dispatch too, so a listener must be
        # quick and must not take the executors lock: ``shutdown`` takes that
        # one first.
        with self._jobstores_lock:
            if self._stopping:
                return None
            return super()._process_jobs()
