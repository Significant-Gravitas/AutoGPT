from apscheduler.schedulers.background import BackgroundScheduler


class StopSafeBackgroundScheduler(BackgroundScheduler):
    """A BackgroundScheduler whose stop never meets a job pass half-way, and
    never waits on a running job that calls back into it.

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

    Stock ``shutdown`` also waits for running jobs holding the jobstore lock,
    so a running job that calls into the scheduler (the schedule index
    reconcile's ``get_jobs``) waits on that lock for good, and the stop on the
    job. Here the stop waits for them first, holding no lock and with the
    jobstores still open, so their calls go through; a pass that their
    writes wake finds the mark and touches no job. APScheduler's own stop
    runs after them.

    Overrides APScheduler's private ``_process_jobs``, and takes its
    ``_jobstores_lock`` and reads its ``_executors`` and their ``_instances``,
    as of APScheduler 3.11. ``scheduler_shutdown_test`` fails if an upgrade
    renames any of them; one that changes how they work needs this class
    rechecked.
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
        if wait and self.running:
            # Nothing adds an executor during a stop, so the copy takes no lock.
            executors = list(self._executors.values())
            running = sum(sum(executor._instances.values()) for executor in executors)
            self._logger.info(f"Waiting for {running} running job(s) to finish")
            for executor in executors:
                executor.shutdown(wait=True)
        super().shutdown(wait)

    def _process_jobs(self) -> float | None:
        # Held through the pass's own event dispatch too, so a listener must be
        # quick and must not take the executors lock: ``shutdown`` takes that
        # one first.
        with self._jobstores_lock:
            if self._stopping:
                return None
            return super()._process_jobs()
