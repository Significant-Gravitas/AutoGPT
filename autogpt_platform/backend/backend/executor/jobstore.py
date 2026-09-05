import logging

from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from sqlalchemy import and_, select

logger = logging.getLogger(__name__)


class ResilientSQLAlchemyJobStore(SQLAlchemyJobStore):
    """Parks jobs it cannot restore instead of deleting them.

    Upstream ``_get_jobs`` DELETEs any row whose ``job_state`` fails to load,
    so a deploy that renames or removes a symbol a persisted job references
    destroys the user's schedule with no recovery path — the table carries no
    audit trail. Parking clears ``next_run_time`` (APScheduler's own "paused"
    marker), which keeps ``job_state`` intact for repair and stops
    ``get_due_jobs`` returning the row, so one bad job cannot spin the
    scheduler.

    Parked jobs do not run. ``get_parked_job_ids`` is the operator surface
    that keeps that from being silent.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parked_ids: set[str] = set()

    def _get_jobs(self, *conditions):
        jobs = []
        selectable = select(self.jobs_t.c.id, self.jobs_t.c.job_state).order_by(
            self.jobs_t.c.next_run_time
        )
        selectable = selectable.where(and_(*conditions)) if conditions else selectable
        unrestorable: set[str] = set()

        with self.engine.begin() as connection:
            for row in connection.execute(selectable):
                try:
                    jobs.append(self._reconstitute_job(row.job_state))
                except BaseException:
                    unrestorable.add(row.id)
                    # Unconditioned _get_jobs still returns parked rows, so
                    # without this every get_jobs() call re-reports them.
                    if row.id not in self._parked_ids:
                        self._parked_ids.add(row.id)
                        self._logger.exception(
                            'Unable to restore job "%s" -- parking it. The row is '
                            "kept; repair it and set next_run_time to resume.",
                            row.id,
                        )

            if unrestorable:
                connection.execute(
                    self.jobs_t.update()
                    .where(
                        and_(
                            self.jobs_t.c.id.in_(unrestorable),
                            self.jobs_t.c.next_run_time.is_not(None),
                        )
                    )
                    .values(next_run_time=None)
                )

        return jobs

    def get_parked_job_ids(self) -> list[str]:
        """Ids of rows that are paused *and* still unrestorable.

        A user-paused job also has ``next_run_time = NULL`` but restores
        fine, so attempting the restore is what separates the two.
        """
        selectable = select(self.jobs_t.c.id, self.jobs_t.c.job_state).where(
            self.jobs_t.c.next_run_time.is_(None)
        )
        parked = []
        with self.engine.begin() as connection:
            for row in connection.execute(selectable):
                try:
                    self._reconstitute_job(row.job_state)
                except BaseException:
                    parked.append(row.id)
        return parked
