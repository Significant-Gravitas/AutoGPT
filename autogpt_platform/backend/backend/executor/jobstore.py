import logging
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from apscheduler.jobstores.base import ConflictingIdError, JobLookupError
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.util import datetime_to_utc_timestamp
from sqlalchemy import Connection, and_, select, text
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

# Seconds a job write in this context may wait for a row another transaction
# holds, or run, on PostgreSQL: set around its own writes by a caller that
# must not be held by one (the memory cron reconcile,
# ``executor/scheduler.py``). The scheduler's other writes leave it unset.
WRITE_TIMEOUT: ContextVar[float | None] = ContextVar(
    "jobstore_write_timeout", default=None
)


class ResilientSQLAlchemyJobStore(SQLAlchemyJobStore):
    """Parks jobs it cannot restore instead of deleting them.

    Upstream ``_get_jobs`` DELETEs any row whose ``job_state`` fails to load,
    so a deploy that renames or removes a symbol a persisted job references
    destroys the user's schedule with no recovery path — the table carries no
    audit trail. Parking clears ``next_run_time`` (APScheduler's own "paused"
    marker), which keeps ``job_state`` intact for repair and stops
    ``get_due_jobs`` returning the row, so one bad job cannot spin the
    scheduler.

    A listing that can restore a parked job resumes its serialized schedule.
    Explicitly paused jobs have no serialized next run and stay paused.
    ``get_parked_job_ids`` reports jobs that still cannot be restored.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parked_ids: set[str] = set()

    def _get_jobs(self, *conditions, statement_timeout: float | None = None):
        """Upstream's read, parking what it cannot restore. On PostgreSQL a
        ``statement_timeout`` (seconds) bounds this read alone, in its own
        transaction, for a caller that must not wait on a stalled query: the
        engine's other queries keep their limits."""
        jobs = []
        selectable = select(
            self.jobs_t.c.id, self.jobs_t.c.job_state, self.jobs_t.c.next_run_time
        ).order_by(self.jobs_t.c.next_run_time)
        selectable = selectable.where(and_(*conditions)) if conditions else selectable
        unrestorable: list = []
        restored: list = []

        with self.engine.begin() as connection:
            if statement_timeout is not None:
                _limit(connection, statement_timeout, "statement_timeout")
            for row in connection.execute(selectable):
                try:
                    job = self._reconstitute_job(row.job_state)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as e:
                    unrestorable.append((row, e))
                else:
                    jobs.append(job)
                    if row.next_run_time is None and job.next_run_time is not None:
                        restored.append((row, job.next_run_time))

            for row, next_run_time in restored:
                if self._unpark(connection, row, next_run_time):
                    self._parked_ids.discard(row.id)
                    self._logger.info('Resuming restored job "%s"', row.id)

            for row, error in unrestorable:
                if not self._park(connection, row):
                    continue
                # Unconditioned _get_jobs still returns parked rows, so
                # without this every get_jobs() call re-reports them.
                if row.id not in self._parked_ids:
                    self._parked_ids.add(row.id)
                    self._logger.error(
                        'Unable to restore job "%s" -- parking it. The row is '
                        "kept; repair it and set next_run_time to resume.",
                        row.id,
                        exc_info=error,
                    )

        return jobs

    def add_job(self, job):
        """Upstream's insert, bounded by ``WRITE_TIMEOUT`` when set."""
        insert = self.jobs_t.insert().values(
            id=job.id,
            next_run_time=datetime_to_utc_timestamp(job.next_run_time),
            job_state=pickle.dumps(job.__getstate__(), self.pickle_protocol),
        )
        with self._write() as connection:
            try:
                connection.execute(insert)
            except IntegrityError:
                raise ConflictingIdError(job.id) from None

    def update_job(self, job):
        """Upstream's update, bounded by ``WRITE_TIMEOUT`` when set."""
        update = (
            self.jobs_t.update()
            .values(
                next_run_time=datetime_to_utc_timestamp(job.next_run_time),
                job_state=pickle.dumps(job.__getstate__(), self.pickle_protocol),
            )
            .where(self.jobs_t.c.id == job.id)
        )
        with self._write() as connection:
            if connection.execute(update).rowcount == 0:
                raise JobLookupError(job.id)

    def remove_job(self, job_id):
        """Upstream's delete, bounded by ``WRITE_TIMEOUT`` when set."""
        delete = self.jobs_t.delete().where(self.jobs_t.c.id == job_id)
        with self._write() as connection:
            if connection.execute(delete).rowcount == 0:
                raise JobLookupError(job_id)

    @contextmanager
    def _write(self) -> Iterator[Connection]:
        """A write's transaction, its lock waits and statements limited to
        ``WRITE_TIMEOUT`` when this context sets it: the write then fails
        instead of holding its thread, and APScheduler's lock, on a row
        another transaction keeps."""
        with self.engine.begin() as connection:
            timeout = WRITE_TIMEOUT.get()
            if timeout is not None:
                _limit(connection, timeout, "lock_timeout", "statement_timeout")
            yield connection

    def _park(self, connection, row) -> bool:
        """Clear ``next_run_time`` for the row we read, and say whether we did.

        Matching the scanned ``job_state`` too keeps a repair or resume that
        landed since the scan from being silently undone.
        """
        result = connection.execute(
            self.jobs_t.update()
            .where(
                and_(
                    self.jobs_t.c.id == row.id,
                    self.jobs_t.c.job_state == row.job_state,
                    self.jobs_t.c.next_run_time.is_not(None),
                )
            )
            .values(next_run_time=None)
        )
        return result.rowcount == 1

    def _unpark(self, connection, row, next_run_time) -> bool:
        result = connection.execute(
            self.jobs_t.update()
            .where(
                and_(
                    self.jobs_t.c.id == row.id,
                    self.jobs_t.c.job_state == row.job_state,
                    self.jobs_t.c.next_run_time.is_(None),
                )
            )
            .values(next_run_time=datetime_to_utc_timestamp(next_run_time))
        )
        return result.rowcount == 1

    def get_parked_job_ids(self, limit: int | None = None) -> list[str]:
        """Ids of rows that are paused *and* still unrestorable.

        A user-paused job also has ``next_run_time = NULL`` but restores
        fine, so attempting the restore is what separates the two. *limit*
        caps the rows scanned: paused and fired-once rows accumulate
        forever, and startup must not pay for the whole backlog.
        """
        selectable = select(self.jobs_t.c.id, self.jobs_t.c.job_state).where(
            self.jobs_t.c.next_run_time.is_(None)
        )
        if limit is not None:
            selectable = selectable.limit(limit)
        parked = []
        with self.engine.begin() as connection:
            for row in connection.execute(selectable):
                try:
                    self._reconstitute_job(row.job_state)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException:
                    parked.append(row.id)
        return parked

    def reconcile_repaired_jobs(self, batch_size: int = 500) -> list[str]:
        """Finish parking rows whose payload has since been repaired.

        Parking can only clear the ``next_run_time`` COLUMN: the row is by
        definition one we could not deserialize, so the copy inside
        ``job_state`` keeps its old value. Once the payload is repaired the
        two disagree, and the row is stranded — it deserializes, so it is no
        longer reported as parked; the column still reads paused, so
        ``get_due_jobs`` skips it; and ``resume`` reads the stale non-None
        copy and refuses. Writing that copy back to None makes it an
        ordinary paused job, which resume revives.

        Listings automatically resume restorable jobs. This explicit operator
        path instead normalizes them to paused before a listing resumes them;
        ``jobstore_backfill`` also pauses rows it rewrites. It is not run at
        startup. It walks the whole paused set in id order, a batch per
        transaction: a cap alone would keep re-reading the first page and
        never reach a repaired row behind a backlog nothing deletes, while one
        transaction over the lot would hold locks across every row it writes.
        """
        healed: list[str] = []
        after = ""
        while True:
            selectable = (
                select(self.jobs_t.c.id, self.jobs_t.c.job_state)
                .where(
                    and_(
                        self.jobs_t.c.next_run_time.is_(None),
                        self.jobs_t.c.id > after,
                    )
                )
                .order_by(self.jobs_t.c.id)
                .limit(batch_size)
            )
            rows = self._reconcile_batch(selectable, healed)
            if len(rows) < batch_size:
                return healed
            after = rows[-1]

    def _reconcile_batch(self, selectable, healed: list[str]) -> list[str]:
        """One batch in one transaction; returns the ids it read, in order."""
        seen = []
        with self.engine.begin() as connection:
            for row in connection.execute(selectable):
                seen.append(row.id)
                try:
                    job = self._reconstitute_job(row.job_state)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException:
                    continue  # still unrestorable: genuinely parked
                if job.next_run_time is None:
                    continue  # an ordinary paused job; nothing to reconcile
                job.next_run_time = None
                result = connection.execute(
                    self.jobs_t.update()
                    .where(
                        and_(
                            self.jobs_t.c.id == row.id,
                            self.jobs_t.c.job_state == row.job_state,
                            self.jobs_t.c.next_run_time.is_(None),
                        )
                    )
                    .values(
                        job_state=pickle.dumps(job.__getstate__(), self.pickle_protocol)
                    )
                )
                if result.rowcount != 1:
                    continue  # resumed or repaired under the scan; leave it be
                self._parked_ids.discard(row.id)
                healed.append(row.id)
        return seen


def _limit(connection: Connection, seconds: float, *settings: str) -> None:
    """Limit ``settings`` for the rest of ``connection``'s transaction, on
    PostgreSQL; other databases have no such settings."""
    if connection.dialect.name != "postgresql":
        return
    for setting in settings:
        connection.execute(
            text("SELECT set_config(:setting, :ms, true)"),
            {"setting": setting, "ms": str(int(seconds * 1000))},
        )
