import os
import pickle
import tempfile
from contextlib import contextmanager
from enum import Enum
from unittest.mock import MagicMock, patch

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select

from backend.executor.jobstore import ResilientSQLAlchemyJobStore
from backend.executor.jobstore_backfill import _strip_enums
from backend.executor.scheduler import Jobstores, Scheduler

_SCHEDULER_PATH = "backend.executor.scheduler"


class _RemovedEnum(Enum):
    GONE = "gone"


# Pickled while GONE exists, then the class is rebound without it — the exact
# shape of SENTRY-1392, where a deploy dropped a NotificationType member.
POISONED_STATE = pickle.dumps(
    {"kwargs": {"x": _RemovedEnum.GONE}}, pickle.HIGHEST_PROTOCOL
)


class _RemovedEnum(Enum):  # noqa: F811 - deliberate rebind, drops GONE
    KEPT = "kept"


def noop():
    pass


def test_poison_reproduces_the_production_error():
    """The poison must fail the way production did — on the missing enum member,
    not merely as invalid bytes. Which exception carries that is the
    interpreter's business: 3.11.2 reduces a plain Enum member by NAME and
    raises AttributeError, every later build reduces by value and raises
    ValueError, so pinning one wording fails on the other."""
    with pytest.raises((ValueError, AttributeError)) as caught:
        pickle.loads(POISONED_STATE)

    assert "GONE" in str(caught.value) or "gone" in str(caught.value)


def test_unrestorable_job_is_parked_not_deleted():
    with _store() as (store, scheduler):
        scheduler.add_job(noop, "interval", seconds=3600, id="good")
        scheduler.add_job(noop, "interval", seconds=3600, id="poisoned")
        _poison(store, "poisoned")

        jobs = store._get_jobs()

        assert [j.id for j in jobs] == ["good"]
        # The row survives — this is the whole point of the fix.
        assert _ids(store) == {"good", "poisoned"}
        assert _next_run_time(store, "poisoned") is None
        assert _next_run_time(store, "good") is not None


def test_active_jobs_read_parks_the_row_then_stops_unpickling_it():
    """The ``next_run_time IS NOT NULL`` listing path meets a poisoned row
    first, because parking is what makes that column NULL."""
    with _store() as (store, scheduler):
        scheduler.add_job(noop, "interval", seconds=3600, id="good")
        scheduler.add_job(noop, "interval", seconds=3600, id="poisoned")
        _poison(store, "poisoned")
        active = store.jobs_t.c.next_run_time.isnot(None)

        assert [j.id for j in store._get_jobs(active)] == ["good"]
        assert _ids(store) == {"good", "poisoned"}
        assert _next_run_time(store, "poisoned") is None

        with patch.object(
            store, "_reconstitute_job", wraps=store._reconstitute_job
        ) as restore:
            assert [j.id for j in store._get_jobs(active)] == ["good"]
        # The filter now excludes the parked row in SQL, so the second read
        # never fetches it — one restore, for "good".
        assert restore.call_count == 1


def test_scheduler_listing_path_parks_an_unrestorable_schedule():
    """The seam between the two fixes: the listing read is filtered at the SQL
    level (#14439) and must still park rather than delete (#14218), so the
    jobstore behind it has to be the resilient one."""
    with _store() as (store, scheduler):
        scheduler.add_job(noop, "interval", seconds=3600, id="poisoned")
        _poison(store, "poisoned")

        with _scheduler_wired_to(store) as sched:
            assert (
                sched._execution_jobstore
                is sched._persistent_jobstores[Jobstores.EXECUTION.value]
            )
            assert sched.get_execution_schedules() == []

        assert _ids(store) == {"poisoned"}
        assert _next_run_time(store, "poisoned") is None
        # Parked by that very read, through this instance — not a second one
        # the operator surface would be reporting on separately.
        assert store._parked_ids == {"poisoned"}
        assert sched.get_parked_jobs()[Jobstores.EXECUTION.value] == ["poisoned"]


def test_repaired_job_can_be_resumed_and_then_fires():
    """Park, repair, resume, run. Parking can only clear the COLUMN, so without
    reconciliation the repaired row is stranded: it reports healthy, never comes
    due, and resume refuses it because the pickled copy is still set."""
    with _store() as (store, scheduler):
        scheduler.add_job(noop, "interval", seconds=3600, id="poisoned")
        healthy_state = _job_state(store, "poisoned")
        _poison(store, "poisoned")
        store._get_jobs()
        assert store.get_parked_job_ids() == ["poisoned"]

        # Repair rewrites job_state and leaves its next_run_time untouched,
        # which is exactly what jobstore_backfill does.
        _set_job_state(store, "poisoned", healthy_state)
        assert store.get_parked_job_ids() == []
        assert store._get_jobs()[0].next_run_time is not None  # stranded
        assert _next_run_time(store, "poisoned") is None

        assert store.reconcile_repaired_jobs() == ["poisoned"]

        # Now an ordinary paused job: both copies agree, so resume revives it.
        assert store._get_jobs()[0].next_run_time is None
        scheduler.resume_job("poisoned")
        assert _next_run_time(store, "poisoned") is not None
        assert [j.id for j in store.get_due_jobs(_utc(2**31 - 1))] == ["poisoned"]


def test_reconcile_leaves_a_user_paused_job_alone():
    with _store() as (store, scheduler):
        scheduler.add_job(noop, "interval", seconds=3600, id="paused")
        scheduler.pause_job("paused")

        assert store.reconcile_repaired_jobs() == []
        assert _next_run_time(store, "paused") is None


def test_reconcile_leaves_a_still_broken_job_parked():
    with _store() as (store, scheduler):
        scheduler.add_job(noop, "interval", seconds=3600, id="poisoned")
        _poison(store, "poisoned")
        store._get_jobs()

        assert store.reconcile_repaired_jobs() == []
        assert store.get_parked_job_ids() == ["poisoned"]


def test_reconcile_walks_past_its_batch_size():
    """Batching bounds the transaction, not the work: every stranded row must
    still be reached, which a plain cap would not do."""
    with _store() as (store, scheduler):
        for i in range(7):
            scheduler.add_job(noop, "interval", seconds=3600, id=f"j{i}")
            healthy = _job_state(store, f"j{i}")
            _poison(store, f"j{i}")
            store._get_jobs()
            _set_job_state(store, f"j{i}", healthy)

        healed = store.reconcile_repaired_jobs(batch_size=2)

        assert sorted(healed) == [f"j{i}" for i in range(7)]
        assert all(_next_run_time(store, f"j{i}") is None for i in range(7))
        assert store.reconcile_repaired_jobs(batch_size=2) == []


def test_a_row_repaired_under_the_scan_is_not_parked():
    """Parking must not undo a repair that landed after the row was read."""
    with _store() as (store, scheduler):
        scheduler.add_job(noop, "interval", seconds=3600, id="poisoned")
        healthy_state = _job_state(store, "poisoned")
        _poison(store, "poisoned")

        def _repair_midway(job_state):
            _set_job_state(store, "poisoned", healthy_state)
            raise ValueError("still poisoned as far as this scan knows")

        with patch.object(store, "_reconstitute_job", side_effect=_repair_midway):
            store._get_jobs()

        # The repair survives: the row is runnable, not re-parked.
        assert _next_run_time(store, "poisoned") is not None
        assert store.get_parked_job_ids() == []


def test_reconcile_leaves_a_row_that_changed_under_the_scan_alone():
    """A resume landing mid-reconcile must not be reverted to paused."""
    with _store() as (store, scheduler):
        scheduler.add_job(noop, "interval", seconds=3600, id="poisoned")
        healthy_state = _job_state(store, "poisoned")
        _poison(store, "poisoned")
        store._get_jobs()
        _set_job_state(store, "poisoned", healthy_state)

        real = store._reconstitute_job

        def _resume_midway(job_state):
            with store.engine.begin() as conn:
                conn.execute(
                    store.jobs_t.update()
                    .where(store.jobs_t.c.id == "poisoned")
                    .values(next_run_time=1.0)
                )
            return real(job_state)

        with patch.object(store, "_reconstitute_job", side_effect=_resume_midway):
            assert store.reconcile_repaired_jobs() == []

        assert _next_run_time(store, "poisoned") == 1.0


def test_parked_scan_limit_bounds_the_rows_read():
    with _store() as (store, scheduler):
        for i in range(5):
            scheduler.add_job(noop, "interval", seconds=3600, id=f"j{i}")
            _poison(store, f"j{i}")
        store._get_jobs()

        assert len(store.get_parked_job_ids()) == 5
        assert len(store.get_parked_job_ids(limit=2)) == 2


@pytest.mark.parametrize("filtered", [False, True], ids=["all", "active_only"])
def test_upstream_jobstore_would_have_deleted_the_row(filtered: bool):
    """Control: pins the upstream behaviour this subclass exists to prevent, on
    both reads — including the active-only one the listing path now issues."""
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

    with _store(cls=SQLAlchemyJobStore) as (store, scheduler):
        scheduler.add_job(noop, "interval", seconds=3600, id="poisoned")
        _poison(store, "poisoned")

        conditions = (store.jobs_t.c.next_run_time.isnot(None),) if filtered else ()
        store._get_jobs(*conditions)

        assert _ids(store) == set()


def test_parked_job_is_never_returned_as_due():
    with _store() as (store, scheduler):
        scheduler.add_job(noop, "interval", seconds=1, id="poisoned")
        _poison(store, "poisoned")
        store._get_jobs()

        far_future = _utc(2**31 - 1)
        assert store.get_due_jobs(far_future) == []
        assert _ids(store) == {"poisoned"}


def test_get_parked_job_ids_ignores_a_healthy_paused_job():
    with _store() as (store, scheduler):
        scheduler.add_job(noop, "interval", seconds=3600, id="paused")
        scheduler.add_job(noop, "interval", seconds=3600, id="poisoned")
        scheduler.pause_job("paused")
        _poison(store, "poisoned")
        store._get_jobs()

        # A paused job also has next_run_time NULL, so only the restore
        # attempt separates it from a parked one.
        assert store.get_parked_job_ids() == ["poisoned"]


def test_repeated_loads_report_each_bad_job_once():
    with _store() as (store, scheduler):
        scheduler.add_job(noop, "interval", seconds=3600, id="poisoned")
        _poison(store, "poisoned")

        store._get_jobs()
        store._get_jobs()

        assert store._parked_ids == {"poisoned"}


def test_strip_enums_recurses_through_containers():
    class Provider(str, Enum):
        GITHUB = "github"

    stripped = _strip_enums(
        {"creds": {"a": {"provider": Provider.GITHUB}}, "list": [Provider.GITHUB]}
    )

    assert stripped == {"creds": {"a": {"provider": "github"}}, "list": ["github"]}
    assert not isinstance(stripped["list"][0], Enum)


def test_strip_enums_leaves_plain_values_alone():
    payload = {"user_id": "u1", "graph_version": 3, "cron": "0 * * * *"}
    assert _strip_enums(payload) == payload


@contextmanager
def _store(cls=ResilientSQLAlchemyJobStore):
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = cls(url=f"sqlite:///{path}")
    scheduler = BackgroundScheduler()
    scheduler.add_jobstore(store, "default")
    scheduler.start(paused=True)
    try:
        yield store, scheduler
    finally:
        scheduler.shutdown(wait=False)
        os.unlink(path)


@contextmanager
def _scheduler_wired_to(store):
    """Run ``Scheduler.run_service`` far enough to wire *store* in as the
    EXECUTION jobstore, with every other dependency stubbed out."""
    spare = MagicMock(get_parked_job_ids=MagicMock(return_value=[]))
    with (
        patch(f"{_SCHEDULER_PATH}.BackgroundScheduler", return_value=MagicMock()),
        patch(f"{_SCHEDULER_PATH}.load_dotenv"),
        patch(f"{_SCHEDULER_PATH}._init_feature_flags_for_scheduler"),
        patch(f"{_SCHEDULER_PATH}.asyncio.new_event_loop", return_value=MagicMock()),
        patch(f"{_SCHEDULER_PATH}.threading.Thread", return_value=MagicMock()),
        patch(f"{_SCHEDULER_PATH}.create_engine", return_value=MagicMock()),
        patch(
            f"{_SCHEDULER_PATH}.ResilientSQLAlchemyJobStore",
            side_effect=[store, spare],
        ),
        patch(f"{_SCHEDULER_PATH}.MemoryJobStore", return_value=MagicMock()),
        patch(
            f"{_SCHEDULER_PATH}._extract_schema_from_url",
            return_value=("public", "sqlite://"),
        ),
        patch(f"{_SCHEDULER_PATH}.ensure_embeddings_coverage"),
        patch("backend.util.service.AppService.run_service"),
    ):
        sched = Scheduler(register_system_tasks=False)
        sched.run_service()
        sched._invalidate_jobs_cache()
        yield sched


def _job_state(store, job_id: str):
    with store.engine.begin() as conn:
        return conn.execute(
            select(store.jobs_t.c.job_state).where(store.jobs_t.c.id == job_id)
        ).scalar()


def _set_job_state(store, job_id: str, job_state) -> None:
    with store.engine.begin() as conn:
        conn.execute(
            store.jobs_t.update()
            .where(store.jobs_t.c.id == job_id)
            .values(job_state=job_state)
        )


def _poison(store, job_id: str) -> None:
    with store.engine.begin() as conn:
        conn.execute(
            store.jobs_t.update()
            .where(store.jobs_t.c.id == job_id)
            .values(job_state=POISONED_STATE)
        )


def _ids(store) -> set[str]:
    with store.engine.begin() as conn:
        return {r.id for r in conn.execute(select(store.jobs_t.c.id))}


def _next_run_time(store, job_id: str):
    with store.engine.begin() as conn:
        return conn.execute(
            select(store.jobs_t.c.next_run_time).where(store.jobs_t.c.id == job_id)
        ).scalar()


def _utc(timestamp: float):
    from datetime import datetime, timezone

    return datetime.fromtimestamp(timestamp, tz=timezone.utc)
