import os
import pickle
import tempfile
from contextlib import contextmanager
from enum import Enum

from apscheduler.schedulers.background import BackgroundScheduler
from sqlalchemy import select

from backend.executor.jobstore import ResilientSQLAlchemyJobStore
from backend.executor.jobstore_backfill import _strip_enums


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
    try:
        pickle.loads(POISONED_STATE)
    except ValueError as e:
        assert "is not a valid _RemovedEnum" in str(e)
    else:
        raise AssertionError("expected the enum lookup to fail")


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


def test_upstream_jobstore_would_have_deleted_the_row():
    """Control: pins the upstream behaviour this subclass exists to prevent."""
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

    with _store(cls=SQLAlchemyJobStore) as (store, scheduler):
        scheduler.add_job(noop, "interval", seconds=3600, id="poisoned")
        _poison(store, "poisoned")

        store._get_jobs()

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
