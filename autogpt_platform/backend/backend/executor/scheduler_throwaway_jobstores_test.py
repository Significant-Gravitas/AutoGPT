"""The test server's scheduler never runs or rewrites the jobs a real
scheduler saved in the platform database, and saves nothing there itself."""

import os
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from apscheduler.job import Job
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import Engine, MetaData, create_engine, text

from backend.executor import scheduler as scheduler_module
from backend.executor.jobstore import ResilientSQLAlchemyJobStore
from backend.executor.scheduler import Jobstores, Scheduler, _extract_schema_from_url
from backend.util.test import SpinTestServer

_EXECUTION = Jobstores.EXECUTION.value
_RUNS: list[str] = []
_MARKER_RAN = threading.Event()


def record_run(name: str) -> None:
    _RUNS.append(name)
    if name == "marker":
        _MARKER_RAN.set()


def _postgres_url() -> str | None:
    direct = os.getenv("DIRECT_URL") or ""
    return direct if direct.startswith("postgresql") else None


@pytest.mark.skipif(_postgres_url() is None, reason="needs the PostgreSQL test DB")
def test_the_test_servers_scheduler_leaves_saved_jobs_alone(
    platform_db: tuple[Engine, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, schema = platform_db
    _save_overdue_job(engine, schema)
    saved = _rows(engine, schema)
    scheduler = SpinTestServer().scheduler

    with _serving(scheduler, monkeypatch):
        # Recurring, so a marker the platform database got keeps its row there.
        scheduler.scheduler.add_job(
            record_run,
            "interval",
            hours=6,
            args=["marker"],
            id="marker",
            next_run_time=datetime.now(timezone.utc),
            jobstore=_EXECUTION,
        )
        # Due before the marker, so a loaded saved job is submitted first and
        # cleanup's shutdown(wait=True) lets it finish.
        assert _MARKER_RAN.wait(10)
        database = scheduler._execution_jobstore.engine.url.database

    assert _RUNS == ["marker"]
    assert _rows(engine, schema) == saved
    assert database and not os.path.exists(database)


@pytest.fixture
def platform_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Engine, str]]:
    """A schema of its own in the test PostgreSQL, standing in for a platform
    database a real scheduler has used; DIRECT_URL points the scheduler at it."""
    _RUNS.clear()
    _MARKER_RAN.clear()
    url = _extract_schema_from_url(_postgres_url())[1]
    schema = f"saved_jobs_{uuid.uuid4().hex[:12]}"
    engine = create_engine(url)
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    monkeypatch.setenv(
        "DIRECT_URL", f"{url}{'&' if '?' in url else '?'}schema={schema}"
    )
    try:
        yield engine, schema
    finally:
        with engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        engine.dispose()


def _save_overdue_job(engine: Engine, schema: str) -> None:
    """An hour-overdue interval job with the scheduler's job defaults, written
    to the store directly: shutting down a paused scheduler processes due jobs."""
    store = ResilientSQLAlchemyJobStore(
        engine=engine, metadata=MetaData(schema=schema), tablename="apscheduler_jobs"
    )
    seeder = BackgroundScheduler(timezone=timezone.utc)
    store.start(seeder, _EXECUTION)
    store.add_job(
        Job(
            seeder,
            id="saved",
            func=record_run,
            args=("saved",),
            kwargs={},
            trigger=IntervalTrigger(hours=6, timezone=timezone.utc),
            next_run_time=datetime.now(timezone.utc) - timedelta(hours=1),
            executor="default",
            coalesce=True,
            misfire_grace_time=None,
            max_instances=1000,
        )
    )


def _rows(engine: Engine, schema: str) -> list[tuple]:
    with engine.connect() as conn:
        return [
            (row.id, row.next_run_time, bytes(row.job_state))
            for row in conn.execute(
                text(
                    "SELECT id, next_run_time, job_state "
                    f'FROM "{schema}".apscheduler_jobs ORDER BY id'
                )
            )
        ]


@contextmanager
def _serving(service: Scheduler, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """``run_service`` up to where the process would serve RPCs, then the
    service's own cleanup; the module globals it sets are restored after."""
    for name in ("_event_loop", "_event_loop_thread", "_scheduler_instance"):
        monkeypatch.setattr(scheduler_module, name, getattr(scheduler_module, name))
    with (
        patch.object(scheduler_module, "load_dotenv"),
        patch("backend.util.service.AppService.run_service"),
        patch("backend.util.service.AppService.cleanup"),
    ):
        service.run_service()
        try:
            yield
        finally:
            service.cleanup()
            if scheduler_module._event_loop is not None:
                scheduler_module._event_loop.close()
