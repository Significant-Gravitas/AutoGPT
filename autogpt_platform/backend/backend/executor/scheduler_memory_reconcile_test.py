"""The scheduler's side of the memory cron reconcile
(``reconcile_memory_scope_crons``): its filtered listing of the dream-system
crons (inside the pass's ceiling, and stopped by the database), the scope
and timezone each names, the cron store it hands the pass, its registration
as a system job, and removal that another scheduler got to first. The
pass's logic is ``copilot/dream/scope_reconcile_test.py``.

The listing, registration and removal probes are an independent review's
(PR 9 round 2), on real SQLite jobstores as there.
"""

import asyncio
import concurrent.futures
import os
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from backend.copilot.dream.scheduling import (
    COMMUNITY_REBUILD_JOB_PREFIX,
    NIGHTLY_BATCH_JOB_PREFIX,
    dream_system_job,
)
from backend.copilot.dream.scope_reconcile import (
    RECONCILE_INTERVAL_MINUTES,
    CronStore,
    ListedCron,
    ReconcileReport,
)
from backend.copilot.graphiti.scope import MemoryScope
from backend.executor import scheduler as scheduler_module
from backend.executor.jobstore import ResilientSQLAlchemyJobStore
from backend.executor.scheduler import (
    MEMORY_CRON_LISTING_TIMEOUT_SECONDS,
    Jobstores,
    Scheduler,
    _extract_schema_from_url,
    _SchedulerCrons,
    listed_cron,
    reconcile_memory_scope_crons,
)

EXPERT = MemoryScope.for_expert("user-1", "expert-1")
_EXECUTION = Jobstores.EXECUTION.value
_RECONCILE_JOB = "reconcile_memory_scope_crons"


def noop(**kwargs: Any) -> None:
    """A job body the SQLite jobstore can pickle by reference."""


@contextmanager
def _sqlite_file() -> Iterator[str]:
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    try:
        yield f"sqlite:///{path}"
    finally:
        os.unlink(path)


@contextmanager
def _scheduler_on(url: str) -> Iterator[Scheduler]:
    """A ``Scheduler`` whose APScheduler, started paused, keeps its EXECUTION
    jobs in the SQLite database at ``url``, as two schedulers in a rolling
    deploy share one jobstore."""
    store = ResilientSQLAlchemyJobStore(url=url)
    apscheduler = BackgroundScheduler()
    apscheduler.add_jobstore(store, _EXECUTION)
    apscheduler.start(paused=True)
    scheduler = Scheduler(register_system_tasks=False)
    scheduler.scheduler = apscheduler
    scheduler._execution_jobstore = store
    try:
        yield scheduler
    finally:
        # The store goes first: a scheduler shutting down processes the jobs
        # due on it, and would move an overdue one on.
        apscheduler.remove_jobstore(_EXECUTION, shutdown=True)
        apscheduler.shutdown(wait=False)
        store.engine.dispose()  # Windows will not unlink an open file


@pytest.fixture
def scheduler_loop():
    """A real event loop on its own thread, installed as the scheduler's
    shared loop so ``run_async`` bridges to it as in production."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        with patch.object(scheduler_module, "_event_loop", loop):
            yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


def test_the_listing_reads_only_the_dream_system_crons() -> None:
    """One read, filtered by prefix, with ``_`` matched literally: a job id
    that only a LIKE wildcard would match is left out."""
    with _sqlite_file() as url, _scheduler_on(url) as scheduler:
        for job_id in (
            "community_rebuild_user-1",
            "dream_nightly_batch_user-2",
            "communityXrebuild_user-3",
            "dream_nightly_batchy",
            "graph-execution-1",
        ):
            scheduler.scheduler.add_job(
                noop,
                "interval",
                hours=1,
                id=job_id,
                kwargs={"user_id": "user-1"},
                jobstore=_EXECUTION,
            )

        listed = sorted(job.id for job in scheduler._memory_jobs())

    assert listed == ["community_rebuild_user-1", "dream_nightly_batch_user-2"]


def test_the_listing_names_each_crons_scope_and_timezone() -> None:
    """Registered as the scope RPCs register them, each cron lists with the
    scope its kwargs name and the timezone its trigger runs in, the form
    the registry row keeps."""
    account = MemoryScope.for_user("user-1")
    with _sqlite_file() as url, _scheduler_on(url) as scheduler:
        community = dream_system_job(COMMUNITY_REBUILD_JOB_PREFIX)
        nightly = dream_system_job(NIGHTLY_BATCH_JOB_PREFIX)
        scheduler._add_memory_job(noop, "0 4 * * 0", community, EXPERT, "Asia/Tokyo")
        scheduler._add_memory_job(noop, "0 3 * * *", nightly, account, "UTC")

        listed = sorted(scheduler._memory_crons(), key=lambda cron: cron.job_id)

    assert [(cron.job_id, cron.scope, cron.timezone) for cron in listed] == [
        (community.job_id(EXPERT), EXPERT, "Asia/Tokyo"),
        (nightly.job_id(account), account, "UTC"),
    ]


@pytest.mark.parametrize(
    "kwargs, scope",
    [
        ({"user_id": "user-1"}, MemoryScope.for_user("user-1")),
        ({"user_id": "user-1", "expert_id": "expert-1"}, EXPERT),
        ({"user_id": "user-1", "expert_id": None}, MemoryScope.for_user("user-1")),
        ({}, None),
        ({"user_id": ""}, None),
    ],
)
def test_a_listed_cron_names_the_scope_its_kwargs_do(
    kwargs: dict[str, Any], scope: MemoryScope | None
) -> None:
    job = MagicMock(
        id="dream_nightly_batch_x",
        kwargs=kwargs,
        trigger=CronTrigger.from_crontab("0 3 * * *", timezone="Europe/Paris"),
    )

    cron = listed_cron(job)

    assert (cron.scope if cron is not None else None) == scope
    assert cron is None or cron.timezone == "Europe/Paris"


def test_the_listing_asks_the_database_to_stop_it() -> None:
    """The jobstore read carries ``MEMORY_CRON_LISTING_TIMEOUT_SECONDS``:
    cancelling the wait for it leaves its thread running, so the database
    is what stops a stalled read."""
    scheduler = Scheduler(register_system_tasks=False)
    scheduler._execution_jobstore = ResilientSQLAlchemyJobStore(url="sqlite://")

    with patch.object(
        scheduler._execution_jobstore, "_get_jobs", return_value=[]
    ) as read:
        scheduler._memory_jobs()

    assert read.call_args.kwargs == {
        "statement_timeout": MEMORY_CRON_LISTING_TIMEOUT_SECONDS
    }


def test_a_statement_timeout_leaves_a_sqlite_read_as_it_was() -> None:
    """SQLite has no statement timeout; the read goes through unbounded."""
    with _sqlite_file() as url, _scheduler_on(url) as scheduler:
        scheduler.scheduler.add_job(
            noop,
            "interval",
            hours=1,
            id="community_rebuild_user-1",
            jobstore=_EXECUTION,
        )

        jobs = scheduler._execution_jobstore._get_jobs(statement_timeout=0.001)

    assert [job.id for job in jobs] == ["community_rebuild_user-1"]


def _postgres_url() -> str | None:
    """The test database's SQLAlchemy URL, as the scheduler builds it, when
    the run has a PostgreSQL one."""
    direct = os.getenv("DIRECT_URL") or ""
    if not direct.startswith("postgresql"):
        return None
    return _extract_schema_from_url(direct)[1]


@pytest.mark.skipif(_postgres_url() is None, reason="needs the PostgreSQL test DB")
def test_postgresql_stops_a_listing_that_runs_past_its_statement_timeout() -> None:
    """On PostgreSQL the bounded read is cancelled by the server, and the
    same read without the bound runs to its end."""
    store = ResilientSQLAlchemyJobStore(
        url=_postgres_url(), tablename=f"apscheduler_jobs_probe_{os.getpid()}"
    )
    apscheduler = BackgroundScheduler()
    apscheduler.add_jobstore(store, _EXECUTION)
    apscheduler.start(paused=True)
    try:
        apscheduler.add_job(noop, "interval", hours=1, id="a", jobstore=_EXECUTION)
        slow = text("pg_sleep(1) IS NOT NULL")

        with pytest.raises(OperationalError, match="statement timeout"):
            store._get_jobs(slow, statement_timeout=0.1)
        assert [job.id for job in store._get_jobs(slow)] == ["a"]
    finally:
        apscheduler.shutdown(wait=False)
        store.jobs_t.drop(store.engine, checkfirst=True)
        store.engine.dispose()


@pytest.mark.asyncio
async def test_the_cron_store_lists_and_registers_as_the_scope_rpcs_do() -> None:
    """The pass lists through ``_memory_crons``, registers through the same
    body and crontab as the scope RPCs, and removes through
    ``remove_scope_memory_jobs``."""
    scheduler = Scheduler(register_system_tasks=False)
    crons: CronStore = _SchedulerCrons(scheduler)
    listed = [ListedCron(job_id="x", scope=EXPERT, timezone="UTC")]
    with (
        patch.object(scheduler, "_memory_crons", return_value=listed),
        patch.object(scheduler, "_add_memory_job") as add,
        patch.object(scheduler, "remove_scope_memory_jobs") as remove,
    ):
        assert await crons.listing() == listed
        await crons.add(dream_system_job(COMMUNITY_REBUILD_JOB_PREFIX), EXPERT, "UTC")
        await crons.add(dream_system_job(NIGHTLY_BATCH_JOB_PREFIX), EXPERT, "UTC")
        await crons.remove(EXPERT)

    assert [call.args[:2] for call in add.call_args_list] == [
        (scheduler_module.execute_community_rebuild, "0 4 * * 0"),
        (scheduler_module.execute_nightly_batch_sync, "0 3 * * *"),
    ]
    assert [call.args[3:] for call in add.call_args_list] == [(EXPERT, "UTC")] * 2
    remove.assert_called_once_with(EXPERT)


def test_a_pass_hands_the_reconcile_its_cron_store() -> None:
    scheduler = Scheduler(register_system_tasks=False)
    seen: dict[str, Any] = {}

    async def fake_reconcile(store: CronStore):
        seen["store"] = store
        return ReconcileReport(restored=["dream_nightly_batch_user-1"])

    with (
        patch.object(scheduler_module, "reconcile", fake_reconcile),
        patch.object(
            scheduler_module,
            "run_async",
            side_effect=lambda coro, **kwargs: asyncio.run(coro),
        ),
    ):
        scheduler._reconcile_memory_crons()

    assert isinstance(seen["store"], _SchedulerCrons)


def test_a_stalled_listing_is_cut_off_at_the_passs_ceiling(
    scheduler_loop, monkeypatch, caplog
) -> None:
    """The review's probe: the listing ran before the ceiling started, so a
    stalled one held the pass (and, one at a time, every later one). It now
    runs inside the ceiling: the pass returns at the ceiling while the
    listing still hangs."""
    monkeypatch.setattr(scheduler_module, "MEMORY_CRON_RECONCILE_TIMEOUT_SECONDS", 0.2)
    scheduler = Scheduler(register_system_tasks=False)
    release = threading.Event()
    started = threading.Event()

    def stalled_listing() -> list:
        started.set()
        release.wait(timeout=5)
        return []

    monkeypatch.setattr(scheduler, "_memory_jobs", stalled_listing)
    began = time.monotonic()
    try:
        scheduler._reconcile_memory_crons()
        took = time.monotonic() - began
    finally:
        release.set()

    assert started.is_set(), "the listing ran inside the pass"
    assert took < 2
    assert "Memory cron reconcile cut off" in caplog.text


def test_a_pass_cut_off_at_its_timeout_waits_for_the_next(caplog) -> None:
    scheduler = Scheduler(register_system_tasks=False)

    def cut_off(coro, **kwargs):
        coro.close()
        assert kwargs["cancel_on_timeout"] is True
        raise concurrent.futures.TimeoutError

    with patch.object(scheduler_module, "run_async", side_effect=cut_off):
        scheduler._reconcile_memory_crons()

    assert "Memory cron reconcile cut off" in caplog.text


def test_the_system_task_waits_for_a_running_scheduler(caplog) -> None:
    with patch.object(scheduler_module, "_scheduler_instance", None):
        reconcile_memory_scope_crons()

    assert "scheduler not running" in caplog.text


class TestRegistration:
    """Registered after start and left alone while unchanged, as the PostHog
    sweep is, so neither a restart nor a second scheduler postpones it."""

    def _register(self, existing=None) -> MagicMock:
        scheduler = Scheduler.__new__(Scheduler)
        scheduler.scheduler = MagicMock()
        scheduler.scheduler.get_job.return_value = existing
        Scheduler._register_memory_cron_reconcile(scheduler)
        return scheduler.scheduler.add_job

    def test_registers_the_interval_job_when_missing(self) -> None:
        add_job = self._register()

        add_job.assert_called_once()
        assert add_job.call_args.args[0] is reconcile_memory_scope_crons
        assert str(add_job.call_args.args[1]) == str(
            IntervalTrigger(minutes=RECONCILE_INTERVAL_MINUTES)
        )
        assert add_job.call_args.kwargs == {
            "id": _RECONCILE_JOB,
            "replace_existing": True,
            "max_instances": 1,
            "coalesce": True,
            "jobstore": _EXECUTION,
        }

    def test_leaves_an_unchanged_job_alone(self) -> None:
        existing = MagicMock(
            trigger=IntervalTrigger(minutes=RECONCILE_INTERVAL_MINUTES)
        )

        self._register(existing=existing).assert_not_called()

    def test_replaces_a_job_whose_interval_changed(self) -> None:
        existing = MagicMock(trigger=IntervalTrigger(minutes=60))

        self._register(existing=existing).assert_called_once()

    def test_a_second_scheduler_starting_keeps_an_overdue_pass_due(self) -> None:
        """The review's probe on a shared SQLite jobstore: the first
        scheduler registers the job, the pass falls overdue, and a second
        scheduler starting registers it again. One job, still due when it
        was."""
        overdue = datetime.now(timezone.utc) - timedelta(minutes=5)
        with _sqlite_file() as url:
            with _scheduler_on(url) as first:
                first._register_memory_cron_reconcile()
                first.scheduler.modify_job(
                    _RECONCILE_JOB, jobstore=_EXECUTION, next_run_time=overdue
                )
            with _scheduler_on(url) as second:
                second._register_memory_cron_reconcile()
                jobs = second.scheduler.get_jobs(jobstore=_EXECUTION)

        assert [job.id for job in jobs] == [_RECONCILE_JOB]
        assert jobs[0].next_run_time == overdue


def test_a_removal_another_scheduler_made_first_is_no_error() -> None:
    """The review's probe: two schedulers on one jobstore both read a
    scope's crons, the other removes them first, and this one's removal
    finds them gone. It reports only what it removed, and raises nothing."""
    with _sqlite_file() as url, _scheduler_on(url) as one, _scheduler_on(url) as other:
        for prefix in (COMMUNITY_REBUILD_JOB_PREFIX, NIGHTLY_BATCH_JOB_PREFIX):
            one._add_memory_job(
                noop, "0 3 * * *", dream_system_job(prefix), EXPERT, "UTC"
            )
        get_job = one.scheduler.get_job

        def read_then_lose(job_id: str, jobstore: str):
            job = get_job(job_id, jobstore=jobstore)
            if job_id.startswith(NIGHTLY_BATCH_JOB_PREFIX):
                other.scheduler.remove_job(job_id, jobstore=jobstore)
            return job

        with patch.object(one.scheduler, "get_job", side_effect=read_then_lose):
            removed = one.remove_scope_memory_jobs(EXPERT)

        left = one.scheduler.get_jobs(jobstore=_EXECUTION)

    assert removed == [dream_system_job(COMMUNITY_REBUILD_JOB_PREFIX).job_id(EXPERT)]
    assert left == []
