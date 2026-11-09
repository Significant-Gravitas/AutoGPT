"""The scheduler's side of the hourly community sweep, on a real SQLite
jobstore: registering memory scopes adds no job per scope, the sweep's
registration, and its removal of the per-scope crons registered before it.
The sweep's choice of scopes is ``copilot/dream/community_sweep_test.py``.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from prisma.enums import MemoryScopeScheduleState

from backend.copilot.dream import registry, scope_jobs
from backend.copilot.dream.registry_fakes_test import install_registry_env
from backend.copilot.dream.scheduling import NIGHTLY_BATCH_JOB_PREFIX, dream_system_job
from backend.copilot.graphiti.scope import MemoryScope
from backend.executor import scheduler as sm
from backend.executor.jobstore import WRITE_TIMEOUT, ResilientSQLAlchemyJobStore

_EXECUTION = sm.Jobstores.EXECUTION.value
_MEMORY = sm.Jobstores.MEMORY_CRON_RECONCILE.value


def noop(**kwargs: Any) -> None:
    """A job body the SQLite jobstore can pickle by reference."""


@contextmanager
def _scheduler(tmp_path: Path) -> Iterator[sm.Scheduler]:
    """A ``Scheduler`` whose APScheduler, started paused, keeps its EXECUTION
    jobs in a SQLite file."""
    store = ResilientSQLAlchemyJobStore(url=f"sqlite:///{tmp_path / 'jobs.sqlite'}")
    apscheduler = BackgroundScheduler(
        jobstores={_EXECUTION: store, _MEMORY: MemoryJobStore()},
        timezone=UTC,
    )
    apscheduler.start(paused=True)
    service = sm.Scheduler(register_system_tasks=False)
    service.scheduler, service._execution_jobstore = apscheduler, store
    try:
        yield service
    finally:
        apscheduler.remove_jobstore(_EXECUTION)
        apscheduler.shutdown(wait=True)
        store.engine.dispose()


class _SchedulerClient:
    """``get_scheduler_client()`` over a real ``Scheduler``: every RPC the
    registry makes runs the scheduler's own method."""

    def __init__(self, service: sm.Scheduler) -> None:
        self._service = service

    def __getattr__(self, name: str) -> Any:
        method = getattr(self._service, name)

        async def call(**kwargs: Any) -> Any:
            return await asyncio.to_thread(method, **kwargs)

        return call


def _execution_job_ids(service: sm.Scheduler) -> set[str]:
    return {job.id for job in service.scheduler.get_jobs(jobstore=_EXECUTION)}


def _add_old_community_cron(service: sm.Scheduler, scope: MemoryScope) -> None:
    """A per-scope weekly cron as the registry used to add it."""
    func, kwargs = sm.execute_community_rebuild, {"user_id": scope.owner_user_id}
    if scope.expert_id is not None:
        kwargs |= {"expert_id": scope.expert_id, "func": func}
        func = sm.execute_expert_memory_cron
    service.scheduler.add_job(
        func,
        CronTrigger.from_crontab("0 4 * * 0", timezone="Asia/Kolkata"),
        kwargs=kwargs,
        id=f"community_rebuild_{scope.scope_key}",
        jobstore=_EXECUTION,
    )


@pytest.mark.asyncio
async def test_registering_memory_scopes_adds_no_job_per_scope(monkeypatch, tmp_path):
    """Each scope gets its registry row, which the hourly sweep reads, and
    the scheduler gets no job: with every flag off there is no cron left to
    register for it."""
    env = install_registry_env(monkeypatch)
    env.flags = {}
    scopes = [MemoryScope.for_user(f"user-{n}") for n in range(5)]
    scopes.append(MemoryScope.for_expert("user-0", "expert-1"))
    with _scheduler(tmp_path) as service:
        monkeypatch.setattr(
            scope_jobs, "get_scheduler_client", lambda: _SchedulerClient(service)
        )
        for scope in scopes:
            await registry.ensure_scope_scheduled(scope)

        assert _execution_job_ids(service) == set()
    assert {key: row.state for key, row in env.db.rows.items()} == {
        scope.scope_key: MemoryScopeScheduleState.ACTIVE for scope in scopes
    }


def test_the_old_per_scope_community_crons_are_removed_in_batches(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(sm, "COMMUNITY_CRON_REMOVAL_BATCH", 3)
    old = [MemoryScope.for_user(f"user-{n}") for n in range(6)]
    old.append(MemoryScope.for_expert("user-0", "expert-1"))
    nightly = dream_system_job(NIGHTLY_BATCH_JOB_PREFIX)
    with _scheduler(tmp_path) as service:
        for scope in old:
            _add_old_community_cron(service, scope)
        service._add_memory_job(
            *sm._MEMORY_CRONS[NIGHTLY_BATCH_JOB_PREFIX], nightly, old[0], "UTC"
        )
        # Matched only if the prefix's ``_`` were a LIKE wildcard.
        decoy = "community_rebuildXuser-9"
        service.scheduler.add_job(
            noop, "interval", hours=1, id=decoy, jobstore=_EXECUTION
        )

        store = service._execution_jobstore
        remove, bounds = store.remove_job, []

        def bounded_remove(job_id: str) -> None:
            bounds.append(WRITE_TIMEOUT.get())
            remove(job_id)

        # Each delete gives up on a row another transaction holds.
        monkeypatch.setattr(store, "remove_job", bounded_remove)

        assert service._remove_community_crons() == len(old)
        assert _execution_job_ids(service) == {nightly.job_id(old[0]), decoy}
        assert service._remove_community_crons() == 0
    assert bounds == [sm.MEMORY_CRON_WRITE_TIMEOUT_SECONDS] * len(old)
    assert WRITE_TIMEOUT.get() is None


def test_the_sweep_runs_hourly_in_each_process_from_its_start(tmp_path):
    """Process-local, so an older scheduler never loads it, and due at once,
    so a deploy removes the per-scope crons without waiting an hour."""
    with _scheduler(tmp_path) as service:
        before = datetime.now(UTC)
        service._register_community_sweep()

        [job] = service.scheduler.get_jobs(jobstore=_MEMORY)
        assert job.func is sm.rebuild_due_communities_job
        assert str(job.trigger) == str(IntervalTrigger(hours=1))
        assert before <= job.next_run_time <= datetime.now(UTC)
        assert _execution_job_ids(service) == set()


@pytest.fixture
def scheduler_loop() -> Iterator[asyncio.AbstractEventLoop]:
    """A real event loop on its own thread, installed as the scheduler's
    shared loop so ``run_async`` bridges to it as in production."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        with patch.object(sm, "_event_loop", loop):
            yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=2)
        loop.close()


@pytest.mark.parametrize("removal_fails", [False, True])
def test_the_hourly_job_removes_the_old_crons_and_sweeps(
    monkeypatch, tmp_path, scheduler_loop, removal_fails
):
    sweep = AsyncMock()
    monkeypatch.setattr(sm, "rebuild_due_communities", sweep)
    with _scheduler(tmp_path) as service:
        _add_old_community_cron(service, MemoryScope.for_user("user-1"))
        if removal_fails:
            monkeypatch.setattr(
                service, "_remove_community_crons", MagicMock(side_effect=OSError)
            )
        with patch.object(sm, "_scheduler_instance", service):
            sm.rebuild_due_communities_job()

        assert bool(_execution_job_ids(service)) is removal_fails
    sweep.assert_awaited_once_with(
        sm._expert_scope_status, sm.SCHEDULER_DREAM_OPERATION_TIMEOUT_SECONDS
    )
