"""The scheduler's side of the memory cron reconcile
(``reconcile_memory_scope_crons``): its one filtered listing of the
dream-system crons, the scope each names, the cron store it hands the pass,
and the pass itself. The pass's logic is ``copilot/dream/
scope_reconcile_test.py``; its registration as a system job is in
``scheduler_unit_test.py``."""

import asyncio
import concurrent.futures
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from apscheduler.schedulers.background import BackgroundScheduler

from backend.copilot.dream.scheduling import (
    COMMUNITY_REBUILD_JOB_PREFIX,
    NIGHTLY_BATCH_JOB_PREFIX,
    dream_system_job,
)
from backend.copilot.dream.scope_reconcile import CronStore, ListedCron, ReconcileReport
from backend.copilot.graphiti.scope import MemoryScope
from backend.executor import scheduler as scheduler_module
from backend.executor.jobstore import ResilientSQLAlchemyJobStore
from backend.executor.scheduler import (
    Scheduler,
    _SchedulerCrons,
    listed_cron,
    reconcile_memory_scope_crons,
)

EXPERT = MemoryScope.for_expert("user-1", "expert-1")


def noop(**kwargs: Any) -> None:
    """A job body the SQLite jobstore can pickle by reference."""


@contextmanager
def _sqlite_store() -> (
    Iterator[tuple[ResilientSQLAlchemyJobStore, BackgroundScheduler]]
):
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    store = ResilientSQLAlchemyJobStore(url=f"sqlite:///{path}")
    apscheduler = BackgroundScheduler()
    apscheduler.add_jobstore(store, "default")
    apscheduler.start(paused=True)
    try:
        yield store, apscheduler
    finally:
        apscheduler.shutdown(wait=False)
        store.engine.dispose()  # Windows will not unlink an open file
        os.unlink(path)


def test_the_listing_reads_only_the_dream_system_crons() -> None:
    """One read, filtered by prefix, with ``_`` matched literally: a job id
    that only a LIKE wildcard would match is left out."""
    with _sqlite_store() as (store, apscheduler):
        for job_id in (
            "community_rebuild_user-1",
            "dream_nightly_batch_user-2",
            "communityXrebuild_user-3",
            "dream_nightly_batchy",
            "graph-execution-1",
        ):
            apscheduler.add_job(
                noop, "interval", hours=1, id=job_id, kwargs={"user_id": "user-1"}
            )
        scheduler = Scheduler(register_system_tasks=False)
        scheduler._execution_jobstore = store

        listed = sorted(job.id for job in scheduler._memory_jobs())

    assert listed == ["community_rebuild_user-1", "dream_nightly_batch_user-2"]


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
    job = MagicMock(id="dream_nightly_batch_x", kwargs=kwargs)

    cron = listed_cron(job)

    assert (cron.scope if cron is not None else None) == scope


@pytest.mark.asyncio
async def test_the_cron_store_registers_each_cron_on_its_body_and_crontab() -> None:
    """The pass registers through the same body and crontab as the scope
    RPCs, and removes through ``remove_scope_memory_jobs``."""
    scheduler = Scheduler(register_system_tasks=False)
    crons: CronStore = _SchedulerCrons(scheduler)
    with (
        patch.object(scheduler, "_add_memory_job") as add,
        patch.object(scheduler, "remove_scope_memory_jobs") as remove,
    ):
        await crons.add(dream_system_job(COMMUNITY_REBUILD_JOB_PREFIX), EXPERT, "UTC")
        await crons.add(dream_system_job(NIGHTLY_BATCH_JOB_PREFIX), EXPERT, "UTC")
        await crons.remove(EXPERT)

    assert [call.args[:2] for call in add.call_args_list] == [
        (scheduler_module.execute_community_rebuild, "0 4 * * 0"),
        (scheduler_module.execute_nightly_batch_sync, "0 3 * * *"),
    ]
    assert [call.args[3:] for call in add.call_args_list] == [(EXPERT, "UTC")] * 2
    remove.assert_called_once_with(EXPERT)


def test_a_pass_hands_its_one_listing_and_its_cron_store_to_the_reconcile() -> None:
    scheduler = Scheduler(register_system_tasks=False)
    jobs = [
        MagicMock(id="dream_nightly_batch_user-1", kwargs={"user_id": "user-1"}),
        MagicMock(id="community_rebuild_broken", kwargs={}),
    ]
    seen: dict[str, Any] = {}

    async def fake_reconcile(listed: list[ListedCron], store: CronStore):
        seen.update(listed=listed, store=store)
        return ReconcileReport(restored=["dream_nightly_batch_user-1"])

    with (
        patch.object(scheduler, "_memory_jobs", return_value=jobs) as memory_jobs,
        patch.object(scheduler_module, "reconcile", fake_reconcile),
        patch.object(
            scheduler_module,
            "run_async",
            side_effect=lambda coro, **kwargs: asyncio.run(coro),
        ),
    ):
        scheduler._reconcile_memory_crons()

    memory_jobs.assert_called_once_with()
    assert [cron.job_id for cron in seen["listed"]] == ["dream_nightly_batch_user-1"]
    assert isinstance(seen["store"], _SchedulerCrons)


def test_a_pass_cut_off_at_its_timeout_waits_for_the_next(caplog) -> None:
    scheduler = Scheduler(register_system_tasks=False)

    def cut_off(coro, **kwargs):
        coro.close()
        assert kwargs["cancel_on_timeout"] is True
        raise concurrent.futures.TimeoutError

    with (
        patch.object(scheduler, "_memory_jobs", return_value=[]),
        patch.object(scheduler_module, "run_async", side_effect=cut_off),
    ):
        scheduler._reconcile_memory_crons()

    assert "Memory cron reconcile cut off" in caplog.text


def test_the_system_task_waits_for_a_running_scheduler(caplog) -> None:
    with patch.object(scheduler_module, "_scheduler_instance", None):
        reconcile_memory_scope_crons()

    assert "scheduler not running" in caplog.text
