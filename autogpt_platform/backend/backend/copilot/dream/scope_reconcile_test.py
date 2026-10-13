"""Tests for the scheduler's reconcile pass over the memory-scope crons
(``scope_reconcile.py``), against the registry fakes: it restores what the
event-driven paths lose, removes what they leave behind, and converges.

The two race probes are an independent review's (PR 9 round 1), which left
an ACTIVE row with its job ids recorded and no crons; each now ends with the
crons back after one pass.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from backend.util.feature_flag import Flag

from . import registry, scope_crons, scope_reconcile
from .registry_fakes_test import (
    ACCOUNT,
    ACTIVE,
    EXPERT,
    OTHER_EXPERT,
    PARIS,
    PAUSED,
    TOKYO,
    WIPED,
    RegistryEnv,
    install_registry_env,
    job_ids,
)

TIMEOUT = 2


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    return install_registry_env(monkeypatch)


async def _stale_takedown(env: RegistryEnv, monkeypatch: pytest.MonkeyPatch) -> None:
    """One interleaving from the review: an ensure parks before recording
    while a pause lands, its record is refused, and its takedown parks while
    a resume registers and records the crons; then the stale takedown lands
    and takes them."""
    record, remove = env.db.record_scope_jobs, env.scheduler._remove
    record_reached, record_go = asyncio.Event(), asyncio.Event()
    remove_reached, remove_go = asyncio.Event(), asyncio.Event()
    ensure: asyncio.Task | None = None

    async def parked_record(*args: Any, **kwargs: Any) -> bool:
        if asyncio.current_task() is ensure and not record_reached.is_set():
            record_reached.set()
            await record_go.wait()
        return await record(*args, **kwargs)

    async def parked_remove(*, scope):
        if asyncio.current_task() is ensure and not remove_reached.is_set():
            remove_reached.set()
            await remove_go.wait()
        return await remove(scope=scope)

    monkeypatch.setattr(env.db, "record_scope_jobs", parked_record)
    env.scheduler.remove_scope_memory_jobs.side_effect = parked_remove
    ensure = asyncio.create_task(
        registry.ensure_scope_scheduled(EXPERT, force_refresh=True)
    )
    await asyncio.wait_for(record_reached.wait(), TIMEOUT)
    assert await registry.pause_scope(EXPERT)
    record_go.set()
    await asyncio.wait_for(remove_reached.wait(), TIMEOUT)
    await registry.resume_scope(EXPERT)
    assert set(env.scheduler.jobs) == job_ids(EXPERT)
    remove_go.set()
    await asyncio.wait_for(ensure, TIMEOUT)
    monkeypatch.setattr(env.db, "record_scope_jobs", record)
    env.scheduler.remove_scope_memory_jobs.side_effect = remove


@pytest.mark.asyncio
@pytest.mark.parametrize("cycles", [1, 2])
async def test_every_pause_resume_interleaving_is_restored_by_one_pass(
    env, monkeypatch, cycles
):
    """After ``cycles`` stale takedowns the row is ACTIVE with both ids
    recorded and no crons, and an ordinary ensure, trusting the ids,
    restores nothing. One pass restores them."""
    await registry.ensure_scope_scheduled(EXPERT)
    for _ in range(cycles):
        await _stale_takedown(env, monkeypatch)
    row = env.db.rows[EXPERT.scope_key]
    assert row.state == ACTIVE
    assert {row.community_job_id, row.nightly_job_id} == job_ids(EXPERT)
    assert env.scheduler.jobs == {}
    await registry.ensure_scope_scheduled(EXPERT)
    assert env.scheduler.jobs == {}, "an ensure trusts the recorded ids"

    report = await env.reconcile()

    assert set(env.scheduler.jobs) == job_ids(EXPERT)
    assert sorted(report.restored) == sorted(job_ids(EXPERT))


@pytest.mark.asyncio
async def test_a_pause_whose_re_read_fails_reports_false_and_is_restored(
    env, monkeypatch
):
    """The review's probe through the pause: a resume registers its crons just
    before the pause's removal takes them, and the pause's one re-read
    fails. The pause no longer reports success, and one pass restores the
    crons."""
    await registry.ensure_scope_scheduled(EXPERT)
    remove = env.scheduler._remove
    get = env.db.get_scope_schedule

    async def resume_before_remove(*, scope):
        await registry.resume_scope(scope)
        removed = await remove(scope=scope)
        monkeypatch.setattr(
            env.db, "get_scope_schedule", AsyncMock(side_effect=ConnectionError)
        )
        return removed

    env.scheduler.remove_scope_memory_jobs.side_effect = resume_before_remove

    assert not await registry.pause_scope(EXPERT), "the re-read failed"
    assert env.db.rows[EXPERT.scope_key].state == ACTIVE
    assert env.scheduler.jobs == {}

    monkeypatch.setattr(env.db, "get_scope_schedule", get)
    env.scheduler.remove_scope_memory_jobs.side_effect = remove
    await env.reconcile()

    assert set(env.scheduler.jobs) == job_ids(EXPERT)


@pytest.mark.asyncio
async def test_a_gate_fire_under_a_paused_row_never_takes_crons_down(env):
    """The review's probe through the gate needed the gate to remove a
    straggler's crons, racing a resume. It no longer does: a fire under a
    PAUSED row only skips, so a resume after it keeps the crons, and the
    pass after leaves them."""
    await registry.ensure_scope_scheduled(EXPERT)
    await env.db.set_scope_state(EXPERT.owner_user_id, EXPERT.scope_key, PAUSED)

    assert not await scope_crons.memory_scope_may_fire(
        EXPERT, AsyncMock(return_value="active")
    )
    env.scheduler.remove_scope_memory_jobs.assert_not_awaited()
    await registry.resume_scope(EXPERT)
    await env.reconcile()

    assert env.db.rows[EXPERT.scope_key].state == ACTIVE
    assert set(env.scheduler.jobs) == job_ids(EXPERT)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [PAUSED, WIPED])
async def test_a_pass_removes_the_crons_of_a_scope_that_left_active(env, state):
    await registry.ensure_scope_scheduled(EXPERT)
    await registry.ensure_scope_scheduled(OTHER_EXPERT)
    await env.db.set_scope_state(EXPERT.owner_user_id, EXPERT.scope_key, state)

    report = await env.reconcile()

    assert set(env.scheduler.jobs) == job_ids(OTHER_EXPERT)
    assert report.removed == [EXPERT.scope_key]


@pytest.mark.asyncio
async def test_a_pass_removes_a_deleted_experts_crons_and_keeps_a_legacy_accounts(
    env,
):
    """An expert's crons with no row are an expert deleted; an account's are
    crons from before the registry, which the cron gate lets run."""
    for scope in (EXPERT, ACCOUNT):
        await env.scheduler.add_scope_nightly_batch_schedule(
            scope=scope, user_timezone="UTC"
        )

    report = await env.reconcile()

    assert set(env.scheduler.jobs) == {f"dream_nightly_batch_{ACCOUNT.scope_key}"}
    assert report.removed == [EXPERT.scope_key]


@pytest.mark.asyncio
async def test_a_pass_registers_only_what_the_owners_flags_enable(env):
    await registry.ensure_scope_scheduled(EXPERT)
    env.scheduler.jobs.clear()
    env.scheduler.scopes.clear()
    env.flags[Flag.GRAPHITI_COMMUNITIES_ENABLED] = False

    await env.reconcile()
    assert set(env.scheduler.jobs) == {f"dream_nightly_batch_{EXPERT.scope_key}"}

    env.flags[Flag.DREAM_PASS_ENABLED] = False
    env.scheduler.jobs.clear()
    env.scheduler.scopes.clear()
    report = await env.reconcile()

    assert env.scheduler.jobs == {}
    assert report.restored == []


@pytest.mark.asyncio
async def test_a_pass_restores_a_cron_in_the_rows_timezone(env):
    env.timezone = TOKYO
    await registry.ensure_scope_scheduled(EXPERT)
    env.timezone = PARIS  # the owner moved; the row still says Tokyo
    env.scheduler.jobs.clear()
    env.scheduler.scopes.clear()

    await env.reconcile()

    assert set(env.scheduler.jobs.values()) == {TOKYO}


@pytest.mark.asyncio
async def test_a_pause_landing_during_a_pass_wins(env, monkeypatch):
    """The listing saw the row ACTIVE and its crons gone; a pause lands before
    the pass registers them: the pass reads the row again and registers
    nothing."""
    await registry.ensure_scope_scheduled(EXPERT)
    env.scheduler.jobs.clear()
    env.scheduler.scopes.clear()
    listing = env.db.list_scope_schedules

    async def paused_after_listing(**kwargs: Any):
        page = await listing(**kwargs)
        await env.db.set_scope_state(EXPERT.owner_user_id, EXPERT.scope_key, PAUSED)
        return page

    monkeypatch.setattr(env.db, "list_scope_schedules", paused_after_listing)

    report = await env.reconcile()

    assert env.scheduler.jobs == {}
    assert report.restored == []


@pytest.mark.asyncio
async def test_a_resume_landing_during_a_pass_wins(env, monkeypatch):
    """The listing saw the row PAUSED with crons behind it; a resume lands
    before the pass removes them: the pass reads the row again and removes
    nothing."""
    await registry.ensure_scope_scheduled(EXPERT)
    await env.db.set_scope_state(EXPERT.owner_user_id, EXPERT.scope_key, PAUSED)
    listing = env.db.list_scope_schedules

    async def resumed_after_listing(**kwargs: Any):
        page = await listing(**kwargs)
        await env.db.set_scope_state(EXPERT.owner_user_id, EXPERT.scope_key, ACTIVE)
        return page

    monkeypatch.setattr(env.db, "list_scope_schedules", resumed_after_listing)

    report = await env.reconcile()

    assert set(env.scheduler.jobs) == job_ids(EXPERT)
    assert report.removed == []


@pytest.mark.asyncio
async def test_two_passes_at_once_converge(env):
    """Two schedulers reconciling at once, as in a rolling deploy, each
    listing before the other changes anything: the same end state, and
    nothing failed."""
    await registry.ensure_scope_scheduled(EXPERT)
    await registry.ensure_scope_scheduled(OTHER_EXPERT)
    env.scheduler.jobs.clear()
    env.scheduler.scopes.clear()
    await registry.ensure_scope_scheduled(ACCOUNT)
    await env.db.set_scope_state(ACCOUNT.owner_user_id, ACCOUNT.scope_key, PAUSED)
    listed = env.crons.listed()

    reports = await asyncio.gather(
        scope_reconcile.reconcile(listed, env.crons),
        scope_reconcile.reconcile(listed, env.crons),
    )

    assert set(env.scheduler.jobs) == job_ids(EXPERT) | job_ids(OTHER_EXPERT)
    assert all(report.failed == [] for report in reports)
    again = await env.reconcile()
    assert (again.restored, again.removed) == ([], [])


@pytest.mark.asyncio
async def test_a_pass_that_cannot_list_the_registry_changes_nothing(env):
    await registry.ensure_scope_scheduled(EXPERT)
    await env.db.set_scope_state(EXPERT.owner_user_id, EXPERT.scope_key, PAUSED)
    env.db.fail = True

    report = await env.reconcile()

    assert set(env.scheduler.jobs) == job_ids(EXPERT)
    assert report == scope_reconcile.ReconcileReport()


@pytest.mark.asyncio
async def test_a_failed_registration_is_reported_and_the_pass_goes_on(env):
    await registry.ensure_scope_scheduled(EXPERT)
    await registry.ensure_scope_scheduled(OTHER_EXPERT)
    env.scheduler.jobs.clear()
    env.scheduler.scopes.clear()
    add = env.scheduler.add_scope_nightly_batch_schedule.side_effect

    async def refuse_expert_one(*, scope, user_timezone):
        if scope == EXPERT:
            raise ConnectionError("jobstore down")
        return await add(scope=scope, user_timezone=user_timezone)

    env.scheduler.add_scope_nightly_batch_schedule.side_effect = refuse_expert_one

    report = await env.reconcile()

    assert report.failed == [EXPERT.scope_key]
    assert job_ids(OTHER_EXPERT) <= set(env.scheduler.jobs)


@pytest.mark.asyncio
async def test_a_pass_reads_every_page_of_the_registry(env, monkeypatch):
    monkeypatch.setattr(scope_reconcile, "PAGE_SIZE", 2)
    scopes = [registry.MemoryScope.for_expert("user-1", f"e-{i}") for i in range(5)]
    for scope in scopes:
        await registry.ensure_scope_scheduled(scope)
    env.scheduler.jobs.clear()
    env.scheduler.scopes.clear()

    await env.reconcile()

    assert set(env.scheduler.jobs) == set().union(*(job_ids(s) for s in scopes))
