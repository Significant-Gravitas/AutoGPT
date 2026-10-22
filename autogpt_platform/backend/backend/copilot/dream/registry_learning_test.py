"""Tests for the account's skill-learning cron in the registry: recorded per
account like the scope's other crons, on its own flag, never for an expert,
and paused, resumed, wiped, re-timed and reconciled under the same rules.

Fakes come from ``registry_fakes_test.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.util.feature_flag import Flag

from . import registry, registry_backfill
from .registry_fakes_test import (
    ACCOUNT,
    ACTIVE,
    EXPERT,
    PARIS,
    PAUSED,
    TOKYO,
    USER,
    WIPED,
    RegistryEnv,
    install_registry_env,
    job_ids,
    learning_job_id,
)

LEARNING = learning_job_id(ACCOUNT)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    env = install_registry_env(monkeypatch)
    env.flags[Flag.DREAM_SKILL_LEARNING_ENABLED] = True
    return env


@pytest.mark.asyncio
async def test_learning_alone_registers_and_records_the_accounts_cron(env):
    """Skill learning runs nightly per account when its flag is on, the
    dream flags off or not."""
    env.flags = {Flag.DREAM_SKILL_LEARNING_ENABLED: True}

    results = await registry.ensure_scope_scheduled(ACCOUNT)

    assert results["skill_learning_nightly"]["id"] == LEARNING
    assert results["dream_nightly_batch"] == {
        "skipped": True,
        "reason": "dream_pass_disabled",
    }
    assert env.scheduler.jobs == {f"community_rebuild_{USER}": PARIS, LEARNING: PARIS}
    row = env.db.rows[USER]
    assert (row.state, row.learning_job_id, row.nightly_job_id) == (
        ACTIVE,
        LEARNING,
        None,
    )


@pytest.mark.asyncio
async def test_an_expert_scope_never_gets_a_learning_cron(env):
    """One pass reviews the account's skills and its experts': an expert's
    scope neither registers nor even considers a learning cron."""
    results = await registry.ensure_scope_scheduled(EXPERT)

    assert set(results) == {"community_rebuild", "dream_nightly_batch"}
    assert set(env.scheduler.jobs) == job_ids(EXPERT)
    assert env.db.rows[EXPERT.scope_key].learning_job_id is None
    env.scheduler.add_scope_skill_learning_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_pause_takes_the_learning_cron_down_and_a_resume_brings_it_back(env):
    await registry.ensure_scope_scheduled(ACCOUNT)
    assert set(env.scheduler.jobs) == job_ids(ACCOUNT) | {LEARNING}

    assert await registry.pause_scope(ACCOUNT)

    row = env.db.rows[USER]
    assert (row.state, row.learning_job_id) == (PAUSED, None)
    assert env.scheduler.jobs == {}
    skipped = await registry.ensure_scope_scheduled(ACCOUNT)
    assert skipped["skill_learning_nightly"] == {
        "skipped": True,
        "reason": "scope_paused",
    }
    assert env.scheduler.jobs == {}

    await registry.resume_scope(ACCOUNT)

    row = env.db.rows[USER]
    assert (row.state, row.learning_job_id) == (ACTIVE, LEARNING)
    assert set(env.scheduler.jobs) == job_ids(ACCOUNT) | {LEARNING}


@pytest.mark.asyncio
async def test_a_wipe_takes_the_learning_cron_down_for_good(env):
    await registry.ensure_scope_scheduled(ACCOUNT)

    assert await registry.mark_wiped(ACCOUNT)

    row = env.db.rows[USER]
    assert (row.state, row.learning_job_id) == (WIPED, None)
    assert env.scheduler.jobs == {}
    await registry.resume_scope(ACCOUNT)
    await registry.ensure_scope_scheduled(ACCOUNT)
    await env.reconcile()
    assert env.db.rows[USER].state == WIPED
    assert env.scheduler.jobs == {}


@pytest.mark.asyncio
async def test_the_reconcile_restores_a_lost_learning_cron(env):
    await registry.ensure_scope_scheduled(ACCOUNT)
    del env.scheduler.jobs[LEARNING]
    del env.scheduler.scopes[LEARNING]

    report = await env.reconcile()

    assert report.restored == [LEARNING]
    assert env.scheduler.jobs[LEARNING] == PARIS
    assert LEARNING in env.scheduler.tagged


@pytest.mark.asyncio
async def test_the_reconcile_removes_a_learning_cron_straggling_past_a_pause(env):
    """A registration that lands after the pause (or a removal that failed)
    leaves the learning cron behind under a PAUSED row: the next pass takes
    it down, and its body's gate keeps it from running meanwhile."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    assert await registry.pause_scope(ACCOUNT)
    await env.scheduler.add_scope_skill_learning_schedule(
        scope=ACCOUNT, user_timezone=PARIS
    )

    report = await env.reconcile()

    assert report.removed == [USER]
    assert env.scheduler.jobs == {}
    assert env.db.rows[USER].state == PAUSED


@pytest.mark.asyncio
async def test_the_reconcile_takes_back_a_tagged_learning_cron_no_row_records(env):
    """The learning flag turned off and an ensure recorded no learning job,
    while a restore the reconcile made still runs: the next pass takes back
    its own tagged cron."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    del env.scheduler.jobs[LEARNING]
    del env.scheduler.scopes[LEARNING]
    await env.reconcile()
    assert LEARNING in env.scheduler.tagged
    env.flags[Flag.DREAM_SKILL_LEARNING_ENABLED] = False
    await registry.ensure_scope_scheduled(ACCOUNT, force_refresh=True)
    assert env.db.rows[USER].learning_job_id is None

    report = await env.reconcile()

    assert report.dropped == [LEARNING]
    assert set(env.scheduler.jobs) == job_ids(ACCOUNT)


@pytest.mark.asyncio
async def test_a_timezone_change_moves_the_learning_cron(env):
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.timezone = TOKYO

    await registry.reregister_user(USER)

    assert env.scheduler.jobs[LEARNING] == TOKYO
    assert env.db.rows[USER].learning_job_id == LEARNING


@pytest.mark.asyncio
async def test_a_timezone_change_reaches_an_owner_with_only_learning_on(env):
    """An owner whose dream flags are off but whose learning flag is on is
    not "every dream flag off": the change goes through."""
    env.flags = {Flag.DREAM_SKILL_LEARNING_ENABLED: True}
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.timezone = TOKYO

    results = await registry.reregister_user(USER)

    assert results[USER]["skill_learning_nightly"]["id"] == LEARNING
    assert env.scheduler.jobs == {f"community_rebuild_{USER}": TOKYO, LEARNING: TOKYO}


def _backfill_finds(monkeypatch: pytest.MonkeyPatch, learning_owners: list[str]):
    """The backfill's listings: no graph, no expert, no ACTIVE row; only the
    learning sources' owners, every one of them a live user."""
    module = registry_backfill.memory_schedule

    async def learners(*, after: str | None, limit: int) -> list[str]:
        return [] if after else learning_owners

    async def existing(user_ids: list[str]) -> set[str]:
        return set(user_ids)

    monkeypatch.setattr(
        registry_backfill, "list_account_graph_owners", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(module, "list_learning_account_ids", learners)
    monkeypatch.setattr(module, "existing_user_ids", existing)
    monkeypatch.setattr(
        module, "list_active_scope_schedules", AsyncMock(return_value=[])
    )
    monkeypatch.setattr(module, "list_live_expert_scopes", AsyncMock(return_value=[]))


@pytest.mark.asyncio
async def test_the_backfill_adopts_a_graphless_accounts_legacy_learning_cron(
    env, monkeypatch
):
    """The review's probe: a legacy learning cron (registered by #14551's
    capture hook, before the registry) of an account with no graph. The
    backfill finds it through the account's learning sources and records it,
    keeping the same job."""
    env.flags = {Flag.DREAM_SKILL_LEARNING_ENABLED: True}
    await env.scheduler.add_scope_skill_learning_schedule(
        scope=ACCOUNT, user_timezone=PARIS
    )
    _backfill_finds(monkeypatch, [USER])

    report = await registry_backfill.backfill_schedules()

    assert (report.accounts, report.learning_accounts) == (1, 1)
    assert env.db.rows[USER].learning_job_id == LEARNING
    assert env.scheduler.jobs == {f"community_rebuild_{USER}": PARIS, LEARNING: PARIS}


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [PAUSED, WIPED])
async def test_the_backfill_leaves_a_stopped_learning_account_stopped(
    env, monkeypatch, state
):
    await registry.ensure_scope_scheduled(ACCOUNT)
    stop = registry.pause_scope if state == PAUSED else registry.mark_wiped
    assert await stop(ACCOUNT)
    _backfill_finds(monkeypatch, [USER])

    report = await registry_backfill.backfill_schedules()

    assert report.skipped == 1
    row = env.db.rows[USER]
    assert (row.state, row.learning_job_id) == (state, None)
    assert env.scheduler.jobs == {}
