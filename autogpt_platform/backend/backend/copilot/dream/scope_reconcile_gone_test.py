"""Tests for an account's dream-system crons with no registry row once its
user is gone: the reconcile removes them, keeps a live account's for the
registry to adopt, and leaves both when it cannot tell; the cron gate skips
the gone account's. No flag is read.

The probe is an independent review's: a deleted user's untagged legacy
learning cron was kept for good, and its gate let it run. Fakes come from
``registry_fakes_test.py``; ``backend/data/memory_schedule_db_test.py``
runs the probe with a real user, scheduler and jobstore.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.copilot.graphiti.scope import MemoryScope

from . import registry, scope_crons, scope_reconcile
from .registry_fakes_test import (
    ACCOUNT,
    EXPERT,
    PARIS,
    RegistryEnv,
    install_registry_env,
)
from .scheduling import DREAM_SYSTEM_JOBS

LIVE = MemoryScope.for_user("live-user")
GONE = MemoryScope.for_user("gone-user")


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    return install_registry_env(monkeypatch)


async def _legacy_crons(env: RegistryEnv, *scopes: MemoryScope) -> None:
    """Crons from before the registry: registered untagged, no row."""
    for scope in scopes:
        for job in DREAM_SYSTEM_JOBS:
            await job.register(env.scheduler, scope, PARIS)


def _ids(scope: MemoryScope) -> set[str]:
    return {job.job_id(scope) for job in DREAM_SYSTEM_JOBS}


@pytest.mark.asyncio
async def test_a_gone_accounts_crons_go_and_a_live_ones_stay(env):
    await _legacy_crons(env, LIVE, GONE)
    env.db.gone_users = {GONE.owner_user_id}

    report = await env.reconcile()

    assert report.removed == [GONE.scope_key]
    assert set(env.scheduler.jobs) == _ids(LIVE)
    assert env.db.rows == {}


@pytest.mark.asyncio
async def test_an_unanswered_existence_check_leaves_them_for_a_later_pass(env):
    await _legacy_crons(env, LIVE, GONE)
    env.db.gone_users = {GONE.owner_user_id}
    env.db.existence_fails = True

    report = await env.reconcile()

    assert (report.removed, report.failed) == ([], [])
    assert set(env.scheduler.jobs) == _ids(LIVE) | _ids(GONE)

    env.db.existence_fails = False
    report = await env.reconcile()

    assert report.removed == [GONE.scope_key]
    assert set(env.scheduler.jobs) == _ids(LIVE)


@pytest.mark.asyncio
async def test_the_check_asks_about_rowless_accounts_only_a_page_at_a_time(
    env, monkeypatch
):
    """One query per page of owners, none about an account with a row or an
    expert, whose crons with no row go whatever its owner."""
    monkeypatch.setattr(scope_reconcile, "PAGE_SIZE", 2)
    rowless = [MemoryScope.for_user(f"legacy-{n}") for n in range(5)]
    await _legacy_crons(env, *rowless)
    await registry.ensure_scope_scheduled(ACCOUNT)
    await env.scheduler.add_scope_nightly_batch_schedule(
        scope=EXPERT, user_timezone=PARIS
    )
    asked: list[list[str]] = []
    existing = env.db.existing_user_ids

    async def counted(user_ids: list[str]) -> set[str]:
        asked.append(user_ids)
        return await existing(user_ids)

    monkeypatch.setattr(env.db, "existing_user_ids", counted)

    report = await env.reconcile()

    owners = sorted(scope.owner_user_id for scope in rowless)
    assert asked == [owners[0:2], owners[2:4], owners[4:5]]
    assert report.removed == [EXPERT.scope_key]


@pytest.mark.asyncio
async def test_the_gate_skips_a_gone_accounts_cron_and_lets_a_live_ones_run(env):
    env.db.gone_users = {GONE.owner_user_id}
    status = AsyncMock(return_value="active")

    assert not await scope_crons.memory_scope_may_fire(GONE, status)
    assert await scope_crons.memory_scope_may_fire(LIVE, status)


@pytest.mark.asyncio
async def test_a_gate_that_cannot_tell_skips_the_run(env):
    env.db.existence_fails = True

    assert not await scope_crons.memory_scope_may_fire(
        LIVE, AsyncMock(return_value="active")
    )


@pytest.mark.asyncio
async def test_an_account_with_a_row_is_not_asked_about(env):
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.db.existence_fails = True

    assert await scope_crons.memory_scope_may_fire(
        ACCOUNT, AsyncMock(return_value="active")
    )
    report = await env.reconcile()
    assert (report.removed, report.failed) == ([], [])
