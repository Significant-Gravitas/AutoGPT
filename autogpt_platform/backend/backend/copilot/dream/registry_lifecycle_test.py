"""Tests for changing a scope's lifecycle in the registry: pause (archive),
resume (revive), wipe, a new expert's registration and a timezone change.

Fakes come from ``registry_fakes_test.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from . import registry
from .registry_fakes_test import (
    ACCOUNT,
    ACTIVE,
    EXPERT,
    OTHER_EXPERT,
    PARIS,
    PAUSED,
    TOKYO,
    USER,
    WIPED,
    RegistryEnv,
    install_registry_env,
    job_ids,
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    return install_registry_env(monkeypatch)


@pytest.mark.asyncio
async def test_archive_pauses_and_revive_resumes_the_expert(env):
    await registry.ensure_expert_scheduled(USER, "expert-1")  # hire
    assert set(env.scheduler.jobs) == job_ids(EXPERT)

    await registry.sync_expert_scope(USER, "expert-1", active=False)  # archive
    row = env.db.rows[EXPERT.scope_key]
    assert (row.state, row.community_job_id, row.nightly_job_id) == (PAUSED, None, None)
    assert env.scheduler.jobs == {}

    await registry.sync_expert_scope(USER, "expert-1", active=True)  # re-hire
    row = env.db.rows[EXPERT.scope_key]
    assert row.state == ACTIVE
    assert {row.community_job_id, row.nightly_job_id} == job_ids(EXPERT)
    assert set(env.scheduler.jobs) == job_ids(EXPERT)
    assert list(env.db.rows) == [EXPERT.scope_key]


@pytest.mark.asyncio
async def test_a_new_expert_registration_never_resumes_a_paused_scope(env):
    """A hire registers in the background and may land after an archive or a
    pause: it must leave the scope paused, unlike an explicit resume."""
    await registry.pause_scope(EXPERT)

    await registry.ensure_expert_scheduled(USER, "expert-1")

    assert env.db.rows[EXPERT.scope_key].state == PAUSED
    assert env.scheduler.jobs == {}
    assert env.adds() == 0


@pytest.mark.asyncio
async def test_the_lifecycle_hook_never_fails_its_caller(env, monkeypatch):
    """Hire, archive, pause and resume call in; nothing in memory
    scheduling may fail them — not an invalid id, not a bug."""
    monkeypatch.setattr(
        registry, "pause_scope", AsyncMock(side_effect=RuntimeError("bug"))
    )

    await registry.sync_expert_scope(USER, "expert-1", active=False)
    await registry.sync_expert_scope(USER, "", active=True)
    await registry.ensure_expert_scheduled(USER, "")

    assert env.db.rows == {}


@pytest.mark.asyncio
async def test_pausing_a_scope_without_a_row_records_it_paused(env):
    assert await registry.pause_scope(EXPERT)

    row = env.db.rows[EXPERT.scope_key]
    assert (row.state, row.timezone, row.expert_id) == (PAUSED, PARIS, "expert-1")
    env.scheduler.remove_scope_memory_jobs.assert_awaited_once_with(scope=EXPERT)


@pytest.mark.asyncio
async def test_a_failed_pause_write_leaves_the_crons_alone(env):
    await registry.ensure_scope_scheduled(EXPERT)
    env.db.fail = True

    assert not await registry.pause_scope(EXPERT)

    env.scheduler.remove_scope_memory_jobs.assert_not_awaited()
    assert set(env.scheduler.jobs) == job_ids(EXPERT)


@pytest.mark.asyncio
async def test_a_pause_whose_job_removal_fails_reports_false(env):
    """The row says PAUSED, so the cron gate already holds the jobs back,
    but the pause did not finish and must say so."""
    await registry.ensure_scope_scheduled(EXPERT)
    env.scheduler.remove_scope_memory_jobs.side_effect = RuntimeError("down")

    assert not await registry.pause_scope(EXPERT)

    assert env.db.rows[EXPERT.scope_key].state == PAUSED


@pytest.mark.asyncio
async def test_a_pause_racing_a_first_registration_still_ends_paused(env, monkeypatch):
    """A registration claims the row ACTIVE between the pause's first write
    (no row yet) and its claim: the pause must set PAUSED explicitly."""
    claim = env.db.claim_scope_schedule

    async def claimed_active_first(user_id, scope_key, expert_id, tz, state=ACTIVE):
        await claim(user_id, scope_key, expert_id, tz)  # the registration wins
        return await claim(user_id, scope_key, expert_id, tz, state)

    monkeypatch.setattr(env.db, "claim_scope_schedule", claimed_active_first)

    assert await registry.pause_scope(EXPERT)

    assert env.db.rows[EXPERT.scope_key].state == PAUSED
    env.scheduler.remove_scope_memory_jobs.assert_awaited_once_with(scope=EXPERT)


@pytest.mark.asyncio
async def test_a_pause_whose_claim_fails_still_sets_the_state(env, monkeypatch):
    """A claim that fails after a concurrent registration created the row is
    not the end: the pause re-reads by writing the state itself."""
    claim = env.db.claim_scope_schedule

    async def created_elsewhere_then_failed(user_id, scope_key, expert_id, tz, state):
        await claim(user_id, scope_key, expert_id, tz)
        raise ConnectionError("lost the response")

    monkeypatch.setattr(env.db, "claim_scope_schedule", created_elsewhere_then_failed)

    assert await registry.pause_scope(EXPERT)

    assert env.db.rows[EXPERT.scope_key].state == PAUSED


@pytest.mark.asyncio
async def test_wipe_removes_the_jobs_and_resume_does_not_undo_it(env):
    await registry.ensure_scope_scheduled(ACCOUNT)

    assert await registry.mark_wiped(ACCOUNT)

    assert env.db.rows[USER].state == WIPED
    assert env.scheduler.jobs == {}
    skipped = await registry.ensure_scope_scheduled(ACCOUNT)
    assert {r["reason"] for r in skipped.values()} == {"scope_wiped"}

    resumed = await registry.resume_scope(ACCOUNT)

    assert {r["reason"] for r in resumed.values()} == {"scope_wiped"}
    assert env.db.rows[USER].state == WIPED
    assert env.scheduler.jobs == {}


@pytest.mark.asyncio
async def test_a_revive_of_a_wiped_expert_leaves_it_wiped(env):
    await registry.mark_wiped(EXPERT)

    await registry.sync_expert_scope(USER, "expert-1", active=True)

    assert env.db.rows[EXPERT.scope_key].state == WIPED
    assert env.adds() == 0


@pytest.mark.asyncio
async def test_wiping_an_account_without_a_row_removes_its_legacy_crons(env):
    env.scheduler.jobs = {job_id: PARIS for job_id in job_ids(ACCOUNT)}

    assert await registry.mark_wiped(ACCOUNT)

    assert env.db.rows[USER].state == WIPED
    assert env.scheduler.jobs == {}


@pytest.mark.asyncio
async def test_timezone_change_re_registers_every_active_scope(env):
    await registry.ensure_scope_scheduled(ACCOUNT)
    await registry.ensure_scope_scheduled(EXPERT)
    await registry.pause_scope(OTHER_EXPERT)
    env.timezone = TOKYO

    results = await registry.reregister_user(USER)

    assert set(results) == {
        ACCOUNT.scope_key,
        EXPERT.scope_key,
        OTHER_EXPERT.scope_key,
    }
    paused = results[OTHER_EXPERT.scope_key]
    assert {r["reason"] for r in paused.values()} == {"scope_paused"}
    assert set(env.scheduler.jobs) == job_ids(ACCOUNT) | job_ids(EXPERT)
    assert set(env.scheduler.jobs.values()) == {TOKYO}
    assert env.db.rows[EXPERT.scope_key].timezone == TOKYO


@pytest.mark.asyncio
async def test_timezone_change_of_a_flag_off_owner_costs_no_database_call(env):
    env.flags = {}
    env.db.fail = True  # any read would raise

    results = await registry.reregister_user(USER)

    assert set(results) == {USER}
    assert {r["reason"] for r in results[USER].values()} == {
        "graphiti_communities_disabled",
        "dream_pass_disabled",
    }


@pytest.mark.asyncio
async def test_timezone_change_re_registers_a_legacy_account_without_row(env):
    results = await registry.reregister_user(USER)

    assert list(results) == [USER]
    assert set(env.scheduler.jobs) == job_ids(ACCOUNT)
