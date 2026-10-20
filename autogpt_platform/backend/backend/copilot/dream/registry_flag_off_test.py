"""Tests for a dream-system cron whose flag goes off: the registry leaves
the scope's recorded crons in place, and pause or wipe removes them. Their
bodies check the flag at run time and do nothing; the scheduler's tests pin
that (``TestSkillLearningScopeGate::test_the_flag_off_still_short_circuits``,
``TestExecuteNightlyBatchSyncRuntimeGate``,
``TestExecuteCommunityRebuildRuntimeGate``). Cleaning them up on flag-off
needs generation-aware intent, which is left for later.

Fakes come from ``registry_fakes_test.py``.
"""

from __future__ import annotations

import pytest

from backend.util.feature_flag import Flag

from . import registry, registry_backfill, scope_jobs
from .registry_fakes_test import (
    ACCOUNT,
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
NIGHTLY = f"dream_nightly_batch_{USER}"
LEARN = Flag.DREAM_SKILL_LEARNING_ENABLED
DREAM = Flag.DREAM_PASS_ENABLED


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    env = install_registry_env(monkeypatch)
    env.flags = {LEARN: True}
    return env


@pytest.mark.asyncio
async def test_turning_learning_off_leaves_its_cron_recorded(env):
    """A learning-only account: the ensure after the flag went off returns
    before it reads the row, and the reconcile keeps the cron the row
    records."""
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.flags[LEARN] = False

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert result["skill_learning_nightly"] == {
        "skipped": True,
        "reason": "skill_learning_disabled",
    }
    assert env.db.rows[USER].learning_job_id == LEARNING
    reports = [await env.reconcile() for _ in range(3)]
    assert env.scheduler.jobs == {LEARNING: PARIS}
    assert all(
        not (report.dropped or report.removed or report.restored) for report in reports
    )


@pytest.mark.asyncio
async def test_turning_learning_off_with_the_dream_on_leaves_both_recorded(env):
    env.flags[DREAM] = True
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.flags[LEARN] = False

    await registry.ensure_scope_scheduled(ACCOUNT)

    row = env.db.rows[USER]
    assert (row.nightly_job_id, row.learning_job_id) == (NIGHTLY, LEARNING)
    assert env.scheduler.jobs == {NIGHTLY: PARIS, LEARNING: PARIS}


@pytest.mark.asyncio
async def test_turning_the_dream_off_leaves_an_experts_crons_recorded(env):
    env.flags = {DREAM: True, Flag.GRAPHITI_COMMUNITIES_ENABLED: True}
    await registry.ensure_scope_scheduled(EXPERT)
    env.flags = {}

    await registry.ensure_scope_scheduled(EXPERT)

    row = env.db.rows[EXPERT.scope_key]
    assert {row.community_job_id, row.nightly_job_id} == job_ids(EXPERT)
    assert set(env.scheduler.jobs) == job_ids(EXPERT)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [PAUSED, WIPED])
async def test_a_stop_removes_the_crons_of_a_flag_turned_off(env, state):
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.flags[LEARN] = False
    await registry.ensure_scope_scheduled(ACCOUNT)

    stop = registry.pause_scope if state == PAUSED else registry.mark_wiped
    assert await stop(ACCOUNT)

    row = env.db.rows[USER]
    assert (row.state, row.learning_job_id) == (state, None)
    assert env.scheduler.jobs == {}


@pytest.mark.asyncio
async def test_every_flag_off_costs_no_database_call(env):
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.flags = {}
    env.db.fail = True

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert {r["reason"] for r in result.values()} == {
        "graphiti_communities_disabled",
        "dream_pass_disabled",
        "skill_learning_disabled",
    }
    env.db.fail = False
    assert env.db.rows[USER].learning_job_id == LEARNING


@pytest.mark.asyncio
@pytest.mark.parametrize("timezone_change", [False, True])
@pytest.mark.parametrize("failure", ["fallback", "exception"])
async def test_unavailable_flags_keep_existing_crons_during_refresh(
    env, monkeypatch, timezone_change, failure
):
    env.flags = {
        Flag.GRAPHITI_COMMUNITIES_ENABLED: True,
        DREAM: True,
        LEARN: True,
    }
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.scheduler.tagged.update(env.scheduler.jobs)
    env.scheduler.add_scope_nightly_batch_schedule.reset_mock()
    env.scheduler.add_scope_skill_learning_schedule.reset_mock()

    async def unavailable_flags(flag, user_id):
        if flag == Flag.GRAPHITI_COMMUNITIES_ENABLED:
            return True, False
        if failure == "exception":
            raise ConnectionError("flag service unavailable")
        return False, False

    monkeypatch.setattr(scope_jobs, "evaluate_feature_flag", unavailable_flags)
    if timezone_change:
        env.timezone = TOKYO

    result = await registry.ensure_scope_scheduled(
        ACCOUNT, force_refresh=not timezone_change
    )

    row = env.db.rows[USER]
    assert row.nightly_job_id == NIGHTLY
    assert row.learning_job_id == LEARNING
    reason = "registration_failed" if failure == "exception" else "flag_unavailable"
    assert result["dream_nightly_batch"]["reason"] == reason
    assert result["skill_learning_nightly"]["reason"] == reason
    env.scheduler.add_scope_nightly_batch_schedule.assert_not_awaited()
    env.scheduler.add_scope_skill_learning_schedule.assert_not_awaited()
    assert (await env.reconcile()).dropped == []
    assert set(env.scheduler.jobs) == job_ids(ACCOUNT) | {LEARNING}


@pytest.mark.asyncio
@pytest.mark.parametrize("scheduler_failure", [False, True])
async def test_fallback_enabled_cron_keeps_intent_when_scheduler_cannot_register(
    env, monkeypatch, scheduler_failure
):
    env.flags = {Flag.GRAPHITI_COMMUNITIES_ENABLED: True}
    await registry.ensure_scope_scheduled(ACCOUNT)
    community = f"community_rebuild_{USER}"
    env.scheduler.tagged.add(community)

    async def unavailable_flags(flag, user_id):
        return flag == Flag.GRAPHITI_COMMUNITIES_ENABLED, False

    monkeypatch.setattr(scope_jobs, "evaluate_feature_flag", unavailable_flags)
    add = env.scheduler.add_scope_community_rebuild_schedule
    add.side_effect = (
        ConnectionError("scheduler unavailable") if scheduler_failure else None
    )
    add.return_value = {
        "skipped": True,
        "reason": "graphiti_communities_disabled",
    }

    result = await registry.ensure_scope_scheduled(ACCOUNT, force_refresh=True)

    reason = "registration_failed" if scheduler_failure else "flag_unavailable"
    assert result["community_rebuild"]["reason"] == reason
    assert registry_backfill._classify(result) == (
        "failed" if scheduler_failure else "skipped"
    )
    assert env.db.rows[USER].community_job_id == community
    assert (await env.reconcile()).dropped == []
    assert set(env.scheduler.jobs) == {community}


@pytest.mark.asyncio
async def test_unconfigured_vendor_registers_only_community_without_backfill_failure(
    env, monkeypatch
):
    async def unavailable_flags(flag, user_id):
        return flag == Flag.GRAPHITI_COMMUNITIES_ENABLED, False

    monkeypatch.setattr(scope_jobs, "evaluate_feature_flag", unavailable_flags)

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert registry_backfill._classify(result) == "registered"
    assert result["dream_nightly_batch"]["reason"] == "flag_unavailable"
    assert result["skill_learning_nightly"]["reason"] == "flag_unavailable"
    assert set(env.scheduler.jobs) == {f"community_rebuild_{USER}"}
    env.scheduler.add_scope_nightly_batch_schedule.assert_not_awaited()
    env.scheduler.add_scope_skill_learning_schedule.assert_not_awaited()


@pytest.mark.asyncio
async def test_authoritative_off_still_clears_stale_recorded_intent(env):
    env.flags[Flag.GRAPHITI_COMMUNITIES_ENABLED] = True
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.scheduler.tagged.add(LEARNING)
    env.flags[LEARN] = False

    await registry.ensure_scope_scheduled(ACCOUNT, force_refresh=True)

    assert env.db.rows[USER].learning_job_id is None
    assert (await env.reconcile()).dropped == [LEARNING]
