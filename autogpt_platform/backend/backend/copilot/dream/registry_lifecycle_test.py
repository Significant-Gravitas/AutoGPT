"""Tests for changing a scope's lifecycle in the registry: pause (archive),
resume (revive), wipe, a new expert's registration and a timezone change.

Fakes come from ``registry_fakes_test.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from backend.util.feature_flag import Flag

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
    assert {r["reason"] for r in skipped.values()} == {
        "scope_wiped",
        "skill_learning_disabled",
    }

    resumed = await registry.resume_scope(ACCOUNT)

    assert {r["reason"] for r in resumed.values()} == {
        "scope_wiped",
        "skill_learning_disabled",
    }
    assert env.db.rows[USER].state == WIPED
    assert env.scheduler.jobs == {}


@pytest.mark.asyncio
async def test_a_resume_landing_inside_a_pause_gets_its_crons_back_next_pass(
    env, monkeypatch
):
    """The pause writes PAUSED; before its removal lands, a resume writes
    ACTIVE and registers the crons, whose job ids are the scope's own. The
    removal takes them, leaving the row ACTIVE with its ids recorded and no
    crons; the pause reports what it did, and the reconcile pass registers
    them again."""
    await registry.ensure_scope_scheduled(EXPERT)
    remove = registry.remove_scope_jobs

    async def resumed_first(scope):
        monkeypatch.setattr(registry, "remove_scope_jobs", remove)
        await registry.resume_scope(scope)
        assert set(env.scheduler.jobs) == job_ids(EXPERT)
        return await remove(scope)

    monkeypatch.setattr(registry, "remove_scope_jobs", resumed_first)

    assert await registry.pause_scope(EXPERT)
    assert env.db.rows[EXPERT.scope_key].state == ACTIVE
    assert env.scheduler.jobs == {}

    await env.reconcile()

    assert set(env.scheduler.jobs) == job_ids(EXPERT)


@pytest.mark.asyncio
async def test_a_resume_after_a_registration_found_the_scope_paused_keeps_its_crons(
    env, monkeypatch
):
    """The round-5 review's stale takedown: a registration records its crons
    after a pause landed (``scope_changed``), and a resume lands right
    after. The registration takes nothing down, so the resume's crons stay,
    recorded on the ACTIVE row, and the next reconcile pass has nothing to
    change."""
    record = env.db.record_scope_jobs
    calls = 0

    async def paused_meanwhile(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            env.db.update(EXPERT.scope_key, state=PAUSED)
            result = await record(*args, **kwargs)
            await registry.resume_scope(EXPERT)
            return result
        return await record(*args, **kwargs)

    monkeypatch.setattr(env.db, "record_scope_jobs", paused_meanwhile)

    result = await registry.ensure_scope_scheduled(EXPERT)

    assert {r["reason"] for r in result.values()} == {"scope_changed"}
    row = env.db.rows[EXPERT.scope_key]
    assert row.state == ACTIVE
    assert {row.community_job_id, row.nightly_job_id} == job_ids(EXPERT)
    assert set(env.scheduler.jobs) == job_ids(EXPERT), "the resume's crons stay"

    report = await env.reconcile()

    assert (report.restored, report.removed, report.dropped) == ([], [], [])
    assert set(env.scheduler.jobs) == job_ids(EXPERT)


@pytest.mark.asyncio
async def test_a_resume_that_cannot_write_the_row_reports_only_enabled_crons_failed(
    env, monkeypatch
):
    """The backfill lists every cron a resume reports unavailable as a
    failure: a cron the owner's flags turn off is reported as its flag says,
    as an ensure would, not as a failure."""
    env.flags[Flag.DREAM_PASS_ENABLED] = False
    monkeypatch.setattr(
        env.db, "set_scope_state", AsyncMock(side_effect=ConnectionError("db down"))
    )

    resumed = await registry.resume_scope(EXPERT)

    assert resumed == {
        "community_rebuild": {
            "skipped": True,
            "reason": "registry_unavailable",
        },
        "dream_nightly_batch": {"skipped": True, "reason": "dream_pass_disabled"},
    }


@pytest.mark.asyncio
async def test_a_revive_of_a_wiped_expert_leaves_it_wiped(env):
    await registry.mark_wiped(EXPERT)

    await registry.sync_expert_scope(USER, "expert-1", active=True)

    assert env.db.rows[EXPERT.scope_key].state == WIPED
    assert env.adds() == 0


@pytest.mark.asyncio
async def test_a_wiped_scope_stays_wiped_through_a_pause_and_a_resume(env):
    """Codex's round-2 reproduction: a pause (or archive) of a wiped expert
    used to write PAUSED over WIPED, which the next resume (or revive) then
    turned back into ACTIVE with both crons. The pause now leaves WIPED and
    only clears any cron that straggled."""
    await registry.ensure_scope_scheduled(EXPERT)
    assert await registry.mark_wiped(EXPERT)
    env.scheduler.jobs = {job_id: PARIS for job_id in job_ids(EXPERT)}

    await registry.sync_expert_scope(USER, "expert-1", active=False)

    assert env.db.rows[EXPERT.scope_key].state == WIPED
    assert env.scheduler.jobs == {}
    await registry.sync_expert_scope(USER, "expert-1", active=True)
    assert env.db.rows[EXPERT.scope_key].state == WIPED
    assert env.scheduler.jobs == {}


@pytest.mark.asyncio
async def test_a_pause_that_loses_its_claim_to_a_wipe_keeps_it_wiped(env, monkeypatch):
    """The pause finds no row, and before its claim lands a wipe creates
    the row WIPED: the pause's fallback write must not demote it either."""
    claim = env.db.claim_scope_schedule

    async def wiped_first(user_id, scope_key, expert_id, tz, state):
        await claim(user_id, scope_key, expert_id, tz, WIPED)
        return await claim(user_id, scope_key, expert_id, tz, state)

    monkeypatch.setattr(env.db, "claim_scope_schedule", wiped_first)

    assert await registry.pause_scope(EXPERT)

    assert env.db.rows[EXPERT.scope_key].state == WIPED


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
async def test_timezone_change_still_requires_the_community_registry(env):
    env.flags = {}
    env.db.fail = True  # any read would raise

    results = await registry.reregister_user(USER)

    assert set(results) == {USER}
    assert {r["reason"] for r in results[USER].values()} == {
        "registry_unavailable",
        "dream_pass_disabled",
        "skill_learning_disabled",
    }


@pytest.mark.asyncio
async def test_timezone_change_re_registers_a_legacy_account_without_row(env):
    results = await registry.reregister_user(USER)

    assert list(results) == [USER]
    assert set(env.scheduler.jobs) == job_ids(ACCOUNT)
