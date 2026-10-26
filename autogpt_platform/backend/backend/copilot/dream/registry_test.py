"""Tests for registering a scope's crons: ``ensure_scope_scheduled`` and the
work it hands to ``scope_jobs.py``.

The data module and the scheduler client are in-memory fakes with the real
ownership rules (``registry_fakes_test.py``). Lifecycle changes are in
``registry_lifecycle_test.py``, deadlines in ``registry_deadline_test.py``.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from backend.util.feature_flag import Flag

from . import registry, registry_backfill, scope_crons, scope_jobs
from .registry_fakes_test import (
    ACCOUNT,
    ACTIVE,
    EXPERT,
    PARIS,
    PAUSED,
    TOKYO,
    USER,
    RegistryEnv,
    install_registry_env,
    job_ids,
)


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    return install_registry_env(monkeypatch)


@pytest.mark.asyncio
async def test_community_registers_without_a_flag_provider(env, monkeypatch):
    flag = AsyncMock(return_value=(False, False))
    monkeypatch.setattr(scope_jobs, "evaluate_feature_flag", flag)
    env.flags = {}

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert result["community_rebuild"]["id"] == f"community_rebuild_{USER}"
    assert env.db.rows[USER].nightly_job_id is None
    assert env.db.rows[USER].learning_job_id is None
    assert result["dream_nightly_batch"]["reason"] == "flag_unavailable"
    assert result["skill_learning_nightly"]["reason"] == "flag_unavailable"
    assert registry_backfill._classify(result) == "registered"
    assert [call.args[0] for call in flag.await_args_list] == [
        Flag.DREAM_PASS_ENABLED,
        Flag.DREAM_SKILL_LEARNING_ENABLED,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_raises", [False, True])
async def test_community_refresh_preserves_unknown_dream_and_learning_intent(
    env, monkeypatch, provider_raises
):
    env.flags[Flag.DREAM_SKILL_LEARNING_ENABLED] = True
    await registry.ensure_scope_scheduled(ACCOUNT)
    for job_id in (f"dream_nightly_batch_{USER}", f"skill_learning_nightly_{USER}"):
        del env.scheduler.jobs[job_id]
        del env.scheduler.scopes[job_id]
    await env.reconcile()
    assert len(env.scheduler.tagged) == 2
    env.timezone = TOKYO
    monkeypatch.setattr(
        scope_jobs,
        "evaluate_feature_flag",
        AsyncMock(
            return_value=(False, False),
            side_effect=(
                RuntimeError("provider unavailable") if provider_raises else None
            ),
        ),
    )
    env.flags = {}

    result = await registry.ensure_scope_scheduled(ACCOUNT, force_refresh=True)

    row = env.db.rows[USER]
    assert result["dream_nightly_batch"]["reason"] == "flag_unavailable"
    assert result["skill_learning_nightly"]["reason"] == "flag_unavailable"
    assert row.nightly_job_id == f"dream_nightly_batch_{USER}"
    assert row.learning_job_id == f"skill_learning_nightly_{USER}"
    report = await env.reconcile()
    assert report.dropped == []
    assert set(env.scheduler.jobs.values()) == {TOKYO}


@pytest.mark.asyncio
async def test_ensure_registers_both_crons_under_the_legacy_account_ids(env):
    result = await registry.ensure_scope_scheduled(ACCOUNT)

    # The account's scope key is its user id: job ids did not move.
    assert result["community_rebuild"]["id"] == f"community_rebuild_{USER}"
    assert result["dream_nightly_batch"]["id"] == f"dream_nightly_batch_{USER}"
    env.scheduler.add_scope_nightly_batch_schedule.assert_awaited_once_with(
        scope=ACCOUNT, user_timezone=PARIS
    )
    row = env.db.rows[USER]
    assert (row.state, row.timezone, row.expert_id) == (ACTIVE, PARIS, None)
    assert {row.community_job_id, row.nightly_job_id} == job_ids(ACCOUNT)


@pytest.mark.asyncio
async def test_ensure_is_idempotent(env):
    await registry.ensure_scope_scheduled(ACCOUNT)
    again = await registry.ensure_scope_scheduled(ACCOUNT)

    assert again == {
        "community_rebuild": None,
        "dream_nightly_batch": None,
        "skill_learning_nightly": {
            "skipped": True,
            "reason": "skill_learning_disabled",
        },
    }
    assert env.adds() == 2


@pytest.mark.asyncio
async def test_exactly_one_row_per_scope(env):
    for _ in range(3):
        await registry.ensure_scope_scheduled(ACCOUNT)
        await registry.ensure_scope_scheduled(EXPERT)

    assert sorted(env.db.rows) == sorted([ACCOUNT.scope_key, EXPERT.scope_key])
    assert env.db.rows[EXPERT.scope_key].expert_id == "expert-1"
    assert set(env.scheduler.jobs) == job_ids(ACCOUNT) | job_ids(EXPERT)
    assert env.adds() == 4


@pytest.mark.asyncio
async def test_an_expert_write_registers_the_expert_not_the_account(env):
    result = await registry.ensure_scope_scheduled(EXPERT)

    nightly = f"dream_nightly_batch_{EXPERT.scope_key}"
    assert result["dream_nightly_batch"]["id"] == nightly
    assert ACCOUNT.scope_key not in env.db.rows


@pytest.mark.asyncio
async def test_other_flags_off_still_requires_community_registry(env):
    env.flags = {}
    env.db.fail = True  # any read would raise

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert result == {
        "community_rebuild": {
            "skipped": True,
            "reason": "registry_unavailable",
        },
        "dream_nightly_batch": {"skipped": True, "reason": "dream_pass_disabled"},
        "skill_learning_nightly": {
            "skipped": True,
            "reason": "skill_learning_disabled",
        },
    }
    assert env.adds() == 0


@pytest.mark.asyncio
async def test_one_flag_off_still_registers_the_other(env):
    env.flags[Flag.DREAM_PASS_ENABLED] = False

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert result["dream_nightly_batch"]["reason"] == "dream_pass_disabled"
    assert env.db.rows[USER].community_job_id == f"community_rebuild_{USER}"
    assert env.db.rows[USER].nightly_job_id is None


@pytest.mark.asyncio
async def test_a_paused_scope_is_never_registered_by_ensure(env):
    await registry.pause_scope(EXPERT)

    result = await registry.ensure_scope_scheduled(EXPERT)

    assert all(
        r == {"skipped": True, "reason": "scope_paused"} for r in result.values()
    )
    assert env.adds() == 0


@pytest.mark.asyncio
async def test_timezone_change_re_registers_in_the_new_zone(env):
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.timezone = TOKYO

    await registry.ensure_scope_scheduled(ACCOUNT)

    assert env.adds() == 4
    assert env.db.rows[USER].timezone == TOKYO
    assert set(env.scheduler.jobs.values()) == {TOKYO}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope,prefix",
    [
        (ACCOUNT, "community_rebuild"),
        (EXPERT, "community_rebuild"),
        (ACCOUNT, "dream_nightly_batch"),
        (EXPERT, "dream_nightly_batch"),
        (ACCOUNT, "skill_learning_nightly"),
    ],
)
async def test_failed_timezone_refresh_keeps_the_cron_repairable(env, scope, prefix):
    env.flags[Flag.DREAM_SKILL_LEARNING_ENABLED] = True
    await registry.ensure_scope_scheduled(scope)
    add = {
        "community_rebuild": env.scheduler.add_scope_community_rebuild_schedule,
        "dream_nightly_batch": env.scheduler.add_scope_nightly_batch_schedule,
        "skill_learning_nightly": env.scheduler.add_scope_skill_learning_schedule,
    }[prefix]
    adder = add.side_effect
    add.side_effect = RuntimeError("scheduler down")
    env.timezone = TOKYO

    result = await registry.ensure_scope_scheduled(scope)

    job_id = f"{prefix}_{scope.scope_key}"
    assert result[prefix]["reason"] == "registration_failed"
    assert env.db.rows[scope.scope_key].timezone == TOKYO
    assert env.scheduler.jobs[job_id] == PARIS
    assert job_id not in env.scheduler.tagged
    add.side_effect = adder

    report = await env.reconcile()

    assert report.restored == [job_id]
    assert report.failed == []
    assert env.scheduler.jobs[job_id] == TOKYO
    assert (await env.reconcile()).restored == []


@pytest.mark.asyncio
async def test_unknown_timezone_is_not_utc_and_changes_nothing(env):
    env.timezone = None

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert {r["reason"] for r in result.values()} == {
        "timezone_lookup_failed",
        "skill_learning_disabled",
    }
    assert env.db.rows == {}
    assert env.adds() == 0


@pytest.mark.asyncio
async def test_registry_read_failure_skips_the_cycle(env):
    env.db.fail = True

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert {r["reason"] for r in result.values()} == {
        "registry_unavailable",
        "skill_learning_disabled",
    }
    assert env.adds() == 0


@pytest.mark.asyncio
async def test_a_failed_registration_is_not_recorded_and_is_retried(env):
    add = env.scheduler.add_scope_community_rebuild_schedule
    adder = add.side_effect
    add.side_effect = RuntimeError("scheduler down")

    first = await registry.ensure_scope_scheduled(ACCOUNT)

    assert first["community_rebuild"] == {
        "skipped": True,
        "reason": "registration_failed",
    }
    assert first["dream_nightly_batch"]["id"] == f"dream_nightly_batch_{USER}"
    assert env.db.rows[USER].community_job_id is None

    add.side_effect = adder
    second = await registry.ensure_scope_scheduled(ACCOUNT)

    assert second["community_rebuild"]["id"] == f"community_rebuild_{USER}"
    assert second["dream_nightly_batch"] is None


@pytest.mark.asyncio
async def test_a_registration_the_row_cannot_record_is_a_failure(env, monkeypatch):
    """The jobs exist but the row does not know them: report it, so the
    backfill counts the scope failed, and the next ensure registers again."""

    async def record_fails(*args, **kwargs):
        raise ConnectionError("db down")

    monkeypatch.setattr(env.db, "record_scope_jobs", record_fails)

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert {r["reason"] for r in result.values()} == {
        "record_failed",
        "skill_learning_disabled",
    }
    assert env.db.rows[USER].nightly_job_id is None


@pytest.mark.asyncio
async def test_a_scheduler_side_flag_skip_is_not_recorded(env, caplog):
    skipped = {"id": None, "skipped": True, "reason": "dream_pass_disabled"}
    env.scheduler.add_scope_nightly_batch_schedule.side_effect = None
    env.scheduler.add_scope_nightly_batch_schedule.return_value = skipped

    with caplog.at_level(logging.WARNING, logger=scope_jobs.logger.name):
        result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert result["dream_nightly_batch"] == skipped
    assert env.db.rows[USER].nightly_job_id is None
    assert any("despite the local flag check" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_pause_that_lands_mid_registration_leaves_its_crons_to_the_reconcile(
    env,
):
    """The registration finds the scope paused when it records its crons: it
    reports ``scope_changed`` and takes nothing down. The fire gate holds
    the paused scope, and the next reconcile pass removes its crons."""
    add = env.scheduler.add_scope_nightly_batch_schedule
    adder = add.side_effect

    async def add_then_archive(*, scope, user_timezone):
        added = await adder(scope=scope, user_timezone=user_timezone)
        env.db.update(scope.scope_key, state=PAUSED)
        return added

    add.side_effect = add_then_archive
    result = await registry.ensure_scope_scheduled(EXPERT)

    # Not a failure: the scope simply stopped wanting its crons.
    assert {r["reason"] for r in result.values()} == {"scope_changed"}
    env.scheduler.remove_scope_memory_jobs.assert_not_awaited()
    assert set(env.scheduler.jobs) == job_ids(EXPERT)
    assert not await scope_crons.memory_scope_may_fire(
        EXPERT, AsyncMock(return_value="active")
    )

    report = await env.reconcile()

    assert report.removed == [EXPERT.scope_key]
    assert env.scheduler.jobs == {}


@pytest.mark.asyncio
async def test_ensure_dream_system_scheduled_is_the_account_scope(env):
    await registry.ensure_dream_system_scheduled(USER)

    assert list(env.db.rows) == [USER]
    assert await registry.ensure_dream_system_scheduled("") == {}


@pytest.mark.asyncio
async def test_timezone_is_durable_before_scheduler_replacement(env, monkeypatch):
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.timezone = TOKYO
    record = env.db.record_scope_jobs
    calls = 0

    async def fail_final_record(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ConnectionError("final record failed")
        return await record(*args, **kwargs)

    monkeypatch.setattr(env.db, "record_scope_jobs", fail_final_record)
    result = await registry.ensure_scope_scheduled(ACCOUNT)
    assert calls == 2
    assert env.db.rows[USER].timezone == TOKYO
    assert result["community_rebuild"]["reason"] == "record_failed"
    assert env.db.rows[USER].community_job_id == f"community_rebuild_{USER}"


@pytest.mark.asyncio
async def test_failed_timezone_intent_does_not_replace_crons(env, monkeypatch):
    await registry.ensure_scope_scheduled(ACCOUNT)
    env.timezone = TOKYO
    env.scheduler.add_scope_community_rebuild_schedule.reset_mock()
    env.scheduler.add_scope_nightly_batch_schedule.reset_mock()
    monkeypatch.setattr(
        env.db, "record_scope_jobs", AsyncMock(side_effect=ConnectionError("db down"))
    )
    result = await registry.ensure_scope_scheduled(ACCOUNT)
    assert result["community_rebuild"]["reason"] == "record_failed"
    assert env.db.rows[USER].timezone == PARIS
    env.scheduler.add_scope_community_rebuild_schedule.assert_not_awaited()
    env.scheduler.add_scope_nightly_batch_schedule.assert_not_awaited()
