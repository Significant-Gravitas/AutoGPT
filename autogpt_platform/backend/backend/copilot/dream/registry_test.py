"""Tests for the per-scope schedule registry (``registry.py`` + ``scope_jobs.py``).

The data module and the scheduler client are in-memory fakes with the real
ownership rules, so each test reads as the lifecycle it pins: a scope is
registered once and stays one row, archive pauses it, revive resumes it, a
wipe takes its crons down, and a timezone change moves every active scope.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from prisma.enums import MemoryScopeScheduleState

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.memory_schedule import MemoryScopeSchedule
from backend.util.exceptions import NotAuthorizedError
from backend.util.feature_flag import Flag

from . import registry, scope_jobs
from .scheduling import dream_system_job

ACTIVE = MemoryScopeScheduleState.ACTIVE
PAUSED = MemoryScopeScheduleState.PAUSED
WIPED = MemoryScopeScheduleState.WIPED

USER = "user-1"
ACCOUNT = MemoryScope.for_user(USER)
EXPERT = MemoryScope.for_expert(USER, "expert-1")
OTHER_EXPERT = MemoryScope.for_expert(USER, "expert-2")
PARIS = "Europe/Paris"
TOKYO = "Asia/Tokyo"


def _job_ids(scope: MemoryScope) -> set[str]:
    return {
        f"community_rebuild_{scope.scope_key}",
        f"dream_nightly_batch_{scope.scope_key}",
    }


class FakeRegistryDB:
    """``backend.data.memory_schedule`` in memory, with its ownership rules."""

    def __init__(self) -> None:
        self.rows: dict[str, MemoryScopeSchedule] = {}
        self.fail = False

    async def get_scope_schedule(
        self, user_id: str, scope_key: str
    ) -> MemoryScopeSchedule | None:
        if self.fail:
            raise ConnectionError("db down")
        row = self.rows.get(scope_key)
        return row if row is not None and row.user_id == user_id else None

    async def claim_scope_schedule(
        self,
        user_id: str,
        scope_key: str,
        expert_id: str | None,
        user_timezone: str,
        state: MemoryScopeScheduleState = ACTIVE,
    ) -> MemoryScopeSchedule:
        if scope_key not in self.rows:
            now = datetime.now(timezone.utc)
            self.rows[scope_key] = MemoryScopeSchedule(
                scope_key=scope_key,
                user_id=user_id,
                expert_id=expert_id,
                timezone=user_timezone,
                state=state,
                community_job_id=None,
                nightly_job_id=None,
                last_nightly_run_at=None,
                last_community_run_at=None,
                created_at=now,
                updated_at=now,
            )
        if self.rows[scope_key].user_id != user_id:
            raise NotAuthorizedError("another user's scope")
        return self.rows[scope_key]

    async def record_scope_jobs(
        self,
        user_id: str,
        scope_key: str,
        *,
        user_timezone: str,
        community_job_id: str | None,
        nightly_job_id: str | None,
    ) -> bool:
        row = await self.get_scope_schedule(user_id, scope_key)
        if row is None or row.state != ACTIVE:
            return False
        self.update(
            scope_key,
            timezone=user_timezone,
            community_job_id=community_job_id,
            nightly_job_id=nightly_job_id,
        )
        return True

    async def set_scope_state(
        self, user_id: str, scope_key: str, state: MemoryScopeScheduleState
    ) -> bool:
        if await self.get_scope_schedule(user_id, scope_key) is None:
            return False
        fields: dict[str, Any] = {"state": state}
        if state != ACTIVE:
            fields |= {"community_job_id": None, "nightly_job_id": None}
        self.update(scope_key, **fields)
        return True

    async def forget_scope_job(self, user_id: str, scope_key: str, job_id: str) -> bool:
        row = await self.get_scope_schedule(user_id, scope_key)
        if row is None:
            return False
        fields: dict[str, Any] = {}
        if row.community_job_id == job_id:
            fields["community_job_id"] = None
        if row.nightly_job_id == job_id:
            fields["nightly_job_id"] = None
        self.update(scope_key, **fields)
        return bool(fields)

    async def record_scope_run(self, user_id: str, scope_key: str, kind: str) -> bool:
        if self.fail:
            raise ConnectionError("db down")
        if await self.get_scope_schedule(user_id, scope_key) is None:
            return False
        field = "last_nightly_run_at" if kind == "nightly" else "last_community_run_at"
        self.update(scope_key, **{field: datetime.now(timezone.utc)})
        return True

    async def list_user_scope_schedules(
        self, user_id: str
    ) -> list[MemoryScopeSchedule]:
        return [row for row in self.rows.values() if row.user_id == user_id]

    def update(self, scope_key: str, **fields: Any) -> None:
        self.rows[scope_key] = self.rows[scope_key].model_copy(update=fields)


class FakeScheduler:
    """The three memory-cron RPCs of ``SchedulerClient``, in memory."""

    def __init__(self) -> None:
        self.jobs: dict[str, str] = {}  # job id -> timezone it runs in
        self.add_scope_community_rebuild_schedule = AsyncMock(
            side_effect=self._adder("community_rebuild")
        )
        self.add_scope_nightly_batch_schedule = AsyncMock(
            side_effect=self._adder("dream_nightly_batch")
        )
        self.remove_scope_memory_jobs = AsyncMock(side_effect=self._remove)

    def _adder(self, prefix: str):
        async def add(*, scope: MemoryScope, user_timezone: str) -> dict:
            job_id = f"{prefix}_{scope.scope_key}"
            self.jobs[job_id] = user_timezone
            return {"id": job_id, "scope_key": scope.scope_key, "next_run_time": None}

        return add

    async def _remove(self, *, scope: MemoryScope) -> list[str]:
        removed = [job_id for job_id in _job_ids(scope) if job_id in self.jobs]
        for job_id in removed:
            del self.jobs[job_id]
        return removed


class RegistryEnv:
    def __init__(self) -> None:
        self.db = FakeRegistryDB()
        self.scheduler = FakeScheduler()
        self.flags = {
            Flag.GRAPHITI_COMMUNITIES_ENABLED: True,
            Flag.DREAM_PASS_ENABLED: True,
        }
        self.timezone: str | None = PARIS

    def adds(self) -> int:
        return (
            self.scheduler.add_scope_community_rebuild_schedule.await_count
            + self.scheduler.add_scope_nightly_batch_schedule.await_count
        )


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    state = RegistryEnv()

    async def flag(flag: Flag, user_id: str) -> bool:
        return state.flags.get(flag, False)

    async def owner_timezone(user_id: str) -> str | None:
        return state.timezone

    monkeypatch.setattr(registry, "memory_schedule_db", lambda: state.db)
    monkeypatch.setattr(scope_jobs, "memory_schedule_db", lambda: state.db)
    monkeypatch.setattr(scope_jobs, "get_scheduler_client", lambda: state.scheduler)
    monkeypatch.setattr(scope_jobs, "is_feature_enabled", flag)
    monkeypatch.setattr(registry, "resolve_user_timezone", owner_timezone)
    return state


# ---------------------------------------------------------------------------
# ensure_scope_scheduled
# ---------------------------------------------------------------------------


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
    assert {row.community_job_id, row.nightly_job_id} == _job_ids(ACCOUNT)


@pytest.mark.asyncio
async def test_ensure_is_idempotent(env):
    await registry.ensure_scope_scheduled(ACCOUNT)
    again = await registry.ensure_scope_scheduled(ACCOUNT)

    assert again == {"community_rebuild": None, "dream_nightly_batch": None}
    assert env.adds() == 2


@pytest.mark.asyncio
async def test_exactly_one_row_per_scope(env):
    for _ in range(3):
        await registry.ensure_scope_scheduled(ACCOUNT)
        await registry.ensure_scope_scheduled(EXPERT)

    assert sorted(env.db.rows) == sorted([ACCOUNT.scope_key, EXPERT.scope_key])
    assert env.db.rows[EXPERT.scope_key].expert_id == "expert-1"
    assert set(env.scheduler.jobs) == _job_ids(ACCOUNT) | _job_ids(EXPERT)
    assert env.adds() == 4


@pytest.mark.asyncio
async def test_an_expert_write_registers_the_expert_not_the_account(env):
    result = await registry.ensure_scope_scheduled(EXPERT)

    nightly = f"dream_nightly_batch_{EXPERT.scope_key}"
    assert result["dream_nightly_batch"]["id"] == nightly
    assert ACCOUNT.scope_key not in env.db.rows


@pytest.mark.asyncio
async def test_flags_off_cost_no_read_and_no_rpc(env):
    env.flags = {}
    env.db.fail = True  # any read would raise

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert result == {
        "community_rebuild": {
            "skipped": True,
            "reason": "graphiti_communities_disabled",
        },
        "dream_nightly_batch": {"skipped": True, "reason": "dream_pass_disabled"},
    }
    assert env.adds() == 0


@pytest.mark.asyncio
async def test_one_flag_off_still_registers_the_other(env):
    env.flags[Flag.GRAPHITI_COMMUNITIES_ENABLED] = False

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert result["community_rebuild"]["reason"] == "graphiti_communities_disabled"
    assert env.db.rows[USER].nightly_job_id == f"dream_nightly_batch_{USER}"
    assert env.db.rows[USER].community_job_id is None


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
async def test_unknown_timezone_is_not_utc_and_changes_nothing(env):
    env.timezone = None

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert {r["reason"] for r in result.values()} == {"timezone_lookup_failed"}
    assert env.db.rows == {}
    assert env.adds() == 0


@pytest.mark.asyncio
async def test_registry_read_failure_skips_the_cycle(env):
    env.db.fail = True

    result = await registry.ensure_scope_scheduled(ACCOUNT)

    assert {r["reason"] for r in result.values()} == {"registry_unavailable"}
    assert env.adds() == 0


@pytest.mark.asyncio
async def test_a_failed_registration_is_not_recorded_and_is_retried(env):
    scheduler = env.scheduler.add_scope_community_rebuild_schedule
    adder = scheduler.side_effect
    scheduler.side_effect = RuntimeError("scheduler down")

    first = await registry.ensure_scope_scheduled(ACCOUNT)

    assert first["community_rebuild"] == {
        "skipped": True,
        "reason": "registration_failed",
    }
    assert first["dream_nightly_batch"]["id"] == f"dream_nightly_batch_{USER}"
    assert env.db.rows[USER].community_job_id is None

    scheduler.side_effect = adder
    second = await registry.ensure_scope_scheduled(ACCOUNT)

    assert second["community_rebuild"]["id"] == f"community_rebuild_{USER}"
    assert second["dream_nightly_batch"] is None


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
async def test_registration_writes_the_redis_marker(env, fake_dream_redis):
    await registry.ensure_scope_scheduled(EXPERT)

    marker = f"dream_nightly_batch_registered:{EXPERT.scope_key}"
    assert fake_dream_redis.store[marker] == PARIS


@pytest.mark.asyncio
async def test_a_pause_that_lands_mid_registration_takes_the_new_jobs_down(env):
    add = env.scheduler.add_scope_nightly_batch_schedule
    adder = add.side_effect

    async def add_then_archive(*, scope: MemoryScope, user_timezone: str) -> dict:
        added = await adder(scope=scope, user_timezone=user_timezone)
        env.db.update(scope.scope_key, state=PAUSED)
        return added

    add.side_effect = add_then_archive
    await registry.ensure_scope_scheduled(EXPERT)

    env.scheduler.remove_scope_memory_jobs.assert_awaited_once_with(scope=EXPERT)
    assert env.scheduler.jobs == {}


@pytest.mark.asyncio
async def test_ensure_dream_system_scheduled_is_the_account_scope(env):
    await registry.ensure_dream_system_scheduled(USER)

    assert list(env.db.rows) == [USER]
    assert await registry.ensure_dream_system_scheduled("") == {}


# ---------------------------------------------------------------------------
# Lifecycle: pause (archive), resume (revive), wipe, timezone change
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_archive_pauses_and_revive_resumes_the_expert(env):
    await registry.sync_expert_scope(USER, "expert-1", active=True)  # hire
    assert set(env.scheduler.jobs) == _job_ids(EXPERT)

    await registry.sync_expert_scope(USER, "expert-1", active=False)  # archive
    row = env.db.rows[EXPERT.scope_key]
    assert (row.state, row.community_job_id, row.nightly_job_id) == (PAUSED, None, None)
    assert env.scheduler.jobs == {}

    await registry.sync_expert_scope(USER, "expert-1", active=True)  # re-hire
    row = env.db.rows[EXPERT.scope_key]
    assert row.state == ACTIVE
    assert {row.community_job_id, row.nightly_job_id} == _job_ids(EXPERT)
    assert set(env.scheduler.jobs) == _job_ids(EXPERT)
    assert list(env.db.rows) == [EXPERT.scope_key]


@pytest.mark.asyncio
async def test_the_lifecycle_hook_never_fails_its_caller(env, monkeypatch):
    """Hire, archive, pause and resume call in; nothing in memory
    scheduling may fail them — not an invalid id, not a bug."""
    monkeypatch.setattr(
        registry, "pause_scope", AsyncMock(side_effect=RuntimeError("bug"))
    )

    await registry.sync_expert_scope(USER, "expert-1", active=False)
    await registry.sync_expert_scope(USER, "", active=True)

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
    assert set(env.scheduler.jobs) == _job_ids(EXPERT)


@pytest.mark.asyncio
async def test_wipe_removes_the_jobs_until_resumed(env):
    await registry.ensure_scope_scheduled(ACCOUNT)

    assert await registry.mark_wiped(ACCOUNT)

    assert env.db.rows[USER].state == WIPED
    assert env.scheduler.jobs == {}
    skipped = await registry.ensure_scope_scheduled(ACCOUNT)
    assert {r["reason"] for r in skipped.values()} == {"scope_wiped"}

    await registry.resume_scope(ACCOUNT)
    assert set(env.scheduler.jobs) == _job_ids(ACCOUNT)


@pytest.mark.asyncio
async def test_wiping_an_account_without_a_row_removes_its_legacy_crons(env):
    env.scheduler.jobs = {job_id: PARIS for job_id in _job_ids(ACCOUNT)}

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
    assert set(env.scheduler.jobs) == _job_ids(ACCOUNT) | _job_ids(EXPERT)
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
    assert set(env.scheduler.jobs) == _job_ids(ACCOUNT)


# ---------------------------------------------------------------------------
# The cron-body side: the gate, run stamps, and in-band deletes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_gate_follows_the_row_state(env):
    await registry.ensure_scope_scheduled(EXPERT)
    assert await scope_jobs.scope_schedule_active(EXPERT)

    await registry.pause_scope(EXPERT)
    assert not await scope_jobs.scope_schedule_active(EXPERT)

    await registry.mark_wiped(EXPERT)
    assert not await scope_jobs.scope_schedule_active(EXPERT)


@pytest.mark.asyncio
async def test_a_legacy_account_without_row_still_fires(env):
    assert await scope_jobs.scope_schedule_active(ACCOUNT)
    env.scheduler.remove_scope_memory_jobs.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_expert_without_row_is_orphaned_and_its_crons_removed(env):
    assert not await scope_jobs.scope_schedule_active(EXPERT)
    env.scheduler.remove_scope_memory_jobs.assert_awaited_once_with(scope=EXPERT)


@pytest.mark.asyncio
async def test_a_failed_registry_read_blocks_the_fire(env):
    env.db.fail = True
    assert not await scope_jobs.scope_schedule_active(ACCOUNT)


@pytest.mark.asyncio
async def test_record_scope_run_stamps_the_row_and_fails_soft(env):
    await registry.ensure_scope_scheduled(ACCOUNT)

    await scope_jobs.record_scope_run(ACCOUNT, "nightly")

    assert env.db.rows[USER].last_nightly_run_at is not None
    assert env.db.rows[USER].last_community_run_at is None
    env.db.fail = True
    await scope_jobs.record_scope_run(ACCOUNT, "community")  # must not raise


@pytest.mark.asyncio
async def test_forgetting_a_deleted_cron_lets_ensure_register_it_again(
    env, fake_dream_redis
):
    await registry.ensure_scope_scheduled(ACCOUNT)
    nightly = dream_system_job("dream_nightly_batch")

    await scope_jobs.forget_registration(ACCOUNT, nightly)

    assert env.db.rows[USER].nightly_job_id is None
    assert env.db.rows[USER].community_job_id == f"community_rebuild_{USER}"
    assert f"dream_nightly_batch_registered:{USER}" not in fake_dream_redis.store
    again = await registry.ensure_scope_scheduled(ACCOUNT)
    assert again["community_rebuild"] is None
    assert again["dream_nightly_batch"]["id"] == f"dream_nightly_batch_{USER}"
