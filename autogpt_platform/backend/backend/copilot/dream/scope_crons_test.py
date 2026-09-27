"""Tests for what the dream-system cron bodies ask of the registry
(``scope_crons.py``): the gate, the run stamp, in-band-delete bookkeeping,
reading a pass's outcome, and the jobs' args and results."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from backend.data.db_manager import DatabaseManagerAsyncClient

from . import registry, scope_crons
from .registry_fakes_test import (
    ACCOUNT,
    EXPERT,
    USER,
    Hang,
    HangingTransport,
    RegistryEnv,
    install_registry_env,
    use_short_deadline,
)
from .scheduling import dream_system_job

PROMPT = 1.0


@pytest.fixture
def env(monkeypatch: pytest.MonkeyPatch) -> RegistryEnv:
    return install_registry_env(monkeypatch)


def _status(value: str) -> AsyncMock:
    return AsyncMock(return_value=value)


@pytest.mark.asyncio
async def test_the_gate_follows_the_row_state(env):
    await registry.ensure_scope_scheduled(EXPERT)
    active = _status("active")
    assert await scope_crons.memory_scope_may_fire(EXPERT, active)

    await registry.pause_scope(EXPERT)
    assert not await scope_crons.memory_scope_may_fire(EXPERT, active)

    await registry.mark_wiped(EXPERT)
    assert not await scope_crons.memory_scope_may_fire(EXPERT, active)


@pytest.mark.asyncio
async def test_a_legacy_account_without_row_still_fires(env):
    status = _status("active")

    assert await scope_crons.memory_scope_may_fire(ACCOUNT, status)

    status.assert_not_awaited()  # the account has no expert to check
    env.scheduler.remove_scope_memory_jobs.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_expert_without_row_is_orphaned_and_its_crons_removed(env):
    assert not await scope_crons.memory_scope_may_fire(EXPERT, _status("active"))

    env.scheduler.remove_scope_memory_jobs.assert_awaited_once_with(scope=EXPERT)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, fires",
    [
        ("active", True),
        ("paused", False),
        ("archived", False),
        ("unavailable", False),
        ("missing", False),
    ],
)
async def test_an_expert_scope_fires_only_while_the_expert_is_active(
    env, status: str, fires: bool
):
    await registry.ensure_scope_scheduled(EXPERT)
    lookup = _status(status)

    assert await scope_crons.memory_scope_may_fire(EXPERT, lookup) is fires

    lookup.assert_awaited_once_with(USER, "expert-1")


@pytest.mark.asyncio
async def test_a_failed_registry_read_blocks_the_fire(env):
    env.db.fail = True
    assert not await scope_crons.memory_scope_may_fire(ACCOUNT, _status("active"))


@pytest.mark.asyncio
async def test_the_gate_skips_promptly_when_the_registry_hangs(env, monkeypatch):
    use_short_deadline(monkeypatch)
    transport = HangingTransport()
    rpc = transport.client(DatabaseManagerAsyncClient, request_retry=True)
    monkeypatch.setattr(scope_crons, "memory_schedule_db", lambda: rpc)
    started = time.monotonic()
    try:
        fires = await scope_crons.memory_scope_may_fire(ACCOUNT, _status("active"))
    finally:
        await rpc.aclose()

    assert not fires
    assert time.monotonic() - started < PROMPT
    assert transport.cancelled == transport.paths == ["/get_scope_schedule"]


@pytest.mark.asyncio
async def test_the_gate_skips_promptly_when_the_expert_lookup_hangs(env, monkeypatch):
    await registry.ensure_scope_scheduled(EXPERT)
    use_short_deadline(monkeypatch)
    hang = Hang()
    started = time.monotonic()

    fires = await scope_crons.memory_scope_may_fire(EXPERT, hang.forever)

    assert not fires
    assert time.monotonic() - started < PROMPT
    assert hang.started == hang.cancelled == 1


@pytest.mark.asyncio
async def test_record_scope_run_stamps_the_row_and_fails_soft(env):
    await registry.ensure_scope_scheduled(ACCOUNT)

    assert await scope_crons.record_scope_run(ACCOUNT, "nightly")

    assert env.db.rows[USER].last_nightly_run_at is not None
    assert env.db.rows[USER].last_community_run_at is None
    env.db.fail = True
    assert not await scope_crons.record_scope_run(ACCOUNT, "community")


@pytest.mark.asyncio
async def test_record_scope_run_gives_up_at_the_deadline(env, monkeypatch):
    await registry.ensure_scope_scheduled(ACCOUNT)
    use_short_deadline(monkeypatch)
    hang = Hang()
    monkeypatch.setattr(env.db, "record_scope_run", hang.forever)
    started = time.monotonic()

    assert not await scope_crons.record_scope_run(ACCOUNT, "nightly")

    assert time.monotonic() - started < PROMPT
    assert hang.started == hang.cancelled == 1


@pytest.mark.asyncio
async def test_forgetting_a_deleted_cron_lets_ensure_register_it_again(
    env, fake_dream_redis
):
    await registry.ensure_scope_scheduled(ACCOUNT)

    await scope_crons.forget_registration(
        ACCOUNT, dream_system_job("dream_nightly_batch")
    )

    assert env.db.rows[USER].nightly_job_id is None
    assert env.db.rows[USER].community_job_id == f"community_rebuild_{USER}"
    assert f"dream_nightly_batch_registered:{USER}" not in fake_dream_redis.store
    again = await registry.ensure_scope_scheduled(ACCOUNT)
    assert again["community_rebuild"] is None
    assert again["dream_nightly_batch"]["id"] == f"dream_nightly_batch_{USER}"


def test_job_kwargs_keep_the_accounts_legacy_shape():
    assert scope_crons.memory_job_kwargs(ACCOUNT) == {"user_id": USER}
    assert scope_crons.memory_job_kwargs(EXPERT) == {
        "user_id": USER,
        "expert_id": "expert-1",
    }


def test_job_results_carry_the_scope_key():
    when = datetime(2026, 9, 28, 4, tzinfo=timezone.utc)

    added = scope_crons.memory_job_result(EXPERT, "job-1", when, "Asia/Tokyo")
    skipped = scope_crons.memory_job_skipped(ACCOUNT, "UTC", "dream_pass_disabled")

    assert added == {
        "id": "job-1",
        "user_id": USER,
        "scope_key": EXPERT.scope_key,
        "user_timezone": "Asia/Tokyo",
        "next_run_time": when.isoformat(),
    }
    assert skipped["skipped"] is True and skipped["scope_key"] == USER
