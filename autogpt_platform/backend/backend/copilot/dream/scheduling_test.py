"""Tests for the dream-system cron table, the owner-timezone lookup and the
Redis registration markers.

Contracts pinned here:

  1. Table shape — 2 rows (community + nightly batch), distinct markers,
     distinct flags, distinct registry columns.
  2. Job ids are keyed by scope, and the account's are the per-user ids
     the crons had before they were per scope.
  3. Each row registers through the scheduler's scope-keyed method.
  4. Timezone lookup failure is "unknown", not "UTC" — a transient DB blip
     must never re-register the owner's local-time crons onto UTC.
  5. Markers are per scope and cleared with a single-key DEL.

Registration itself (ensure, pause, resume, wipe) is pinned in
``registry_test.py``.
"""

from __future__ import annotations

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.model import USER_TIMEZONE_NOT_SET
from backend.util.feature_flag import Flag

from . import scheduling
from .scheduling import (
    DREAM_SYSTEM_JOBS,
    REGISTRATION_TTL_SECONDS,
    dream_system_job,
    resolve_user_timezone,
)

_PATH_USER_DB = "backend.copilot.dream.scheduling.user_db"
USER = "abc"
ACCOUNT = MemoryScope.for_user(USER)
EXPERT = MemoryScope.for_expert(USER, "expert-1")


# ---------------------------------------------------------------------------
# Table shape — guard against accidental edits
# ---------------------------------------------------------------------------


def test_table_contains_two_jobs_in_cron_frequency_order():
    """Weekly community first, daily nightly batch second — the order
    matches how schedules build up over time and reads naturally in
    the log narrative."""
    prefixes = [j.job_id_prefix for j in DREAM_SYSTEM_JOBS]
    assert prefixes == ["community_rebuild", "dream_nightly_batch"]


def test_each_job_has_distinct_registration_key_prefix():
    keys = [j.registration_key_prefix for j in DREAM_SYSTEM_JOBS]
    assert len(set(keys)) == len(keys)


def test_each_job_has_its_own_registry_column():
    fields = [j.row_field for j in DREAM_SYSTEM_JOBS]
    assert fields == ["community_job_id", "nightly_job_id"]


def test_community_and_nightly_batch_have_distinct_flags():
    """Community rebuild and nightly batch are independent features
    behind separate LD flags. A user can have one enabled without
    the other."""
    assert dream_system_job("community_rebuild").flag == (
        Flag.GRAPHITI_COMMUNITIES_ENABLED
    )
    assert dream_system_job("dream_nightly_batch").flag == Flag.DREAM_PASS_ENABLED


def test_unknown_job_prefix_raises():
    with pytest.raises(KeyError):
        dream_system_job("morning_briefing")


def test_exported_prefixes_match_table_rows():
    """The scheduler imports these constants for its job ids and the
    registry for its markers — drift would register or clear the wrong
    key."""
    by_prefix = {j.job_id_prefix: j.registration_key_prefix for j in DREAM_SYSTEM_JOBS}
    assert by_prefix == {
        scheduling.COMMUNITY_REBUILD_JOB_PREFIX: (
            scheduling.COMMUNITY_REBUILD_REGISTRATION_PREFIX
        ),
        scheduling.NIGHTLY_BATCH_JOB_PREFIX: (
            scheduling.NIGHTLY_BATCH_REGISTRATION_PREFIX
        ),
    }


# ---------------------------------------------------------------------------
# Job ids and registration calls
# ---------------------------------------------------------------------------


def test_account_job_ids_are_the_legacy_per_user_ids():
    """Existing crons were registered as ``{prefix}_{user_id}``; the account
    scope must land on exactly those ids so nothing is duplicated."""
    assert [j.job_id(ACCOUNT) for j in DREAM_SYSTEM_JOBS] == [
        "community_rebuild_abc",
        "dream_nightly_batch_abc",
    ]


def test_expert_job_ids_are_keyed_by_the_expert_scope():
    assert [j.job_id(EXPERT) for j in DREAM_SYSTEM_JOBS] == [
        f"community_rebuild_{EXPERT.scope_key}",
        f"dream_nightly_batch_{EXPERT.scope_key}",
    ]


@pytest.mark.asyncio
async def test_each_row_registers_through_the_scope_keyed_method():
    client = MagicMock()
    client.add_scope_community_rebuild_schedule = AsyncMock(return_value={"id": "c"})
    client.add_scope_nightly_batch_schedule = AsyncMock(return_value={"id": "n"})

    for job in DREAM_SYSTEM_JOBS:
        await job.register(client, EXPERT, "Europe/Paris")

    client.add_scope_community_rebuild_schedule.assert_awaited_once_with(
        scope=EXPERT, user_timezone="Europe/Paris"
    )
    client.add_scope_nightly_batch_schedule.assert_awaited_once_with(
        scope=EXPERT, user_timezone="Europe/Paris"
    )


# ---------------------------------------------------------------------------
# Owner timezone
# ---------------------------------------------------------------------------


def _user_db(**get_user_by_id) -> MagicMock:
    accessor = MagicMock()
    accessor.get_user_by_id = AsyncMock(**get_user_by_id)
    return accessor


@pytest.mark.asyncio
async def test_resolve_user_timezone_works_without_local_prisma_connection():
    """Dev outage AUTOGPT: the resolver ran in the copilot-executor
    process, which never connects a local Prisma client — a direct
    ``User.prisma()`` call raised ``ClientNotConnectedError`` on every
    invocation, so dream crons were never registered for anyone. The
    lookup must route through the ``user_db()`` accessor, which falls
    back to the DatabaseManager RPC in Prisma-less processes."""
    accessor = _user_db(return_value=MagicMock(timezone="Europe/Madrid"))
    with patch(_PATH_USER_DB, return_value=accessor):
        assert await resolve_user_timezone(USER) == "Europe/Madrid"
    accessor.get_user_by_id.assert_awaited_once_with(USER)


@pytest.mark.asyncio
async def test_resolve_user_timezone_missing_user_is_authoritative_utc():
    """get_user_by_id raises ValueError for a missing row — that's an
    authoritative answer (UTC), not a lookup failure (None)."""
    accessor = _user_db(side_effect=ValueError("User not found"))
    with patch(_PATH_USER_DB, return_value=accessor):
        assert await resolve_user_timezone(USER) == "UTC"


@pytest.mark.asyncio
async def test_resolve_user_timezone_returns_none_when_db_lookup_fails():
    """The resolver distinguishes 'lookup failed' (None) from
    'genuinely unset' (UTC)."""
    accessor = _user_db(side_effect=ConnectionError("db down"))
    with patch(_PATH_USER_DB, return_value=accessor):
        assert await resolve_user_timezone(USER) is None


@pytest.mark.asyncio
async def test_resolve_user_timezone_unset_value_falls_back_to_utc():
    accessor = _user_db(return_value=MagicMock(timezone=USER_TIMEZONE_NOT_SET))
    with patch(_PATH_USER_DB, return_value=accessor):
        assert await resolve_user_timezone(USER) == "UTC"


# ---------------------------------------------------------------------------
# Redis markers — a cache of the registry, one per scope and cron
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_registration_marker_stores_the_timezone_per_scope(
    fake_dream_redis,
):
    await scheduling.write_registration_marker(
        EXPERT, "dream_nightly_batch_registered", "Asia/Tokyo"
    )

    key = f"dream_nightly_batch_registered:{EXPERT.scope_key}"
    assert fake_dream_redis.store[key] == "Asia/Tokyo"
    assert fake_dream_redis.ttls[key] == REGISTRATION_TTL_SECONDS


@pytest.mark.asyncio
async def test_a_hung_redis_never_holds_up_a_marker_write(monkeypatch):
    """The marker is a cache written inline by hires and resumes; the Redis
    client's connect retries must not hold those up."""

    async def never_connects():
        await asyncio.sleep(3600)

    monkeypatch.setattr(scheduling, "MARKER_TIMEOUT_SECONDS", 0.05)
    with patch("backend.data.redis_client.get_redis_async", new=never_connects):
        await asyncio.wait_for(
            scheduling.write_registration_marker(EXPERT, "x_registered", "UTC"),
            timeout=2,
        )


@pytest.mark.asyncio
async def test_clear_registration_marker_deletes_the_single_per_cron_key():
    """Single-key DEL (Redis-cluster-safe) on the per-cron marker."""
    redis = AsyncMock()
    with patch(
        "backend.data.redis_client.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        await scheduling.clear_registration_marker(
            ACCOUNT, "dream_nightly_batch_registered"
        )
    redis.delete.assert_awaited_once_with("dream_nightly_batch_registered:abc")


@pytest.mark.asyncio
async def test_clear_registration_marker_swallows_redis_failure(caplog):
    """Best-effort: a Redis outage during clear must never break the
    delete RPC — the marker self-heals via its 7-day TTL. The except block
    must still emit the WARNING (with exc_info) that documents the fallback,
    so a silent-swallow regression is caught."""
    with patch(
        "backend.data.redis_client.get_redis_async",
        new=AsyncMock(side_effect=ConnectionError("redis down")),
    ), caplog.at_level(logging.WARNING, logger=scheduling.logger.name):
        await scheduling.clear_registration_marker(ACCOUNT, "x_registered")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("marker will expire via TTL" in r.getMessage() for r in warnings)
    assert any(r.exc_info is not None for r in warnings)


def test_registration_ttl_matches_longest_cron_cadence():
    """TTL must be at least as long as the weekly community rebuild
    cadence."""
    one_week_seconds = 7 * 24 * 3600
    assert REGISTRATION_TTL_SECONDS >= one_week_seconds
