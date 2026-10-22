"""Tests for the dream-system cron table and the owner-timezone lookup.

Contracts pinned here:

  1. Table shape — 3 rows (community, nightly batch, skill learning),
     distinct registry columns; community maintenance has no flag and skill learning is the
     account's alone.
  2. Job ids are keyed by scope, and the account's are the per-user ids
     the crons had before they were per scope.
  3. Each row registers through the scheduler's scope-keyed method.
  4. Timezone lookup failure is "unknown", not "UTC" — a transient DB blip
     must never re-register the owner's local-time crons onto UTC.

Registration itself (ensure, pause, resume, wipe) is pinned in
``registry_test.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.graphiti.scope import MemoryScope
from backend.data.model import USER_TIMEZONE_NOT_SET
from backend.util.feature_flag import Flag

from . import deadline, scheduling
from .registry_fakes_test import Hang
from .scheduling import (
    DREAM_SYSTEM_JOBS,
    dream_system_job,
    jobs_for,
    resolve_user_timezone,
)

_PATH_USER_DB = "backend.copilot.dream.scheduling.user_db"
USER = "abc"
ACCOUNT = MemoryScope.for_user(USER)
EXPERT = MemoryScope.for_expert(USER, "expert-1")


# ---------------------------------------------------------------------------
# Table shape — guard against accidental edits
# ---------------------------------------------------------------------------


def test_table_contains_three_jobs_in_cron_frequency_order():
    """Weekly community first, then the daily crons — the order matches
    how schedules build up over time and reads naturally in the log
    narrative."""
    prefixes = [j.job_id_prefix for j in DREAM_SYSTEM_JOBS]
    assert prefixes == [
        "community_rebuild",
        "dream_nightly_batch",
        "skill_learning_nightly",
    ]


def test_each_job_has_its_own_registry_column():
    fields = [j.row_field for j in DREAM_SYSTEM_JOBS]
    assert fields == ["community_job_id", "nightly_job_id", "learning_job_id"]


def test_community_is_ungated_and_nightly_batch_keeps_its_flag():
    assert dream_system_job("community_rebuild").flag is None
    assert dream_system_job("dream_nightly_batch").flag == Flag.DREAM_PASS_ENABLED


def test_unknown_job_prefix_raises():
    with pytest.raises(KeyError):
        dream_system_job("morning_briefing")


def test_exported_prefixes_match_table_rows():
    """The scheduler imports these constants for its job ids: drift would
    register or remove the wrong job."""
    assert [j.job_id_prefix for j in DREAM_SYSTEM_JOBS] == [
        scheduling.COMMUNITY_REBUILD_JOB_PREFIX,
        scheduling.NIGHTLY_BATCH_JOB_PREFIX,
        scheduling.SKILL_LEARNING_JOB_PREFIX,
    ]


def test_skill_learning_is_gated_independently_of_the_dream_pass():
    """Enabling skill learning must not imply the dream pass (or vice
    versa): its own flag, its own cron id, its own registry column."""
    learning = dream_system_job("skill_learning_nightly")
    assert learning.flag == Flag.DREAM_SKILL_LEARNING_ENABLED
    assert learning.skip_reason == "skill_learning_disabled"
    assert learning.row_field == "learning_job_id"


def test_skill_learning_is_the_accounts_alone():
    """One pass reviews the account's skills and its experts': an expert's
    scope has no skill-learning cron, and the account has all three."""
    assert [j.job_id_prefix for j in jobs_for(ACCOUNT)] == [
        "community_rebuild",
        "dream_nightly_batch",
        "skill_learning_nightly",
    ]
    assert [j.job_id_prefix for j in jobs_for(EXPERT)] == [
        "community_rebuild",
        "dream_nightly_batch",
    ]
    assert [j.job_id_prefix for j in DREAM_SYSTEM_JOBS if j.account_only] == [
        "skill_learning_nightly"
    ]


# ---------------------------------------------------------------------------
# Job ids and registration calls
# ---------------------------------------------------------------------------


def test_account_job_ids_are_the_legacy_per_user_ids():
    """Existing crons were registered as ``{prefix}_{user_id}``; the account
    scope must land on exactly those ids so nothing is duplicated."""
    assert [j.job_id(ACCOUNT) for j in DREAM_SYSTEM_JOBS] == [
        "community_rebuild_abc",
        "dream_nightly_batch_abc",
        "skill_learning_nightly_abc",
    ]


def test_expert_job_ids_are_keyed_by_the_expert_scope():
    assert [j.job_id(EXPERT) for j in jobs_for(EXPERT)] == [
        f"community_rebuild_{EXPERT.scope_key}",
        f"dream_nightly_batch_{EXPERT.scope_key}",
    ]


@pytest.mark.asyncio
async def test_each_row_registers_through_the_scope_keyed_method():
    client = MagicMock()
    client.add_scope_community_rebuild_schedule = AsyncMock(return_value={"id": "c"})
    client.add_scope_nightly_batch_schedule = AsyncMock(return_value={"id": "n"})
    client.add_scope_skill_learning_schedule = AsyncMock(return_value={"id": "l"})

    for job in DREAM_SYSTEM_JOBS:
        await job.register(client, ACCOUNT, "Europe/Paris")

    for method in (
        client.add_scope_community_rebuild_schedule,
        client.add_scope_nightly_batch_schedule,
        client.add_scope_skill_learning_schedule,
    ):
        method.assert_awaited_once_with(scope=ACCOUNT, user_timezone="Europe/Paris")


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
async def test_resolve_user_timezone_gives_up_at_the_deadline(monkeypatch):
    """A hanging lookup is a failed lookup ("unknown", not "UTC"), and the
    hanging call is cancelled."""
    monkeypatch.setattr(deadline, "REGISTRY_CALL_TIMEOUT_SECONDS", 0.05)
    hang = Hang()
    accessor = MagicMock()
    accessor.get_user_by_id = hang.forever
    with patch(_PATH_USER_DB, return_value=accessor):
        assert await resolve_user_timezone(USER) is None
    assert hang.started == hang.cancelled == 1


@pytest.mark.asyncio
async def test_resolve_user_timezone_unset_value_falls_back_to_utc():
    accessor = _user_db(return_value=MagicMock(timezone=USER_TIMEZONE_NOT_SET))
    with patch(_PATH_USER_DB, return_value=accessor):
        assert await resolve_user_timezone(USER) == "UTC"
