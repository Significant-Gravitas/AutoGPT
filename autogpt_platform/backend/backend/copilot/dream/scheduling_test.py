"""Registry-driven dream-system auto-registration tests.

Contracts pinned here:

  1. Registry shape — 2 rows (community + nightly batch), distinct
     dedup keys, distinct flags.
  2. Per-job flag gating — flag-off → recorded as skipped with the
     right reason; scheduler RPC never called.
  3. Per-job idempotency — each cron has its own Redis dedup key so
     flipping one flag mid-life only re-enters that cron.
  4. Per-job failure isolation — one cron's scheduler RPC failure
     doesn't block the other cron.
  5. Timezone drift detection — Redis dedup key stores the
     registered timezone; on mismatch with current timezone, force
     re-register. ``force_refresh=True`` skips the drift check and
     always re-registers.
  6. Shared per-call caching — timezone resolution happens at most
     once per invocation.
  7. Timezone lookup failure is "unknown", not "UTC" — a transient
     DB blip must never re-register the user's local-time crons onto
     UTC; the cycle is skipped and the existing cron left untouched.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.util.feature_flag import Flag

from . import scheduling
from .scheduling import (
    DREAM_SYSTEM_JOBS,
    REGISTRATION_TTL_SECONDS,
    ensure_dream_system_scheduled,
)

# Paths used by the helper — patched at the module level since the
# helper does lazy imports of the scheduler client + Redis client to
# avoid bootstrap circular imports.
_PATH_FLAG = "backend.copilot.dream.scheduling.is_feature_enabled"
_PATH_TZ = "backend.copilot.dream.scheduling._resolve_user_timezone"
_PATH_READ_TZ = "backend.copilot.dream.scheduling._read_registration_tz"
_PATH_WRITE_TZ = "backend.copilot.dream.scheduling._write_registration_tz"
_PATH_CLIENT = "backend.util.clients.get_scheduler_client"


def _mock_scheduler_client(*, fail_jobs: tuple[str, ...] = ()) -> MagicMock:
    """SchedulerClient stub returning a documented dict per call.

    ``fail_jobs`` = tuple of job_id_prefix strings that should raise
    instead of returning success — for per-job failure isolation tests.
    """
    client = MagicMock()

    def _maker(prefix: str):
        if prefix in fail_jobs:
            return AsyncMock(side_effect=RuntimeError(f"{prefix} scheduler down"))
        return AsyncMock(
            return_value={
                "id": f"{prefix}_abc",
                "user_id": "abc",
                "next_run_time": None,
            }
        )

    client.add_community_rebuild_schedule = _maker("community_rebuild")
    client.add_nightly_batch_schedule = _maker("dream_nightly_batch")
    return client


def _flag_mock(flag_map: dict[Flag, bool]) -> AsyncMock:
    async def fake(flag: Flag, user_id: str) -> bool:
        return flag_map.get(flag, False)

    return AsyncMock(side_effect=fake)


def _all_flags_on() -> dict[Flag, bool]:
    return {
        Flag.GRAPHITI_COMMUNITIES_ENABLED: True,
        Flag.DREAM_PASS_ENABLED: True,
    }


# ---------------------------------------------------------------------------
# Registry shape — guard against accidental edits
# ---------------------------------------------------------------------------


def test_registry_contains_two_jobs_in_cron_frequency_order():
    """Weekly community first, daily nightly batch second — the order
    matches how schedules build up over time and reads naturally in
    the log narrative."""
    prefixes = [j.job_id_prefix for j in DREAM_SYSTEM_JOBS]
    assert prefixes == ["community_rebuild", "dream_nightly_batch"]


def test_each_job_has_distinct_registration_key_prefix():
    """Per-job dedup keys MUST be distinct — a shared key would block
    recovery when a flag flips off→on for one cron but not the other."""
    keys = [j.registration_key_prefix for j in DREAM_SYSTEM_JOBS]
    assert len(set(keys)) == len(keys)


def test_community_and_nightly_batch_have_distinct_flags():
    """Community rebuild and nightly batch are independent features
    behind separate LD flags. A user can have one enabled without
    the other."""
    community = next(
        j for j in DREAM_SYSTEM_JOBS if j.job_id_prefix == "community_rebuild"
    )
    nightly = next(
        j for j in DREAM_SYSTEM_JOBS if j.job_id_prefix == "dream_nightly_batch"
    )
    assert community.flag == Flag.GRAPHITI_COMMUNITIES_ENABLED
    assert nightly.flag == Flag.DREAM_PASS_ENABLED


# ---------------------------------------------------------------------------
# Empty / null user_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_user_id_returns_empty_dict_without_running():
    result = await ensure_dream_system_scheduled("")
    assert result == {}


# ---------------------------------------------------------------------------
# Happy path — both flags on, no existing registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_both_crons_register_when_their_flags_are_on():
    client = _mock_scheduler_client()
    with patch(_PATH_FLAG, new=_flag_mock(_all_flags_on())), patch(
        _PATH_TZ, new=AsyncMock(return_value="America/New_York")
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value=None)), patch(
        _PATH_WRITE_TZ, new=AsyncMock()
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    assert result["community_rebuild"]["id"] == "community_rebuild_abc"
    assert result["dream_nightly_batch"]["id"] == "dream_nightly_batch_abc"

    # Both scheduler calls threaded the same resolved timezone.
    client.add_community_rebuild_schedule.assert_awaited_once_with(
        user_id="abc", user_timezone="America/New_York"
    )
    client.add_nightly_batch_schedule.assert_awaited_once_with(
        user_id="abc", user_timezone="America/New_York"
    )


# ---------------------------------------------------------------------------
# Layer 1 — flag gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_community_flag_off_skips_only_community():
    """One flag off → only that cron is skipped; the other registers."""
    client = _mock_scheduler_client()
    flag_map = _all_flags_on()
    flag_map[Flag.GRAPHITI_COMMUNITIES_ENABLED] = False
    with patch(_PATH_FLAG, new=_flag_mock(flag_map)), patch(
        _PATH_TZ, new=AsyncMock(return_value="UTC")
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value=None)), patch(
        _PATH_WRITE_TZ, new=AsyncMock()
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    assert result["community_rebuild"] == {
        "skipped": True,
        "reason": "graphiti_communities_disabled",
    }
    assert result["dream_nightly_batch"]["id"] == "dream_nightly_batch_abc"
    client.add_community_rebuild_schedule.assert_not_called()


@pytest.mark.asyncio
async def test_dream_flag_off_skips_only_nightly_batch():
    client = _mock_scheduler_client()
    flag_map = _all_flags_on()
    flag_map[Flag.DREAM_PASS_ENABLED] = False
    with patch(_PATH_FLAG, new=_flag_mock(flag_map)), patch(
        _PATH_TZ, new=AsyncMock(return_value="UTC")
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value=None)), patch(
        _PATH_WRITE_TZ, new=AsyncMock()
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    assert result["community_rebuild"]["id"] == "community_rebuild_abc"
    assert result["dream_nightly_batch"] == {
        "skipped": True,
        "reason": "dream_pass_disabled",
    }
    client.add_nightly_batch_schedule.assert_not_called()


@pytest.mark.asyncio
async def test_all_flags_off_returns_all_skipped_no_rpc_no_tz_lookup():
    """When nothing is enabled, don't waste a DB lookup for tz or an
    RPC to the scheduler service."""
    client = _mock_scheduler_client()
    tz_mock = AsyncMock(return_value="UTC")
    with patch(_PATH_FLAG, new=_flag_mock({})), patch(_PATH_TZ, new=tz_mock), patch(
        _PATH_READ_TZ, new=AsyncMock(return_value=None)
    ), patch(_PATH_WRITE_TZ, new=AsyncMock()), patch(_PATH_CLIENT, return_value=client):
        result = await ensure_dream_system_scheduled("abc")

    assert all(r.get("skipped") is True for r in result.values())  # type: ignore[union-attr]
    tz_mock.assert_not_called()
    client.add_community_rebuild_schedule.assert_not_called()
    client.add_nightly_batch_schedule.assert_not_called()


# ---------------------------------------------------------------------------
# Layer 2 — timezone drift detection (replaces old SETNX-only semantics)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_drift_returns_none_without_rpc():
    """Cron already registered with the user's current timezone → no
    work, no RPC."""
    client = _mock_scheduler_client()
    with patch(_PATH_FLAG, new=_flag_mock(_all_flags_on())), patch(
        _PATH_TZ, new=AsyncMock(return_value="America/New_York")
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value="America/New_York")), patch(
        _PATH_WRITE_TZ, new=AsyncMock()
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    assert result["community_rebuild"] is None
    assert result["dream_nightly_batch"] is None
    client.add_community_rebuild_schedule.assert_not_called()
    client.add_nightly_batch_schedule.assert_not_called()


@pytest.mark.asyncio
async def test_timezone_drift_re_registers_via_replace_existing():
    """User moved NY → Paris; existing cron is still bound to NY tz.
    Helper detects the drift via the stored value and re-registers."""
    client = _mock_scheduler_client()
    write_spy = AsyncMock()
    with patch(_PATH_FLAG, new=_flag_mock(_all_flags_on())), patch(
        _PATH_TZ, new=AsyncMock(return_value="Europe/Paris")
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value="America/New_York")), patch(
        _PATH_WRITE_TZ, new=write_spy
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    # Both crons re-registered with the new tz.
    client.add_community_rebuild_schedule.assert_awaited_once_with(
        user_id="abc", user_timezone="Europe/Paris"
    )
    client.add_nightly_batch_schedule.assert_awaited_once_with(
        user_id="abc", user_timezone="Europe/Paris"
    )
    # Stored tz updated for both.
    written_tzs = [call.args[2] for call in write_spy.await_args_list]
    assert written_tzs == ["Europe/Paris", "Europe/Paris"]
    assert result["community_rebuild"]["id"] == "community_rebuild_abc"
    assert result["dream_nightly_batch"]["id"] == "dream_nightly_batch_abc"


@pytest.mark.asyncio
async def test_force_refresh_skips_drift_check_and_always_re_registers():
    """``force_refresh=True`` (from the User.timezone update hook)
    bypasses drift detection and always re-registers."""
    client = _mock_scheduler_client()
    read_spy = AsyncMock(return_value="America/New_York")
    with patch(_PATH_FLAG, new=_flag_mock(_all_flags_on())), patch(
        _PATH_TZ, new=AsyncMock(return_value="America/New_York")
    ), patch(_PATH_READ_TZ, new=read_spy), patch(
        _PATH_WRITE_TZ, new=AsyncMock()
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc", force_refresh=True)

    # Read NEVER consulted when force_refresh=True — saves a Redis
    # round-trip when the eager hook KNOWS something changed.
    read_spy.assert_not_called()
    client.add_community_rebuild_schedule.assert_awaited_once()
    client.add_nightly_batch_schedule.assert_awaited_once()
    assert result["community_rebuild"]["id"] == "community_rebuild_abc"


@pytest.mark.asyncio
async def test_redis_read_failure_treats_as_first_registration():
    """Redis unavailable → ``_read_registration_tz`` returns None →
    helper falls through to scheduler RPC. APScheduler's
    ``replace_existing=True`` is the durable backstop."""
    client = _mock_scheduler_client()
    with patch(_PATH_FLAG, new=_flag_mock(_all_flags_on())), patch(
        _PATH_TZ, new=AsyncMock(return_value="UTC")
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value=None)), patch(
        _PATH_WRITE_TZ, new=AsyncMock()
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    assert result["community_rebuild"]["id"] == "community_rebuild_abc"
    assert result["dream_nightly_batch"]["id"] == "dream_nightly_batch_abc"


# ---------------------------------------------------------------------------
# Timezone lookup failure — "unknown" must not become "UTC"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timezone_lookup_failure_leaves_existing_crons_untouched():
    """A transient DB failure during tz resolution must NOT be read as
    'user is in UTC' — that would re-register the 03:00-local crons
    onto 03:00 UTC and overwrite the stored tz. Skip the cycle: no
    scheduler RPC, no Redis write."""
    client = _mock_scheduler_client()
    write_spy = AsyncMock()
    with patch(_PATH_FLAG, new=_flag_mock(_all_flags_on())), patch(
        _PATH_TZ, new=AsyncMock(return_value=None)
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value="America/Chicago")), patch(
        _PATH_WRITE_TZ, new=write_spy
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    assert result["community_rebuild"] == {
        "skipped": True,
        "reason": "timezone_lookup_failed",
    }
    assert result["dream_nightly_batch"] == {
        "skipped": True,
        "reason": "timezone_lookup_failed",
    }
    client.add_community_rebuild_schedule.assert_not_called()
    client.add_nightly_batch_schedule.assert_not_called()
    write_spy.assert_not_called()


@pytest.mark.asyncio
async def test_timezone_lookup_failure_attempted_once_not_per_job():
    """A failed lookup is cached for the invocation like a successful
    one — two enabled crons must not trigger a second DB round-trip."""
    tz_mock = AsyncMock(return_value=None)
    client = _mock_scheduler_client()
    with patch(_PATH_FLAG, new=_flag_mock(_all_flags_on())), patch(
        _PATH_TZ, new=tz_mock
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value=None)), patch(
        _PATH_WRITE_TZ, new=AsyncMock()
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        await ensure_dream_system_scheduled("abc")

    tz_mock.assert_awaited_once_with("abc")


@pytest.mark.asyncio
async def test_force_refresh_with_failed_tz_lookup_still_skips_re_registration():
    """Even the eager path (timezone-update endpoint) must not
    re-register onto UTC when it can't read the new timezone back —
    the lazy drift-detection path recovers once the DB is healthy."""
    client = _mock_scheduler_client()
    with patch(_PATH_FLAG, new=_flag_mock(_all_flags_on())), patch(
        _PATH_TZ, new=AsyncMock(return_value=None)
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value=None)), patch(
        _PATH_WRITE_TZ, new=AsyncMock()
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc", force_refresh=True)

    assert result["community_rebuild"] == {
        "skipped": True,
        "reason": "timezone_lookup_failed",
    }
    client.add_community_rebuild_schedule.assert_not_called()
    client.add_nightly_batch_schedule.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_user_timezone_works_without_local_prisma_connection():
    """Dev outage AUTOGPT: the resolver ran in the copilot-executor
    process, which never connects a local Prisma client — a direct
    ``User.prisma()`` call raised ``ClientNotConnectedError`` on every
    invocation, so dream crons were never registered for anyone. The
    lookup must route through the ``user_db()`` accessor, which falls
    back to the DatabaseManager RPC in Prisma-less processes."""
    accessor = MagicMock()
    accessor.get_user_by_id = AsyncMock(
        return_value=MagicMock(timezone="Europe/Madrid")
    )
    with patch("backend.data.db_accessors.user_db", return_value=accessor):
        assert await scheduling._resolve_user_timezone("abc") == "Europe/Madrid"
    accessor.get_user_by_id.assert_awaited_once_with("abc")


@pytest.mark.asyncio
async def test_resolve_user_timezone_missing_user_is_authoritative_utc():
    """get_user_by_id raises ValueError for a missing row — that's an
    authoritative answer (UTC), not a lookup failure (None)."""
    accessor = MagicMock()
    accessor.get_user_by_id = AsyncMock(side_effect=ValueError("User not found"))
    with patch("backend.data.db_accessors.user_db", return_value=accessor):
        assert await scheduling._resolve_user_timezone("abc") == "UTC"


@pytest.mark.asyncio
async def test_resolve_user_timezone_returns_none_when_db_lookup_fails():
    """The resolver distinguishes 'lookup failed' (None) from
    'genuinely unset' (UTC)."""
    accessor = MagicMock()
    accessor.get_user_by_id = AsyncMock(side_effect=ConnectionError("db down"))
    with patch("backend.data.db_accessors.user_db", return_value=accessor):
        assert await scheduling._resolve_user_timezone("abc") is None


@pytest.mark.asyncio
async def test_resolve_user_timezone_unset_value_falls_back_to_utc():
    """A user whose timezone was never set legitimately defaults to
    UTC — only lookup FAILURES return None."""
    from backend.data.model import USER_TIMEZONE_NOT_SET

    accessor = MagicMock()
    accessor.get_user_by_id = AsyncMock(
        return_value=MagicMock(timezone=USER_TIMEZONE_NOT_SET)
    )
    with patch("backend.data.db_accessors.user_db", return_value=accessor):
        assert await scheduling._resolve_user_timezone("abc") == "UTC"


# ---------------------------------------------------------------------------
# Layer 3 — per-job RPC failure isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_cron_rpc_failure_does_not_block_the_other():
    """Community rebuild scheduler call raises; nightly batch still
    registers cleanly. Failed cron surfaces as skipped dict, not a
    propagated exception."""
    client = _mock_scheduler_client(fail_jobs=("community_rebuild",))
    with patch(_PATH_FLAG, new=_flag_mock(_all_flags_on())), patch(
        _PATH_TZ, new=AsyncMock(return_value="UTC")
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value=None)), patch(
        _PATH_WRITE_TZ, new=AsyncMock()
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    assert result["community_rebuild"] == {
        "skipped": True,
        "reason": "registration_failed",
    }
    assert result["dream_nightly_batch"]["id"] == "dream_nightly_batch_abc"


# ---------------------------------------------------------------------------
# Shared per-call caching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timezone_lookup_runs_at_most_once_per_invocation():
    """Two crons registering must share one Prisma round-trip for the
    user's timezone."""
    client = _mock_scheduler_client()
    tz_mock = AsyncMock(return_value="UTC")
    with patch(_PATH_FLAG, new=_flag_mock(_all_flags_on())), patch(
        _PATH_TZ, new=tz_mock
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value=None)), patch(
        _PATH_WRITE_TZ, new=AsyncMock()
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        await ensure_dream_system_scheduled("abc")

    tz_mock.assert_awaited_once_with("abc")


# ---------------------------------------------------------------------------
# Module constants — defended against TTL drift
# ---------------------------------------------------------------------------


def test_registration_ttl_matches_longest_cron_cadence():
    """TTL must be at least as long as the weekly community rebuild
    cadence so the lazy drift-detection path re-checks at least once
    per cron-tick."""
    one_week_seconds = 7 * 24 * 3600
    assert REGISTRATION_TTL_SECONDS >= one_week_seconds


# Suppress "imported but unused" — ``scheduling`` is the module under test;
# some linters miss the patch() string-target usage.
_ = scheduling
