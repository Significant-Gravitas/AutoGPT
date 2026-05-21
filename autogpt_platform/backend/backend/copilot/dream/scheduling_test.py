"""Registry-driven dream-system auto-registration tests.

Three contracts to pin:

  1. Per-job flag gating — a flag-off job is recorded as skipped with
     the right ``reason`` AND the scheduler RPC is not called.
  2. Per-job idempotency — each job has its OWN Redis SETNX key so
     flipping one flag on after another already registered must let
     the newly-enabled job in.
  3. Per-job failure isolation — a scheduler RPC failure for one job
     does not block the other jobs in the same call.
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
_PATH_SETNX = "backend.copilot.dream.scheduling._try_redis_setnx"
_PATH_CLIENT = "backend.util.clients.get_scheduler_client"


def _mock_scheduler_client(*, fail_jobs: tuple[str, ...] = ()) -> MagicMock:
    """Build a SchedulerClient stub that returns a documented dict for
    each ``add_*_schedule`` call. Optionally make individual jobs raise
    so per-job failure isolation can be exercised.

    ``fail_jobs`` is a tuple of ``job_id_prefix`` values. Any matching
    prefix raises ``RuntimeError`` from its register call.
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
    client.add_dream_pass_schedule = _maker("dream_pass")
    client.add_ratification_pass_schedule = _maker("ratification_pass")
    return client


def _flag_map(**overrides: bool) -> dict[Flag, bool]:
    """Build a per-flag enable map; defaults to all flags on."""
    base = {
        Flag.GRAPHITI_COMMUNITIES_ENABLED: True,
        Flag.DREAM_PASS_ENABLED: True,
    }
    for k, v in overrides.items():
        base[Flag(k) if isinstance(k, str) else k] = v
    return base


def _flag_mock(flag_map: dict[Flag, bool]) -> AsyncMock:
    """Per-flag stub for ``is_feature_enabled(flag, user_id)``."""

    async def fake(flag: Flag, user_id: str) -> bool:
        return flag_map.get(flag, False)

    return AsyncMock(side_effect=fake)


# ---------------------------------------------------------------------------
# Registry shape — guard against accidental edits
# ---------------------------------------------------------------------------


def test_registry_contains_three_jobs_in_documented_order():
    """Cron-frequency order (rarest first) is part of the design — the
    log narrative reads weekly → daily → 6h."""
    prefixes = [j.job_id_prefix for j in DREAM_SYSTEM_JOBS]
    assert prefixes == ["community_rebuild", "dream_pass", "ratification_pass"]


def test_each_job_has_distinct_registration_key_prefix():
    """Per-job Redis dedup keys MUST be distinct — a shared key would
    block recovery when a flag flips on→off→on for one job but not
    the others."""
    keys = [j.registration_key_prefix for j in DREAM_SYSTEM_JOBS]
    assert len(set(keys)) == len(keys)


def test_ratification_rides_dream_pass_master_gate():
    """Ratification has no edges to process unless the dream pass
    wrote tentatives — they MUST share a flag, otherwise we'd schedule
    a ratification loop that has nothing to do."""
    rat = next(j for j in DREAM_SYSTEM_JOBS if j.job_id_prefix == "ratification_pass")
    dream = next(j for j in DREAM_SYSTEM_JOBS if j.job_id_prefix == "dream_pass")
    assert rat.flag == dream.flag == Flag.DREAM_PASS_ENABLED


# ---------------------------------------------------------------------------
# Empty / null user_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_user_id_returns_empty_dict_without_running():
    result = await ensure_dream_system_scheduled("")
    assert result == {}


# ---------------------------------------------------------------------------
# All flags on → all jobs register
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_three_jobs_register_when_their_flags_are_on():
    client = _mock_scheduler_client()
    with patch(_PATH_FLAG, new=_flag_mock(_flag_map())), patch(
        _PATH_TZ, new=AsyncMock(return_value="America/New_York")
    ), patch(_PATH_SETNX, new=AsyncMock(return_value=True)), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    assert result["community_rebuild"]["id"] == "community_rebuild_abc"
    assert result["dream_pass"]["id"] == "dream_pass_abc"
    assert result["ratification_pass"]["id"] == "ratification_pass_abc"

    # Timezone gets threaded through to every scheduler call.
    client.add_community_rebuild_schedule.assert_awaited_once_with(
        user_id="abc", user_timezone="America/New_York"
    )
    client.add_dream_pass_schedule.assert_awaited_once_with(
        user_id="abc", user_timezone="America/New_York"
    )
    client.add_ratification_pass_schedule.assert_awaited_once_with(
        user_id="abc", user_timezone="America/New_York"
    )


# ---------------------------------------------------------------------------
# Layer 1 — flag gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_skips_that_job_without_calling_scheduler():
    """Community-rebuild flag off → that job is skipped but the dream
    + ratification jobs still register. Per-job failure isolation
    means one disabled job doesn't disable the others."""
    client = _mock_scheduler_client()
    flag_map = _flag_map()
    flag_map[Flag.GRAPHITI_COMMUNITIES_ENABLED] = False
    with patch(_PATH_FLAG, new=_flag_mock(flag_map)), patch(
        _PATH_TZ, new=AsyncMock(return_value="UTC")
    ), patch(_PATH_SETNX, new=AsyncMock(return_value=True)), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    assert result["community_rebuild"] == {
        "skipped": True,
        "reason": "graphiti_communities_disabled",
    }
    assert result["dream_pass"]["id"] == "dream_pass_abc"
    assert result["ratification_pass"]["id"] == "ratification_pass_abc"
    client.add_community_rebuild_schedule.assert_not_called()


@pytest.mark.asyncio
async def test_master_gate_off_skips_dream_and_ratification_but_keeps_community():
    """``DREAM_PASS_ENABLED`` off → both dream + ratification skip,
    but community rebuild is independent and still registers."""
    client = _mock_scheduler_client()
    flag_map = _flag_map()
    flag_map[Flag.DREAM_PASS_ENABLED] = False
    with patch(_PATH_FLAG, new=_flag_mock(flag_map)), patch(
        _PATH_TZ, new=AsyncMock(return_value="UTC")
    ), patch(_PATH_SETNX, new=AsyncMock(return_value=True)), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    assert result["community_rebuild"]["id"] == "community_rebuild_abc"
    assert result["dream_pass"] == {
        "skipped": True,
        "reason": "dream_pass_disabled",
    }
    assert result["ratification_pass"] == {
        "skipped": True,
        "reason": "dream_pass_disabled",
    }
    client.add_dream_pass_schedule.assert_not_called()
    client.add_ratification_pass_schedule.assert_not_called()


@pytest.mark.asyncio
async def test_all_flags_off_returns_all_skipped_no_rpc_no_tz_lookup():
    """When nothing is enabled, we shouldn't waste a DB lookup for tz
    or an RPC to the scheduler service."""
    client = _mock_scheduler_client()
    tz_mock = AsyncMock(return_value="UTC")
    with patch(_PATH_FLAG, new=_flag_mock({})), patch(
        _PATH_TZ, new=tz_mock
    ), patch(_PATH_SETNX, new=AsyncMock(return_value=True)), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    assert all(
        r.get("skipped") is True for r in result.values()  # type: ignore[union-attr]
    )
    tz_mock.assert_not_called()
    client.add_community_rebuild_schedule.assert_not_called()
    client.add_dream_pass_schedule.assert_not_called()
    client.add_ratification_pass_schedule.assert_not_called()


# ---------------------------------------------------------------------------
# Layer 2 — Redis SETNX dedup (per-job)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_job_setnx_dedup_skips_only_the_already_registered_job():
    """SETNX returns False for dream_pass (already registered) but True
    for community_rebuild and ratification_pass — only dream_pass
    should be skipped."""
    client = _mock_scheduler_client()

    async def fake_setnx(key: str) -> bool:
        return "dream_pass" not in key  # only dream_pass already registered

    with patch(_PATH_FLAG, new=_flag_mock(_flag_map())), patch(
        _PATH_TZ, new=AsyncMock(return_value="UTC")
    ), patch(_PATH_SETNX, new=AsyncMock(side_effect=fake_setnx)), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    # Skipped via SETNX → result is None (distinct from flag-off skip).
    assert result["dream_pass"] is None
    assert result["community_rebuild"]["id"] == "community_rebuild_abc"
    assert result["ratification_pass"]["id"] == "ratification_pass_abc"
    client.add_dream_pass_schedule.assert_not_called()


@pytest.mark.asyncio
async def test_redis_unavailable_still_registers_via_scheduler_backstop():
    """If Redis is unavailable, the helper proceeds to the scheduler RPC —
    APScheduler's ``replace_existing=True`` is the durable idempotency
    backstop."""
    client = _mock_scheduler_client()
    with patch(_PATH_FLAG, new=_flag_mock(_flag_map())), patch(
        _PATH_TZ, new=AsyncMock(return_value="UTC")
    ), patch(_PATH_SETNX, new=AsyncMock(return_value=None)), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    # Every job still registers despite Redis being down.
    assert result["community_rebuild"]["id"] == "community_rebuild_abc"
    assert result["dream_pass"]["id"] == "dream_pass_abc"
    assert result["ratification_pass"]["id"] == "ratification_pass_abc"


# ---------------------------------------------------------------------------
# Layer 3 — Scheduler RPC failure isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_scheduler_rpc_failure_does_not_block_other_jobs():
    """The dream_pass scheduler call raises; community_rebuild and
    ratification_pass still register cleanly. Failed job surfaces as
    a skipped dict, not a propagated exception."""
    client = _mock_scheduler_client(fail_jobs=("dream_pass",))
    with patch(_PATH_FLAG, new=_flag_mock(_flag_map())), patch(
        _PATH_TZ, new=AsyncMock(return_value="UTC")
    ), patch(_PATH_SETNX, new=AsyncMock(return_value=True)), patch(
        _PATH_CLIENT, return_value=client
    ):
        result = await ensure_dream_system_scheduled("abc")

    assert result["community_rebuild"]["id"] == "community_rebuild_abc"
    assert result["dream_pass"] == {
        "skipped": True,
        "reason": "registration_failed",
    }
    assert result["ratification_pass"]["id"] == "ratification_pass_abc"


# ---------------------------------------------------------------------------
# Shared per-call caching
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timezone_lookup_runs_at_most_once_per_invocation():
    """Three jobs registering must share one Prisma round-trip for the
    user's timezone — anything more is waste."""
    client = _mock_scheduler_client()
    tz_mock = AsyncMock(return_value="UTC")
    with patch(_PATH_FLAG, new=_flag_mock(_flag_map())), patch(
        _PATH_TZ, new=tz_mock
    ), patch(_PATH_SETNX, new=AsyncMock(return_value=True)), patch(
        _PATH_CLIENT, return_value=client
    ):
        await ensure_dream_system_scheduled("abc")

    tz_mock.assert_awaited_once_with("abc")


# ---------------------------------------------------------------------------
# Module constants — defended against TTL drift
# ---------------------------------------------------------------------------


def test_registration_ttl_matches_longest_cron_cadence():
    """The TTL must be at least as long as the longest cron cadence in
    the registry (weekly = 7d community rebuild). Shorter creates
    redundant RPC calls; longer creates a window where the schedule
    can be deleted out-of-band without re-registration."""
    one_week_seconds = 7 * 24 * 3600
    assert REGISTRATION_TTL_SECONDS >= one_week_seconds


def test_registry_jobs_register_methods_are_distinct_callables():
    """No job's register callable should point at another job's
    SchedulerClient method — typo here = silent breakage."""
    callables = {id(j.register) for j in DREAM_SYSTEM_JOBS}
    assert len(callables) == len(DREAM_SYSTEM_JOBS)


# Suppress "imported but unused" — `scheduling` import is the module
# being tested; some linters miss the patch() string-target usage.
_ = scheduling
