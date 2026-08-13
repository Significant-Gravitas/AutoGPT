"""Morning-briefing auto-registration tests.

Single-job counterpart to ``backend/copilot/dream/scheduling_test.py``.
Contracts pinned here:

  1. Empty ``user_id`` → no-op, no flag check, no RPC.
  2. Flag off → no-op, no timezone lookup, no RPC.
  3. Timezone lookup failure is "unknown", not "UTC" — skip the cycle
     rather than risk re-registering onto UTC.
  4. First call (no marker) registers the cron and writes the marker.
  5. Marker present → no-op: no timezone lookup (a DB read on the hot
     chat path) and no RPC.
  6. Cleared marker (the ``update_user_timezone`` path) → re-registers
     with the current timezone and rewrites the marker.
  7. ``ensure_morning_briefing_scheduled`` never raises — it's fired
     via ``asyncio.create_task`` from a hot request path.
  8. ``_resolve_user_timezone`` behaviors copied from the dream module:
     missing user is authoritative UTC, DB failure is None (unknown),
     unset timezone falls back to UTC.
  9. ``clear_briefing_registration_marker`` deletes the single key and
     swallows Redis failures.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from . import scheduling
from .scheduling import (
    BRIEFING_REGISTRATION_PREFIX,
    REGISTRATION_TTL_SECONDS,
    clear_briefing_registration_marker,
    ensure_morning_briefing_scheduled,
)

# Paths used by the helper — patched at the module level since the helper
# does lazy imports of the scheduler client + Redis client to avoid
# bootstrap circular imports (same pattern as backend/copilot/dream).
_PATH_FLAG = "backend.copilot.briefing.scheduling.is_feature_enabled"
_PATH_TZ = "backend.copilot.briefing.scheduling._resolve_user_timezone"
_PATH_READ_TZ = "backend.copilot.briefing.scheduling._read_registration_marker"
_PATH_WRITE_TZ = "backend.copilot.briefing.scheduling._write_registration_marker"
_PATH_CLIENT = "backend.util.clients.get_scheduler_client"


def _mock_scheduler_client(*, fail: bool = False) -> MagicMock:
    client = MagicMock()
    if fail:
        client.add_morning_briefing_schedule = AsyncMock(
            side_effect=RuntimeError("scheduler down")
        )
    else:
        client.add_morning_briefing_schedule = AsyncMock(
            return_value={
                "id": "morning_briefing_abc",
                "user_id": "abc",
                "next_run_time": None,
            }
        )
    return client


# ---------------------------------------------------------------------------
# Empty / null user_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_user_id_no_op():
    flag_mock = AsyncMock(return_value=True)
    with patch(_PATH_FLAG, new=flag_mock):
        await ensure_morning_briefing_scheduled("")

    flag_mock.assert_not_called()


# ---------------------------------------------------------------------------
# Flag gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flag_off_no_op():
    client = _mock_scheduler_client()
    tz_mock = AsyncMock(return_value="UTC")
    with patch(_PATH_FLAG, new=AsyncMock(return_value=False)), patch(
        _PATH_TZ, new=tz_mock
    ), patch(_PATH_CLIENT, return_value=client):
        await ensure_morning_briefing_scheduled("abc")

    tz_mock.assert_not_called()
    client.add_morning_briefing_schedule.assert_not_called()


# ---------------------------------------------------------------------------
# Timezone lookup failure — "unknown" must not become "UTC"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timezone_lookup_failure_skips_registration():
    client = _mock_scheduler_client()
    write_spy = AsyncMock()
    with patch(_PATH_FLAG, new=AsyncMock(return_value=True)), patch(
        _PATH_TZ, new=AsyncMock(return_value=None)
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value=None)), patch(
        _PATH_WRITE_TZ, new=write_spy
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        await ensure_morning_briefing_scheduled("abc")

    client.add_morning_briefing_schedule.assert_not_called()
    write_spy.assert_not_called()


# ---------------------------------------------------------------------------
# First registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_call_registers_and_writes_marker():
    client = _mock_scheduler_client()
    write_spy = AsyncMock()
    with patch(_PATH_FLAG, new=AsyncMock(return_value=True)), patch(
        _PATH_TZ, new=AsyncMock(return_value="America/New_York")
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value=None)), patch(
        _PATH_WRITE_TZ, new=write_spy
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        await ensure_morning_briefing_scheduled("abc")

    client.add_morning_briefing_schedule.assert_awaited_once_with(
        user_id="abc", user_timezone="America/New_York"
    )
    write_spy.assert_awaited_once_with("abc", "America/New_York")


# ---------------------------------------------------------------------------
# Marker present — no-op, and crucially no DB read on the hot chat path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_existing_marker_skips_timezone_lookup_and_registration():
    client = _mock_scheduler_client()
    tz_mock = AsyncMock(return_value="America/New_York")
    with patch(_PATH_FLAG, new=AsyncMock(return_value=True)), patch(
        _PATH_TZ, new=tz_mock
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value="America/New_York")), patch(
        _PATH_CLIENT, return_value=client
    ):
        await ensure_morning_briefing_scheduled("abc")

    tz_mock.assert_not_called()
    client.add_morning_briefing_schedule.assert_not_called()


# ---------------------------------------------------------------------------
# Cleared marker (timezone change) — re-registers on the current timezone
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cleared_marker_reregisters_on_current_timezone():
    client = _mock_scheduler_client()
    write_spy = AsyncMock()
    with patch(_PATH_FLAG, new=AsyncMock(return_value=True)), patch(
        _PATH_TZ, new=AsyncMock(return_value="Europe/Paris")
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value=None)), patch(
        _PATH_WRITE_TZ, new=write_spy
    ), patch(
        _PATH_CLIENT, return_value=client
    ):
        await ensure_morning_briefing_scheduled("abc")

    client.add_morning_briefing_schedule.assert_awaited_once_with(
        user_id="abc", user_timezone="Europe/Paris"
    )
    write_spy.assert_awaited_once_with("abc", "Europe/Paris")


# ---------------------------------------------------------------------------
# Never raises — fired via asyncio.create_task from a hot request path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_rpc_failure_is_swallowed_and_logged(caplog):
    client = _mock_scheduler_client(fail=True)
    with patch(_PATH_FLAG, new=AsyncMock(return_value=True)), patch(
        _PATH_TZ, new=AsyncMock(return_value="UTC")
    ), patch(_PATH_READ_TZ, new=AsyncMock(return_value=None)), patch(
        _PATH_WRITE_TZ, new=AsyncMock()
    ), patch(
        _PATH_CLIENT, return_value=client
    ), caplog.at_level(
        logging.WARNING, logger=scheduling.logger.name
    ):
        await ensure_morning_briefing_scheduled("abc")  # must not raise

    assert any(r.levelno == logging.WARNING for r in caplog.records)


@pytest.mark.asyncio
async def test_unexpected_flag_check_error_is_swallowed():
    with patch(
        _PATH_FLAG, new=AsyncMock(side_effect=ConnectionError("LD unreachable"))
    ):
        await ensure_morning_briefing_scheduled("abc")  # must not raise


# ---------------------------------------------------------------------------
# _resolve_user_timezone — copied logic, behaviors pinned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_user_timezone_missing_user_is_authoritative_utc():
    accessor = MagicMock()
    accessor.get_user_by_id = AsyncMock(side_effect=ValueError("User not found"))
    with patch("backend.copilot.briefing.scheduling.user_db", return_value=accessor):
        assert await scheduling._resolve_user_timezone("abc") == "UTC"


@pytest.mark.asyncio
async def test_resolve_user_timezone_returns_none_when_db_lookup_fails():
    accessor = MagicMock()
    accessor.get_user_by_id = AsyncMock(side_effect=ConnectionError("db down"))
    with patch("backend.copilot.briefing.scheduling.user_db", return_value=accessor):
        assert await scheduling._resolve_user_timezone("abc") is None


@pytest.mark.asyncio
async def test_resolve_user_timezone_unset_value_falls_back_to_utc():
    from backend.data.model import USER_TIMEZONE_NOT_SET

    accessor = MagicMock()
    accessor.get_user_by_id = AsyncMock(
        return_value=MagicMock(timezone=USER_TIMEZONE_NOT_SET)
    )
    with patch("backend.copilot.briefing.scheduling.user_db", return_value=accessor):
        assert await scheduling._resolve_user_timezone("abc") == "UTC"


# ---------------------------------------------------------------------------
# clear_briefing_registration_marker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_marker_deletes_the_registration_key():
    redis = AsyncMock()
    with patch(
        "backend.copilot.briefing.scheduling.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        await clear_briefing_registration_marker("abc")
    redis.delete.assert_awaited_once_with(f"{BRIEFING_REGISTRATION_PREFIX}:abc")


@pytest.mark.asyncio
async def test_clear_marker_swallows_redis_failure(caplog):
    with patch(
        "backend.copilot.briefing.scheduling.get_redis_async",
        new=AsyncMock(side_effect=ConnectionError("redis down")),
    ), caplog.at_level(logging.WARNING, logger=scheduling.logger.name):
        await clear_briefing_registration_marker("abc")  # must not raise

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("marker will expire via TTL" in r.getMessage() for r in warnings)
    assert any(r.exc_info is not None for r in warnings)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------


def test_registration_ttl_bounds_the_recheck_window():
    assert REGISTRATION_TTL_SECONDS == 7 * 24 * 3600


# Suppress "imported but unused" — ``scheduling`` is the module under test.
_ = scheduling
