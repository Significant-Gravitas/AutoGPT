"""Reading a task's start time from what Otto asked for, and saying it back."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.start_time import describe_start, resolve_start_time, user_timezone

_NOW = datetime(2026, 11, 15, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "seconds, at, expected",
    [
        (None, None, None),
        # Under a minute, or already past, is "now".
        (30, None, None),
        (-5, None, None),
        (None, "2026-11-15T11:00:00+00:00", None),
        (300, None, _NOW + timedelta(seconds=300)),
        (None, "2026-11-15T13:30:00Z", datetime(2026, 11, 15, 13, 30, tzinfo=UTC)),
    ],
)
def test_a_start_under_a_minute_away_or_past_starts_now(seconds, at, expected):
    assert resolve_start_time(seconds, at, "UTC", _NOW) == expected


def test_a_time_without_an_offset_is_read_in_the_users_time_zone():
    at = resolve_start_time(None, "2026-11-16T09:00", "America/New_York", _NOW)

    # 09:00 in New York in November (EST, UTC-5) is 14:00 UTC.
    assert at == datetime(2026, 11, 16, 14, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "seconds, at, message",
    [
        (120, "2026-11-16T09:00", "not both"),
        (None, "tomorrow morning", "ISO date-time"),
        (400 * 24 * 3600, None, "a year ahead"),
    ],
)
def test_a_malformed_start_is_refused_with_a_reason(seconds, at, message):
    with pytest.raises(ValueError, match=message):
        resolve_start_time(seconds, at, "UTC", _NOW)


def test_the_start_is_said_back_in_the_users_time_zone():
    at = datetime(2026, 11, 16, 14, 0, tzinfo=UTC)

    assert describe_start(at, "America/New_York") == (
        "Mon 16 Nov 2026, 09:00 America/New_York"
    )


@pytest.mark.parametrize(
    "saved, expected",
    [("Europe/London", ("Europe/London", True)), (None, ("UTC", False))],
)
async def test_the_users_time_zone_falls_back_to_utc(saved, expected):
    users = MagicMock()
    users.get_user_by_id = AsyncMock(return_value=MagicMock(timezone=saved))
    with patch("backend.copilot.start_time.user_db", return_value=users):
        assert await user_timezone("user-1") == expected


async def test_a_missing_user_reads_times_in_utc():
    users = MagicMock()
    users.get_user_by_id = AsyncMock(side_effect=ValueError("gone"))
    with patch("backend.copilot.start_time.user_db", return_value=users):
        assert await user_timezone("user-1") == ("UTC", False)
