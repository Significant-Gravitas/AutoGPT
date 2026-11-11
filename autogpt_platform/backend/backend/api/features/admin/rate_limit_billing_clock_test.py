"""What an admin sees of a user's limits once they run on the billing clock."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from backend.api.features.admin import rate_limit_admin_routes as routes
from backend.copilot.rate_limit import CoPilotUsageStatus, UsageWindow
from backend.data.usage_extra import ExtraWeek

DAILY_RESET = datetime(2026, 11, 5, 20, 30, 56, tzinfo=UTC)
WEEKLY_RESET = datetime(2026, 11, 12, 8, 30, 56, tzinfo=UTC)


def _status(**fields) -> CoPilotUsageStatus:
    return CoPilotUsageStatus(
        daily=UsageWindow(used=2_000_000, limit=22_666_667, resets_at=DAILY_RESET),
        weekly=UsageWindow(used=53_000_000, limit=53_000_000, resets_at=WEEKLY_RESET),
        **fields,
    )


@pytest.mark.asyncio
async def test_on_the_calendar_there_is_no_billing_clock_to_show(monkeypatch):
    weeks = AsyncMock()
    monkeypatch.setattr(routes, "list_extra_weeks", weeks)

    assert await routes._billing_clock_usage("user-1", _status()) is None
    weeks.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_clock_its_resets_the_reserve_and_the_extra_weeks_are_shown(
    monkeypatch,
):
    week = ExtraWeek(
        id="week-1",
        tier="MAX",
        amount=68_000_000,
        used=5_000_000,
        price_credits=8_000,
        purchased_at=1_793_000_000,
        expires_at=1_795_592_000,
    )
    monkeypatch.setattr(routes, "list_extra_weeks", AsyncMock(return_value=[week]))
    monkeypatch.setattr(routes, "now_seconds", lambda: 1_793_824_256)
    status = _status(
        on_billing_clock=True,
        clock_source="held",
        weekly_limit=68_000_000,
        reserve=UsageWindow(used=2_000_000, limit=2_000_000, resets_at=DAILY_RESET),
        blocked_by="weekly",
    )

    shown = await routes._billing_clock_usage("user-1", status)

    assert shown is not None
    assert shown.model_dump() == {
        "source": "held",
        "daily_resets_at": DAILY_RESET,
        "weekly_resets_at": WEEKLY_RESET,
        "weekly_limit_microdollars": 68_000_000,
        "reserve_today_microdollars": 2_000_000,
        "reserve_used_microdollars": 2_000_000,
        "blocked_by": "weekly",
        "extra_weeks": [week.model_dump()],
    }
