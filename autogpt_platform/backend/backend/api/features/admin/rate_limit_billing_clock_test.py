"""What an admin sees of a user's limits once they run on the billing clock."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from backend.api.features.admin import rate_limit_admin_routes as routes
from backend.copilot.rate_limit import CoPilotUsageStatus, UsageWindow
from backend.data.usage_reset import ResetPurchase

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
    purchases = AsyncMock()
    monkeypatch.setattr(routes, "list_reset_purchases", purchases)

    assert await routes._billing_clock_usage("user-1", _status()) is None
    purchases.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_clock_its_resets_the_reserve_and_paid_resets_are_shown(
    monkeypatch,
):
    paid = ResetPurchase(
        id="reset-1",
        user_id="user-1",
        tier="MAX",
        price_credits=8_000,
        paid_with="card",
        status="granted",
        week_start=1_793_824_256,
        week_end=1_794_472_256,
        sequence=1,
        granted_at=1_794_000_000,
        checkout_session_id="cs_test_1",
        payment_intent_id="pi_test_1",
    )
    abandoned = ResetPurchase(
        id="reset-2",
        user_id="user-1",
        tier="MAX",
        price_credits=8_000,
        paid_with="card",
        status="pending",
        checkout_session_id="cs_test_2",
    )
    monkeypatch.setattr(
        routes, "list_reset_purchases", AsyncMock(return_value=[abandoned, paid])
    )
    status = _status(
        on_billing_clock=True,
        clock_source="held",
        weekly_limit=68_000_000,
        reserve=UsageWindow(used=2_000_000, limit=2_000_000, resets_at=DAILY_RESET),
        blocked_by="weekly",
        reset_active=True,
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
        "reset_active": True,
        # How each was paid, and the Stripe ids a refund needs.
        "reset_purchases": [abandoned.model_dump(), paid.model_dump()],
    }
