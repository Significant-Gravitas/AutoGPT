"""POST /api/chat/usage/extra-week: who may buy, what is charged, and what a
refusal leaves untouched. The purchase's transaction is tested against
Postgres in backend/data/usage_extra_integration_test.py."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from prisma.enums import SubscriptionTier as PrismaTier

from backend.api.features.chat import usage_extra as routes
from backend.api.features.chat.usage_extra import (
    ExtraWeekPurchaseRequest,
    buy_extra_week_of_usage,
)
from backend.copilot import usage_limits
from backend.copilot.rate_limit import (
    CoPilotUsageStatus,
    RateLimitUnavailable,
    SubscriptionTier,
    UsageWindow,
)
from backend.data.usage_extra import ExtraWeek, ExtraWeekPurchase
from backend.util.exceptions import InsufficientBalanceError
from backend.util.posthog_events import PostHogEvent

NOW = 1_793_824_256
KEY = "0b0d6a52-6f0e-4a55-9d1f-2f6a2b1c9e11"
REQUEST = ExtraWeekPurchaseRequest(purchase_key=KEY)


def _purchase(charged: bool = True, tier: str = "MAX") -> ExtraWeekPurchase:
    week = ExtraWeek(
        id="week-1",
        tier=tier,
        amount=68_000_000,
        used=0,
        price_credits=8_000,
        purchased_at=NOW,
        expires_at=NOW + 2_592_000,
    )
    return ExtraWeekPurchase(extra_week=week, balance=2_000, charged=charged)


class Shop:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        usage_limits._fetch_plan_limits_flag.cache_clear()
        resets_at = datetime.now(UTC) + timedelta(days=1)
        window = UsageWindow(used=0, limit=1, resets_at=resets_at)
        self.live = AsyncMock(return_value=True)
        self.limits = AsyncMock(
            return_value=(22_666_667, 68_000_000, SubscriptionTier.MAX)
        )
        self.buy = AsyncMock(return_value=_purchase())
        self.status = AsyncMock(
            return_value=CoPilotUsageStatus(daily=window, weekly=window)
        )
        self.events = MagicMock()
        monkeypatch.setattr(routes, "billing_clock_live", self.live)
        monkeypatch.setattr(routes, "now_seconds", lambda: NOW)
        monkeypatch.setattr(routes, "get_global_rate_limits", self.limits)
        monkeypatch.setattr(routes, "buy_extra_week", self.buy)
        monkeypatch.setattr(routes, "get_usage_status", self.status)
        monkeypatch.setattr(routes.posthog_client, "capture", self.events)
        monkeypatch.setattr(routes.settings.config, "enable_credit", True)

    def plan(self, tier: str, weekly: int = 8_000_000) -> None:
        self.limits.return_value = (3_200_000, weekly, SubscriptionTier(tier))

    def event(self) -> tuple[PostHogEvent, dict]:
        _, name, properties = self.events.call_args.args
        return name, properties


@pytest.fixture
def shop(monkeypatch: pytest.MonkeyPatch) -> Shop:
    return Shop(monkeypatch)


@pytest.mark.asyncio
async def test_a_max_subscriber_buys_a_week_of_their_weekly_limit_for_80_dollars(
    shop: Shop,
):
    response = await buy_extra_week_of_usage(REQUEST, "user-1")

    shop.buy.assert_awaited_once_with(
        "user-1",
        KEY,
        tier=PrismaTier.MAX,
        amount=68_000_000,
        price_credits=8_000,
        now=NOW,
    )
    assert (response.credits_charged, response.remaining_balance) == (8_000, 2_000)
    assert response.expires_at == datetime.fromtimestamp(NOW + 30 * 86_400, UTC)
    assert shop.event() == (
        PostHogEvent.USAGE_EXTRA_WEEK_PURCHASED,
        {"tier": "MAX", "price_credits": 8_000},
    )


@pytest.mark.asyncio
async def test_a_pro_subscriber_pays_12_50_for_their_own_weekly_limit(shop: Shop):
    shop.plan("PRO")
    shop.buy.return_value = _purchase(tier="PRO")

    await buy_extra_week_of_usage(REQUEST, "user-1")

    assert shop.buy.await_args.kwargs["amount"] == 8_000_000
    assert shop.buy.await_args.kwargs["price_credits"] == 1_250


@pytest.mark.asyncio
async def test_a_repeated_key_reports_nothing_charged_and_sends_no_second_event(
    shop: Shop,
):
    shop.buy.return_value = _purchase(charged=False)

    response = await buy_extra_week_of_usage(REQUEST, "user-1")

    assert response.credits_charged == 0
    shop.events.assert_not_called()


@pytest.mark.asyncio
async def test_too_few_credits_is_a_402_that_says_what_is_needed(shop: Shop):
    shop.buy.side_effect = InsufficientBalanceError(
        "Insufficient balance", "user-1", balance=300, amount=-8_000
    )

    with pytest.raises(HTTPException) as refused:
        await buy_extra_week_of_usage(REQUEST, "user-1")

    assert refused.value.status_code == 402
    assert refused.value.detail == {
        "message": "Not enough credits for an extra week.",
        "needed": 8_000,
        "balance": 300,
    }
    assert shop.event()[1]["reason"] == "insufficient_credits"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tier", ["TRIAL", "NO_TIER", "BASIC", "BUSINESS", "ENTERPRISE"]
)
async def test_a_plan_that_does_not_sell_one_cannot_buy_one(shop: Shop, tier: str):
    shop.plan(tier)

    with pytest.raises(HTTPException) as refused:
        await buy_extra_week_of_usage(REQUEST, "user-1")

    assert refused.value.status_code == 409
    shop.buy.assert_not_awaited()
    assert shop.event()[1] == {"reason": "not_eligible", "tier": tier}


@pytest.mark.asyncio
@pytest.mark.parametrize("weekly", [0, -1])
async def test_a_plan_without_a_capped_week_cannot_buy_one(shop: Shop, weekly: int):
    shop.plan("PRO", weekly=weekly)

    with pytest.raises(HTTPException) as refused:
        await buy_extra_week_of_usage(REQUEST, "user-1")

    assert refused.value.status_code == 409
    shop.buy.assert_not_awaited()


@pytest.mark.asyncio
async def test_nothing_is_sold_before_the_launch(shop: Shop):
    shop.live.return_value = False

    with pytest.raises(HTTPException) as refused:
        await buy_extra_week_of_usage(REQUEST, "user-1")

    assert refused.value.status_code == 409
    shop.buy.assert_not_awaited()
    assert shop.event()[1]["reason"] == "not_live"


@pytest.mark.asyncio
async def test_nothing_is_sold_with_credits_off(
    shop: Shop, monkeypatch: pytest.MonkeyPatch
):
    """With credits off a spend charges nothing: a sale would be a free week."""
    monkeypatch.setattr(routes.settings.config, "enable_credit", False)

    with pytest.raises(HTTPException) as refused:
        await buy_extra_week_of_usage(REQUEST, "user-1")

    assert refused.value.status_code == 409
    shop.buy.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unreadable_plan_is_a_503_and_sells_nothing(shop: Shop):
    shop.limits.side_effect = RateLimitUnavailable("activation is processing")

    with pytest.raises(HTTPException) as refused:
        await buy_extra_week_of_usage(REQUEST, "user-1")

    assert refused.value.status_code == 503
    assert refused.value.headers == {"Retry-After": "30"}
    shop.buy.assert_not_awaited()


@pytest.mark.parametrize(
    "key", ["short", "x" * 65, "has spaces in it", "semi;colon;key"]
)
def test_a_purchase_key_must_look_like_an_id(key: str):
    with pytest.raises(ValueError):
        ExtraWeekPurchaseRequest(purchase_key=key)
