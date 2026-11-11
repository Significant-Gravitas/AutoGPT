"""POST /api/chat/usage/paid-reset and its confirm: who may buy, how it is
paid, and what a refusal leaves untouched. The purchase's own transactions
are tested against Postgres in backend/data/usage_reset_integration_test.py."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import stripe
from fastapi import HTTPException
from prisma.enums import SubscriptionTier as PrismaTier
from prisma.errors import UniqueViolationError

from backend.api.features.chat import usage_reset as routes
from backend.api.features.chat.usage_reset import (
    buy_usage_reset,
    confirm_usage_reset,
    settle_card_reset,
)
from backend.api.features.chat.usage_reset_models import (
    UsageResetConfirmRequest,
    UsageResetRequest,
)
from backend.copilot.rate_limit import (
    CoPilotUsageStatus,
    RateLimitUnavailable,
    SubscriptionTier,
    UsageWindow,
)
from backend.data.usage_reset import CreditReset, ResetPurchase, ResetWindowChanged
from backend.data.usage_reset_card import CardCheckout, CardGrant, CheckoutUnsettled
from backend.util.exceptions import InsufficientBalanceError
from backend.util.posthog_events import PostHogEvent

NOW = 1_793_824_256
WEEK_START = NOW - 3 * 86_400
WEEK_END = WEEK_START + 7 * 86_400 + 12 * 3600
KEY = "0b0d6a52-6f0e-4a55-9d1f-2f6a2b1c9e11"
REQUEST = UsageResetRequest(purchase_key=KEY, price_credits=8_000)
SESSION = "cs_test_a1b2c3"
CHECKOUT_URL = "https://checkout.stripe.test/pay"


def _purchase(
    status: str = "granted", paid_with: str = "credits", **fields
) -> ResetPurchase:
    values = {
        "id": "reset-1",
        "user_id": "user-1",
        "tier": "MAX",
        "price_credits": 8_000,
        "paid_with": paid_with,
        "status": status,
        "week_start": WEEK_START if status == "granted" else None,
        "week_end": WEEK_END if status == "granted" else None,
        "sequence": 1 if status == "granted" else None,
        "granted_at": NOW if status == "granted" else None,
    }
    return ResetPurchase.model_validate({**values, **fields})


def _status(**fields) -> CoPilotUsageStatus:
    """A Max subscriber stopped by their weekly limit, days before its reset."""
    resets_at = datetime.fromtimestamp(WEEK_END, UTC)
    values = {
        "daily": UsageWindow(used=1, limit=1, resets_at=resets_at),
        "weekly": UsageWindow(used=1, limit=1, resets_at=resets_at),
        "tier": SubscriptionTier.MAX,
        "on_billing_clock": True,
        "blocked_by": "weekly",
        "reset_price": 8_000,
        "reset_refusal": None,
        "reset_count": 0,
        "week_start": WEEK_START,
    }
    return CoPilotUsageStatus.model_validate({**values, **fields})


class Shop:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.live = AsyncMock(return_value=True)
        self.limits = AsyncMock(
            return_value=(22_666_667, 68_000_000, SubscriptionTier.MAX)
        )
        self.status = AsyncMock(return_value=_status())
        self.earlier = AsyncMock(return_value=None)
        self.credits = MagicMock(get_credits=AsyncMock(return_value=10_000))
        self.buy = AsyncMock(
            return_value=CreditReset(purchase=_purchase(), balance=2_000, charged=True)
        )
        self.checkout = AsyncMock(
            return_value=CardCheckout(
                purchase=_purchase("pending", "card"), url=CHECKOUT_URL
            )
        )
        self.fulfill = AsyncMock(return_value=None)
        self.events = MagicMock()
        monkeypatch.setattr(routes, "billing_clock_live", self.live)
        monkeypatch.setattr(routes, "now_seconds", lambda: NOW)
        monkeypatch.setattr(routes, "get_global_rate_limits", self.limits)
        monkeypatch.setattr(routes, "get_usage_status", self.status)
        monkeypatch.setattr(routes, "find_reset_purchase", self.earlier)
        monkeypatch.setattr(
            routes, "get_user_credit_model", AsyncMock(return_value=self.credits)
        )
        monkeypatch.setattr(routes, "buy_reset_with_credits", self.buy)
        monkeypatch.setattr(routes, "start_card_reset", self.checkout)
        monkeypatch.setattr(routes, "fulfill_card_reset", self.fulfill)
        monkeypatch.setattr(routes.posthog_client, "capture", self.events)
        monkeypatch.setattr(routes.settings.config, "enable_credit", True)

    def balance(self, credits: int) -> None:
        self.credits.get_credits.return_value = credits

    def event(self) -> tuple[PostHogEvent, dict]:
        _, name, properties = self.events.call_args.args
        return name, properties

    def nothing_was_sold(self) -> None:
        self.buy.assert_not_awaited()
        self.checkout.assert_not_awaited()


@pytest.fixture
def shop(monkeypatch: pytest.MonkeyPatch) -> Shop:
    return Shop(monkeypatch)


async def _refused_after_trying_credits(shop: Shop) -> HTTPException:
    with pytest.raises(HTTPException) as refused:
        await buy_usage_reset(REQUEST, "user-1")
    shop.checkout.assert_not_awaited()
    return refused.value


async def _refused(shop: Shop, request: UsageResetRequest = REQUEST) -> HTTPException:
    with pytest.raises(HTTPException) as refused:
        await buy_usage_reset(request, "user-1")
    shop.nothing_was_sold()
    return refused.value


@pytest.mark.asyncio
async def test_a_balance_that_covers_the_price_pays_all_of_it(shop: Shop):
    shop.balance(8_000)

    response = await buy_usage_reset(REQUEST, "user-1")

    shop.buy.assert_awaited_once_with(
        "user-1",
        KEY,
        tier=PrismaTier.MAX,
        price_credits=8_000,
        week_start=WEEK_START,
        week_end=WEEK_END,
        sequence=1,
        now=NOW,
    )
    shop.checkout.assert_not_awaited()
    assert (response.status, response.paid_with) == ("granted", "credits")
    assert response.credits_charged == 8_000 and response.checkout_url is None
    # It lasts until the weekly reset the customer already had.
    assert response.resets_at == datetime.fromtimestamp(WEEK_END, UTC)
    assert shop.event() == (
        PostHogEvent.USAGE_RESET_PURCHASED,
        {"tier": "MAX", "price_credits": 8_000, "paid_with": "credits"},
    )


@pytest.mark.asyncio
async def test_a_balance_one_credit_short_pays_none_of_it_the_card_pays_all(
    shop: Shop,
):
    """Never part from each: the whole price goes to the card."""
    shop.balance(7_999)

    response = await buy_usage_reset(REQUEST, "user-1")

    shop.buy.assert_not_awaited()
    shop.checkout.assert_awaited_once_with(
        "user-1", KEY, tier=PrismaTier.MAX, price_credits=8_000, now=NOW
    )
    assert (response.status, response.paid_with) == ("payment_required", "card")
    assert response.checkout_url == CHECKOUT_URL
    # Nothing is reset and nothing is charged until Stripe confirms.
    assert response.credits_charged == 0 and response.usage is None
    assert shop.event() == (
        PostHogEvent.USAGE_RESET_CHECKOUT_STARTED,
        {"tier": "MAX", "price_credits": 8_000},
    )


@pytest.mark.asyncio
async def test_a_second_reset_in_the_window_is_numbered_as_the_second(shop: Shop):
    shop.status.return_value = _status(reset_count=1, reset_active=True)

    await buy_usage_reset(REQUEST, "user-1")

    assert shop.buy.await_args.kwargs["sequence"] == 2


@pytest.mark.asyncio
async def test_a_balance_spent_meanwhile_falls_back_to_the_card(shop: Shop):
    shop.buy.side_effect = InsufficientBalanceError(
        message="spent", user_id="user-1", balance=10, amount=8_000
    )

    response = await buy_usage_reset(REQUEST, "user-1")

    assert response.status == "payment_required"
    shop.checkout.assert_awaited_once()


@pytest.mark.asyncio
async def test_no_checkout_is_opened_if_the_window_was_reset_meanwhile(shop: Shop):
    """The balance went on another reset of this window (a second tab): by
    the time the card would be asked, no limit is reached any more."""
    shop.buy.side_effect = InsufficientBalanceError(
        message="spent", user_id="user-1", balance=10, amount=8_000
    )
    shop.status.side_effect = [
        _status(),
        _status(blocked_by=None, reset_refusal="not_limited", reset_active=True),
    ]

    refused = await _refused_after_trying_credits(shop)

    assert refused.status_code == 409
    assert shop.event()[1]["reason"] == "not_limited"


@pytest.mark.asyncio
async def test_a_key_that_started_a_card_payment_at_the_same_moment_stays_on_it(
    shop: Shop,
):
    shop.buy.side_effect = UniqueViolationError(
        {"user_facing_error": {"meta": {"target": ["purchaseKey"]}}}
    )
    shop.earlier.side_effect = [None, _purchase("pending", "card")]

    response = await buy_usage_reset(REQUEST, "user-1")

    assert response.status == "payment_required"
    assert response.checkout_url == CHECKOUT_URL


@pytest.mark.asyncio
async def test_credits_are_not_taken_from_a_customer_who_confirmed_a_card_payment(
    shop: Shop,
):
    """They were shown "pay by card"; a top-up landed before they confirmed.
    Taking credits now is not what they agreed to: it is refused, and the
    app shows the offer again."""
    by_card = UsageResetRequest(purchase_key=KEY, price_credits=8_000, pay_with="card")

    refused = await _refused(shop, by_card)

    assert (refused.status_code, refused.detail) == (409, "payment_changed")
    assert shop.event()[1]["reason"] == "payment_changed"


@pytest.mark.asyncio
async def test_a_card_confirmation_with_too_few_credits_goes_to_the_card(shop: Shop):
    shop.balance(100)
    by_card = UsageResetRequest(purchase_key=KEY, price_credits=8_000, pay_with="card")

    response = await buy_usage_reset(by_card, "user-1")

    assert response.status == "payment_required"
    shop.buy.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_retry_racing_its_own_grant_gets_the_purchase_not_a_refusal(
    shop: Shop,
):
    """The first request with this key granted the reset between this one's
    look for the purchase and its read of the usage, which now says nothing
    is limited."""
    shop.earlier.side_effect = [None, _purchase()]
    shop.status.return_value = _status(
        blocked_by=None, reset_refusal="not_limited", reset_active=True
    )

    response = await buy_usage_reset(REQUEST, "user-1")

    assert (response.status, response.credits_charged) == ("granted", 0)
    shop.nothing_was_sold()
    shop.events.assert_not_called()


@pytest.mark.asyncio
async def test_a_window_that_changed_as_the_charge_began_charges_nothing(shop: Shop):
    """An upgrade or a renewal landed between the check and the charge."""
    shop.buy.side_effect = ResetWindowChanged()

    with pytest.raises(HTTPException) as refused:
        await buy_usage_reset(REQUEST, "user-1")

    assert refused.value.status_code == 409
    shop.checkout.assert_not_awaited()
    assert shop.event()[1]["reason"] == "window_changed"


@pytest.mark.asyncio
async def test_a_replayed_credit_purchase_reports_nothing_charged(shop: Shop):
    """The same key committed a moment earlier in another request."""
    shop.buy.return_value = CreditReset(
        purchase=_purchase(), balance=2_000, charged=False
    )

    response = await buy_usage_reset(REQUEST, "user-1")

    assert (response.status, response.credits_charged) == ("granted", 0)
    shop.events.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [stripe.APIConnectionError("stripe is down"), CheckoutUnsettled("in flight")],
)
async def test_a_checkout_that_fails_to_open_charges_nothing_and_can_be_retried(
    shop: Shop, failure: Exception
):
    shop.balance(0)
    shop.checkout.side_effect = failure

    with pytest.raises(HTTPException) as failed:
        await buy_usage_reset(REQUEST, "user-1")

    assert failed.value.status_code == 503
    shop.buy.assert_not_awaited()
    shop.events.assert_not_called()


@pytest.mark.asyncio
async def test_a_window_reset_by_another_purchase_meanwhile_is_not_sold_twice(
    shop: Shop,
):
    shop.buy.side_effect = UniqueViolationError(
        {"user_facing_error": {"meta": {"target": ["sequence"]}}}
    )

    with pytest.raises(HTTPException) as refused:
        await buy_usage_reset(REQUEST, "user-1")
    # This key bought nothing: it was the window's number that was taken.
    assert shop.earlier.await_count == 2

    assert refused.value.status_code == 409
    shop.checkout.assert_not_awaited()
    assert shop.event()[1]["reason"] == "not_limited"


@pytest.mark.asyncio
async def test_a_repeated_key_gets_its_purchase_back_and_is_charged_nothing(
    shop: Shop,
):
    shop.earlier.return_value = _purchase()
    # Even though the reset it bought means no limit is reached any more.
    shop.status.return_value = _status(blocked_by=None, reset_refusal="not_limited")

    response = await buy_usage_reset(REQUEST, "user-1")

    assert (response.status, response.credits_charged) == ("granted", 0)
    shop.nothing_was_sold()
    shop.events.assert_not_called()


@pytest.mark.asyncio
async def test_a_key_whose_checkout_is_open_goes_back_to_the_card(shop: Shop):
    """Even if the balance covers the price by now: the purchase was started
    as a card payment and stays one."""
    shop.earlier.return_value = _purchase("pending", "card")

    response = await buy_usage_reset(REQUEST, "user-1")

    shop.buy.assert_not_awaited()
    assert response.checkout_url == CHECKOUT_URL


@pytest.mark.asyncio
async def test_a_key_whose_checkout_was_paid_meanwhile_is_granted(shop: Shop):
    shop.balance(0)
    shop.checkout.return_value = CardCheckout(
        purchase=_purchase("granted", "card"), url=None
    )

    response = await buy_usage_reset(REQUEST, "user-1")

    assert (response.status, response.paid_with) == ("granted", "card")
    assert response.credits_charged == 0


@pytest.mark.asyncio
async def test_before_launch_nothing_is_sold(shop: Shop):
    shop.live.return_value = False

    refused = await _refused(shop)

    assert refused.status_code == 409
    assert shop.event() == (
        PostHogEvent.USAGE_RESET_REFUSED,
        {"reason": "not_live", "tier": None},
    )
    shop.status.assert_not_awaited()


@pytest.mark.asyncio
async def test_with_credits_switched_off_nothing_is_sold(
    shop: Shop, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(routes.settings.config, "enable_credit", False)

    assert (await _refused(shop)).status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,reason",
    [
        # Limits still on the calendar for this account.
        ({"on_billing_clock": False}, "not_live"),
        # A plan that doesn't price a reset.
        ({"reset_price": 0}, "not_eligible"),
        # No limit reached: a reset would throw allowance away.
        ({"blocked_by": None, "reset_refusal": "not_limited"}, "not_limited"),
        # The weekly reset is minutes away.
        ({"reset_refusal": "resets_soon"}, "resets_soon"),
    ],
)
async def test_what_cannot_be_used_is_not_sold(shop: Shop, status: dict, reason: str):
    shop.status.return_value = _status(**status)

    refused = await _refused(shop)

    assert refused.status_code == 409
    assert shop.event() == (
        PostHogEvent.USAGE_RESET_REFUSED,
        {"reason": reason, "tier": "MAX"},
    )


@pytest.mark.asyncio
async def test_a_price_other_than_the_one_shown_is_refused(shop: Shop):
    shown_before = UsageResetRequest(purchase_key=KEY, price_credits=1_250)

    refused = await _refused(shop, shown_before)

    assert (refused.status_code, refused.detail) == (409, "price_changed")


@pytest.mark.asyncio
async def test_limits_that_cannot_be_read_sell_nothing(shop: Shop):
    shop.status.side_effect = RateLimitUnavailable("down")

    refused = await _refused(shop)

    assert refused.status_code == 503
    assert shop.status.await_args.kwargs["fail_closed"] is True


@pytest.mark.asyncio
async def test_a_purchase_stands_when_usage_cannot_be_read_back(shop: Shop):
    shop.status.side_effect = [_status(), RateLimitUnavailable("down")]

    response = await buy_usage_reset(REQUEST, "user-1")

    assert response.status == "granted" and response.usage is None


@pytest.mark.asyncio
async def test_a_checkout_that_cannot_be_opened_is_an_error_not_a_grant(shop: Shop):
    shop.balance(0)
    shop.checkout.return_value = CardCheckout(
        purchase=_purchase("pending", "card"), url=None
    )

    with pytest.raises(HTTPException) as failed:
        await buy_usage_reset(REQUEST, "user-1")

    assert failed.value.status_code == 503


@pytest.mark.asyncio
async def test_returning_from_a_paid_checkout_grants_the_reset(shop: Shop):
    shop.fulfill.return_value = CardGrant(
        purchase=_purchase("granted", "card"), granted_now=True
    )

    response = await confirm_usage_reset(
        UsageResetConfirmRequest(session_id=SESSION), "user-1"
    )

    # Only the returning customer's own session counts.
    shop.fulfill.assert_awaited_once_with(SESSION, NOW, user_id="user-1")
    assert response.status == "granted"
    assert response.resets_at == datetime.fromtimestamp(WEEK_END, UTC)
    assert shop.event() == (
        PostHogEvent.USAGE_RESET_PURCHASED,
        {"tier": "MAX", "price_credits": 8_000, "paid_with": "card"},
    )


@pytest.mark.asyncio
async def test_returning_without_having_paid_grants_nothing(shop: Shop):
    """Declined, abandoned, 3-D Secure not finished, or not their session."""
    response = await confirm_usage_reset(
        UsageResetConfirmRequest(session_id=SESSION), "user-1"
    )

    assert response.status == "pending" and response.usage is None
    shop.events.assert_not_called()


@pytest.mark.asyncio
async def test_a_grant_the_webhook_already_made_is_not_counted_twice(shop: Shop):
    shop.fulfill.return_value = CardGrant(
        purchase=_purchase("granted", "card"), granted_now=False
    )

    purchase = await settle_card_reset(SESSION)

    assert purchase is not None and purchase.status == "granted"
    shop.fulfill.assert_awaited_once_with(SESSION, NOW, user_id=None)
    shop.events.assert_not_called()


@pytest.mark.asyncio
async def test_a_confirmed_grant_stands_when_limits_cannot_be_read(shop: Shop):
    shop.fulfill.return_value = CardGrant(
        purchase=_purchase("granted", "card"), granted_now=True
    )
    shop.limits.side_effect = RateLimitUnavailable("down")

    response = await confirm_usage_reset(
        UsageResetConfirmRequest(session_id=SESSION), "user-1"
    )

    assert response.status == "granted" and response.usage is None


@pytest.mark.parametrize("key", ["short", "has spaces in it", "x" * 65, "semi;colon"])
def test_a_purchase_key_is_a_plain_token(key: str):
    with pytest.raises(ValueError):
        UsageResetRequest(purchase_key=key, price_credits=8_000)


def test_both_routes_are_behind_the_paywall():
    paths = {route.path: route for route in routes.router.routes}

    for path in ("/usage/paid-reset", "/usage/paid-reset/confirm"):
        dependencies = [d.call for d in paths[path].dependant.dependencies]
        assert routes.enforce_payment_paywall in dependencies
