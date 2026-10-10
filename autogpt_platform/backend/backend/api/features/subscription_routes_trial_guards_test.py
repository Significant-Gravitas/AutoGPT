"""Choosing a plan while a trial is cancel-pending converts it in place only
while Stripe still shows that trial alone and cancel-pending, and only for the
same plan: otherwise the request is refused or goes to a new checkout."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import fastapi.testclient
import pytest
import pytest_mock
import stripe
from prisma.enums import SubscriptionTier

from backend.api.features import subscription_routes_test as routes_test
from backend.api.features import subscription_routes_trial_test as trial_test
from backend.api.features.subscription_routes_test import (
    _DEFAULT_TIER_PRICES,
    _DEFAULT_TIER_PRICES_YEARLY,
)
from backend.api.features.subscription_routes_trial_test import (
    TRIAL_RUNNING_DETAIL,
    _post_plan,
    _subscription_list,
)
from backend.data import credit, subscription_checkout
from backend.data.subscription_trial import TrialState

pytest_plugins = ("backend.data.subscription_trial_fixtures",)

client = routes_test.client
_configure_frontend_origin = routes_test._configure_frontend_origin
_stub_lifecycle_emails = routes_test._stub_lifecycle_emails
_stub_pending_subscription_change = routes_test._stub_pending_subscription_change
track_checkout_started = routes_test.track_checkout_started
_stub_subscription_status_lookups = routes_test._stub_subscription_status_lookups
cancel_pending_trial = trial_test.cancel_pending_trial
trial_conversion = trial_test.trial_conversion

ANOTHER_PLAN_LIVE_DETAIL = "Another plan is already active. Manage it in billing."
CHECKOUT_URL = "https://checkout.stripe.com/pay/cs_test_max"


def test_update_subscription_tier_cancel_pending_trial_card_declined_returns_402(
    client: fastapi.testclient.TestClient,
    trial_conversion: MagicMock,
) -> None:
    trial_conversion.modify.side_effect = stripe.CardError(
        "Your card was declined.", param="card", code="card_declined"
    )

    response = _post_plan(client, "PRO")

    assert response.status_code == 402
    assert "card was declined" in response.json()["detail"].lower()
    trial_conversion.sync.assert_not_awaited()
    trial_conversion.modify_for_tier.assert_not_awaited()
    trial_conversion.checkout.assert_not_awaited()


@pytest.mark.parametrize(
    "code,decline_code",
    [
        ("authentication_required", None),
        ("invoice_payment_intent_requires_action", None),
        ("subscription_payment_intent_requires_action", None),
        ("card_declined", "authentication_required"),
    ],
)
def test_update_subscription_tier_cancel_pending_trial_sca_falls_back_to_checkout(
    client: fastapi.testclient.TestClient,
    trial_conversion: MagicMock,
    code: str,
    decline_code: str | None,
) -> None:
    """Only an on-session Checkout can complete 3DS. The failed charge left the
    trial untouched, and the stale-plan cleanup ends it once the plan is paid."""
    trial_conversion.modify.side_effect = stripe.CardError(
        "Authentication required.",
        param="card",
        code=code,
        json_body={"error": {"code": code, "decline_code": decline_code}},
    )

    response = _post_plan(client, "PRO")

    assert response.status_code == 200
    assert response.json()["url"] == CHECKOUT_URL
    trial_conversion.sync.assert_not_awaited()
    trial_conversion.checkout.assert_awaited_once()
    requested = trial_conversion.checkout.call_args.kwargs
    assert requested["tier"] == SubscriptionTier.PRO
    assert requested["billing_cycle"] == "monthly"


@pytest.mark.parametrize(
    "change,detail",
    [
        ({"cancel_at_period_end": False}, TRIAL_RUNNING_DETAIL),
        ({"status": "canceled"}, "This trial has ended. Manage the plan in billing."),
        (
            {"customer": "cus_other"},
            "This trial has ended. Manage the plan in billing.",
        ),
    ],
)
def test_update_subscription_tier_cancel_pending_trial_changed_in_stripe_returns_409(
    client: fastapi.testclient.TestClient,
    trial_conversion: MagicMock,
    change: dict,
    detail: str,
) -> None:
    trial_conversion.retrieve.side_effect = None
    trial_conversion.retrieve.return_value = stripe.Subscription.construct_from(
        {**trial_conversion.live, **change}, "test-key"
    )

    response = _post_plan(client, "PRO")

    assert response.status_code == 409
    assert response.json()["detail"] == detail
    trial_conversion.expire.assert_not_awaited()
    trial_conversion.modify.assert_not_awaited()
    trial_conversion.modify_for_tier.assert_not_awaited()
    trial_conversion.checkout.assert_not_awaited()


def test_update_subscription_tier_cancel_pending_trial_with_another_live_plan_returns_409(
    client: fastapi.testclient.TestClient,
    trial_conversion: MagicMock,
) -> None:
    """A Max Checkout completed before its webhook ended the trial: converting
    the trial as well would bill the customer for two plans."""
    trial_conversion.others.side_effect = None
    trial_conversion.others.return_value = _subscription_list(
        trial_conversion.live, {"id": "sub_max", "status": "active"}
    )

    response = _post_plan(client, "PRO")

    assert response.status_code == 409
    assert response.json()["detail"] == ANOTHER_PLAN_LIVE_DETAIL
    trial_conversion.modify.assert_not_awaited()
    trial_conversion.sync.assert_not_awaited()
    trial_conversion.modify_for_tier.assert_not_awaited()
    trial_conversion.checkout.assert_not_awaited()


def test_update_subscription_tier_cancel_pending_trial_repriced_plan_converts_in_place(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
    trial_conversion: MagicMock,
) -> None:
    """The trial's own plan bills the price it accepted, as the plan card and
    the confirm dialog show, even after the plan was re-priced."""

    async def price_id(
        requested: SubscriptionTier, cycle: str = "monthly"
    ) -> str | None:
        if requested == SubscriptionTier.PRO and cycle == "monthly":
            return "price_pro_v2"
        prices = (
            _DEFAULT_TIER_PRICES_YEARLY if cycle == "yearly" else _DEFAULT_TIER_PRICES
        )
        return prices.get(requested)

    mocker.patch(
        "backend.api.features.billing.subscriptions.routes.get_subscription_price_id",
        side_effect=price_id,
    )

    response = _post_plan(client, "PRO")

    assert response.status_code == 200
    assert response.json()["url"] == ""
    trial_conversion.modify.assert_awaited_once()
    trial_conversion.sync.assert_awaited_once_with(dict(trial_conversion.converted))
    trial_conversion.checkout.assert_not_awaited()


@pytest.mark.parametrize("tier,billing_cycle", [("MAX", "monthly"), ("PRO", "yearly")])
def test_update_subscription_tier_cancel_pending_trial_other_plan_uses_checkout(
    client: fastapi.testclient.TestClient,
    trial_conversion: MagicMock,
    tier: str,
    billing_cycle: str,
) -> None:
    """Another tier or another cycle is a new subscription; the
    stale-subscription cleanup ends the trial once it is paid for."""
    response = _post_plan(client, tier, billing_cycle)

    assert response.status_code == 200
    assert response.json()["url"] == CHECKOUT_URL
    assert trial_conversion.calls == []
    trial_conversion.modify.assert_not_awaited()
    trial_conversion.modify_for_tier.assert_not_awaited()
    trial_conversion.checkout.assert_awaited_once()
    assert trial_conversion.checkout.call_args.kwargs["tier"] == SubscriptionTier(tier)
    assert trial_conversion.checkout.call_args.kwargs["billing_cycle"] == billing_cycle


@asynccontextmanager
async def _unlocked(user_id: str):
    yield


def test_update_subscription_tier_cancel_pending_trial_checkout_with_live_plan_returns_409(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
    trial_conversion: MagicMock,
    cancel_pending_trial: TrialState,
) -> None:
    """Max was paid for, but its webhook has not ended the trial yet: another
    Checkout would bill for a second plan."""
    trial_conversion.others.side_effect = None
    trial_conversion.others.return_value = _subscription_list(
        {"id": "sub_max", "status": "active", "metadata": {}}, trial_conversion.live
    )
    mocker.patch(
        "backend.api.features.billing.subscriptions.routes.create_subscription_checkout",
        credit.create_subscription_checkout,
    )
    mocker.patch.object(credit, "subscription_checkout_lock", _unlocked)
    mocker.patch.object(
        credit, "get_subscription_price_id", AsyncMock(return_value="price_max")
    )
    mocker.patch.object(
        credit, "get_stripe_customer_id", AsyncMock(return_value="cus_1")
    )
    mocker.patch.object(credit, "_expire_open_subscription_sessions", AsyncMock())
    mocker.patch.object(
        subscription_checkout,
        "get_subscription_trial",
        AsyncMock(return_value=cancel_pending_trial),
    )
    create = mocker.patch.object(stripe.checkout.Session, "create_async", AsyncMock())

    response = _post_plan(client, "MAX")

    assert response.status_code == 409
    assert response.json()["detail"] == ANOTHER_PLAN_LIVE_DETAIL
    trial_conversion.others.assert_awaited_once_with(
        customer="cus_1", status="all", limit=100
    )
    create.assert_not_awaited()
    trial_conversion.modify.assert_not_awaited()
