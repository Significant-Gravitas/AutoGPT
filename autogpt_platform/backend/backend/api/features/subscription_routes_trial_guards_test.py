"""Choosing a plan while a trial is cancel-pending converts it in place only
while Stripe still shows that trial alone and cancel-pending, and only for the
same plan: otherwise the request is refused or goes to a new checkout."""

from unittest.mock import MagicMock

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

pytest_plugins = ("backend.data.subscription_trial_fixtures",)

client = routes_test.client
_configure_frontend_origin = routes_test._configure_frontend_origin
_stub_lifecycle_emails = routes_test._stub_lifecycle_emails
_stub_pending_subscription_change = routes_test._stub_pending_subscription_change
track_checkout_started = routes_test.track_checkout_started
_stub_subscription_status_lookups = routes_test._stub_subscription_status_lookups
cancel_pending_trial = trial_test.cancel_pending_trial
trial_conversion = trial_test.trial_conversion


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
    assert (
        response.json()["detail"] == "This trial has ended. Manage the plan in billing."
    )
    trial_conversion.modify.assert_not_awaited()
    trial_conversion.sync.assert_not_awaited()
    trial_conversion.modify_for_tier.assert_not_awaited()
    trial_conversion.checkout.assert_not_awaited()


@pytest.mark.parametrize(
    "tier,billing_cycle,pro_monthly_price",
    [
        ("MAX", "monthly", "price_pro"),
        ("PRO", "yearly", "price_pro"),
        ("PRO", "monthly", "price_pro_v2"),
    ],
)
def test_update_subscription_tier_cancel_pending_trial_other_plan_uses_checkout(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
    trial_conversion: MagicMock,
    tier: str,
    billing_cycle: str,
    pro_monthly_price: str,
) -> None:
    """Another tier, another cycle or a re-priced plan is a new subscription;
    the stale-subscription cleanup ends the trial once it is paid for."""

    async def price_id(
        requested: SubscriptionTier, cycle: str = "monthly"
    ) -> str | None:
        if requested == SubscriptionTier.PRO and cycle == "monthly":
            return pro_monthly_price
        prices = (
            _DEFAULT_TIER_PRICES_YEARLY if cycle == "yearly" else _DEFAULT_TIER_PRICES
        )
        return prices.get(requested)

    mocker.patch(
        "backend.api.features.billing.subscriptions.routes.get_subscription_price_id",
        side_effect=price_id,
    )

    response = _post_plan(client, tier, billing_cycle)

    assert response.status_code == 200
    assert response.json()["url"] == "https://checkout.stripe.com/pay/cs_test_max"
    assert trial_conversion.calls == []
    trial_conversion.modify.assert_not_awaited()
    trial_conversion.modify_for_tier.assert_not_awaited()
    trial_conversion.checkout.assert_awaited_once()
    assert trial_conversion.checkout.call_args.kwargs["tier"] == SubscriptionTier(tier)
    assert trial_conversion.checkout.call_args.kwargs["billing_cycle"] == billing_cycle
