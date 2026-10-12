"""Choosing a plan while a trial is cancel-pending converts the trial's own
subscription, whatever the plan, and only while Stripe still shows that trial
alone and cancel-pending. Stripe limits a customer to one subscription, so no
plan is ever bought through Checkout beside the trial."""

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
from backend.data.subscription_trial_conversion import PENDING_UPDATE_API_VERSION

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
CARD_NEEDED_DETAIL = (
    "Your saved card can't be charged. Update it under Payment method,"
    " then try again."
)
INVOICE_URL = "https://invoice.stripe.com/i/acct_1/test_pay"


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
def test_update_subscription_tier_cancel_pending_trial_sca_waits_on_stripes_invoice_page(
    client: fastapi.testclient.TestClient,
    trial_conversion: MagicMock,
    code: str,
    decline_code: str | None,
) -> None:
    """Checkout would refuse a second subscription, so the same change is held
    as a pending update: nothing changes until the customer pays its invoice,
    on the page returned as ``url``."""
    held = stripe.Subscription.construct_from(
        {
            **trial_conversion.live,
            "pending_update": {"trial_end": 1, "expires_at": 2},
            "latest_invoice": {
                "id": "in_held",
                "status": "open",
                "created": 1,
                "hosted_invoice_url": INVOICE_URL,
            },
        },
        "test-key",
    )
    trial_conversion.modify.side_effect = [
        stripe.CardError(
            "Authentication required.",
            param="card",
            code=code,
            json_body={"error": {"code": code, "decline_code": decline_code}},
        ),
        held,
    ]

    response = _post_plan(client, "PRO")

    assert response.status_code == 200
    assert response.json()["url"] == INVOICE_URL
    assert trial_conversion.modify.await_args_list[-1].kwargs == {
        "cancel_at_period_end": False,
        "trial_end": "now",
        "payment_behavior": "pending_if_incomplete",
        "expand": ["latest_invoice"],
        "stripe_version": PENDING_UPDATE_API_VERSION,
    }
    trial_conversion.sync.assert_not_awaited()
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


@pytest.mark.parametrize("tier", ["PRO", "MAX"])
def test_update_subscription_tier_cancel_pending_trial_with_another_live_plan_returns_409(
    client: fastapi.testclient.TestClient,
    trial_conversion: MagicMock,
    tier: str,
) -> None:
    """A plan bought before this change, whose webhook has not ended the trial
    yet: converting the trial as well would bill the customer for two plans."""
    trial_conversion.others.side_effect = None
    trial_conversion.others.return_value = _subscription_list(
        trial_conversion.live, {"id": "sub_max", "status": "active"}
    )

    response = _post_plan(client, tier)

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


@pytest.mark.parametrize(
    "tier,billing_cycle,price",
    [("MAX", "monthly", "price_max"), ("PRO", "yearly", "price_pro_yearly")],
)
def test_update_subscription_tier_cancel_pending_trial_other_plan_converts_its_subscription(
    client: fastapi.testclient.TestClient,
    trial_conversion: MagicMock,
    tier: str,
    billing_cycle: str,
    price: str,
) -> None:
    """Another tier or cycle moves the trial's own subscription onto that
    plan's price, with the plan recorded for the reconcile to grant."""
    response = _post_plan(client, tier, billing_cycle)

    assert response.status_code == 200
    assert response.json()["url"] == ""
    trial_conversion.modify.assert_awaited_once_with(
        "sub_1",
        cancel_at_period_end=False,
        trial_end="now",
        items=[{"id": "si_1", "price": price}],
        metadata={
            "trial_conversion_price_id": price,
            "trial_conversion_tier": tier,
        },
        payment_behavior="error_if_incomplete",
    )
    trial_conversion.sync.assert_awaited_once_with(dict(trial_conversion.converted))
    trial_conversion.modify_for_tier.assert_not_awaited()
    trial_conversion.checkout.assert_not_awaited()


@pytest.mark.parametrize("tier", ["PRO", "MAX"])
def test_update_subscription_tier_cancel_pending_trial_without_access_can_buy_a_plan(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
    trial_conversion: MagicMock,
    tier: str,
) -> None:
    """A cancel-pending trial whose card stopped verifying is on NO_TIER. It
    still takes the trial path; the trial subscription would otherwise make
    every plan request fail until trial_end."""
    mocker.patch(
        "backend.api.features.billing.subscriptions.routes.get_user_by_id",
        new_callable=AsyncMock,
        return_value=MagicMock(subscription_tier=SubscriptionTier.NO_TIER),
    )

    response = _post_plan(client, tier)

    assert response.status_code == 200
    trial_conversion.modify.assert_awaited_once()
    trial_conversion.modify_for_tier.assert_not_awaited()
    trial_conversion.checkout.assert_not_awaited()


def test_update_subscription_tier_cancel_pending_trial_card_it_cannot_charge_returns_409(
    client: fastapi.testclient.TestClient,
    trial_conversion: MagicMock,
) -> None:
    """The customer fixes the card in the billing portal and tries again;
    the trial is left as it was."""
    trial_conversion.card.return_value = False

    response = _post_plan(client, "MAX")

    assert response.status_code == 409
    assert response.json()["detail"] == CARD_NEEDED_DETAIL
    trial_conversion.modify.assert_not_awaited()
    trial_conversion.checkout.assert_not_awaited()


def test_update_subscription_tier_cancel_pending_trial_converts_without_a_current_price(
    client: fastapi.testclient.TestClient,
    mocker: pytest_mock.MockFixture,
    trial_conversion: MagicMock,
) -> None:
    """The conversion bills the price the trial accepted, so it needs no
    current LaunchDarkly price for the plan."""
    mocker.patch(
        "backend.api.features.billing.subscriptions.routes.get_subscription_price_id",
        new_callable=AsyncMock,
        return_value=None,
    )

    response = _post_plan(client, "PRO")

    assert response.status_code == 200
    trial_conversion.modify.assert_awaited_once()
