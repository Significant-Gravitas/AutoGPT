from unittest.mock import AsyncMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import subscription_trial_stripe as fulfillment

pytest_plugins = ("backend.data.subscription_trial_fixtures",)


@pytest.fixture
def customer_lookup(boundaries, subscription):
    customer = {
        "id": "cus_1",
        "invoice_settings": {
            "default_payment_method": subscription["default_payment_method"],
        },
    }
    with patch.object(
        stripe.Customer, "retrieve_async", AsyncMock(return_value=customer)
    ) as lookup:
        yield lookup


@pytest.mark.asyncio
async def test_trial_accepts_customer_invoice_default(
    trial, subscription, boundaries, customer_lookup
):
    subscription["default_payment_method"] = None
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.TRIAL
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["cardVerifiedAt"] is not None
    assert saved["consumedAt"] is not None
    customer_lookup.assert_awaited_once_with(
        "cus_1", expand=["invoice_settings.default_payment_method"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method,expected",
    [
        (
            {
                "id": "pm_valid",
                "type": "card",
                "card": {"exp_month": 12, "exp_year": 2034},
            },
            SubscriptionTier.TRIAL,
        ),
        (
            {
                "id": "pm_expired",
                "type": "card",
                "card": {"exp_month": 1, "exp_year": 2020},
            },
            SubscriptionTier.NO_TIER,
        ),
        ({"id": "pm_bank", "type": "us_bank_account"}, SubscriptionTier.NO_TIER),
    ],
)
async def test_subscription_method_takes_precedence_over_customer_default(
    trial, subscription, boundaries, customer_lookup, method, expected
):
    subscription["default_payment_method"] = method
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == expected
    customer_lookup.assert_not_awaited()


@pytest.mark.asyncio
async def test_legacy_subscription_source_does_not_use_customer_fallback(
    trial, subscription, boundaries, customer_lookup
):
    subscription.update(default_payment_method=None, default_source="src_legacy")
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.NO_TIER
    customer_lookup.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method",
    [
        None,
        {
            "id": "pm_expired",
            "type": "card",
            "card": {"exp_month": 1, "exp_year": 2020},
        },
        {"id": "pm_bank", "type": "us_bank_account"},
        {"id": "pm_no_card_details", "type": "card"},
    ],
)
async def test_invalid_customer_default_does_not_grant_trial(
    trial, subscription, boundaries, customer_lookup, method
):
    subscription["default_payment_method"] = None
    customer_lookup.return_value["invoice_settings"]["default_payment_method"] = method
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.NO_TIER
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert saved["cardVerifiedAt"] is None


@pytest.mark.asyncio
async def test_customer_default_cannot_bypass_pending_setup(
    trial, subscription, boundaries, customer_lookup
):
    subscription.update(default_payment_method=None, pending_setup_intent="seti_auth")
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.NO_TIER


@pytest.mark.asyncio
async def test_customer_lookup_failure_does_not_commit_entitlements(
    trial, subscription, boundaries, customer_lookup
):
    subscription["default_payment_method"] = None
    customer_lookup.side_effect = stripe.APIConnectionError("temporary failure")
    with pytest.raises(stripe.APIConnectionError):
        await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    boundaries.subscriptiontrial.update.assert_not_awaited()
    boundaries.user.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_customer_fallback_rejects_other_customer(
    trial, subscription, boundaries, customer_lookup
):
    subscription["default_payment_method"] = None
    customer_lookup.return_value["id"] = "cus_other"
    with pytest.raises(ValueError, match="customer ownership"):
        await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    boundaries.subscriptiontrial.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_deleted_customer_cannot_grant_trial(
    trial, subscription, boundaries, customer_lookup
):
    subscription["default_payment_method"] = None
    customer_lookup.return_value = {"id": "cus_1", "deleted": True}
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier.NO_TIER
