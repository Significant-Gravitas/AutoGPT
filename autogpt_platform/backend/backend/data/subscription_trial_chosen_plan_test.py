"""A trial ended early on a plan the customer chose in place of the accepted
one: the reconcile grants that plan once it is paid for, and nothing else."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from prisma.enums import SubscriptionTier

from backend.data import subscription_trial_stripe as fulfillment

pytest_plugins = ("backend.data.subscription_trial_fixtures",)

CHOSEN = {
    "trial_conversion_price_id": "price_max",
    "trial_conversion_tier": "MAX",
}


@pytest.fixture
def switched(subscription: dict) -> dict:
    """The trial's subscription after Max was paid for in its place."""
    end = int(datetime.now(UTC).timestamp()) - 60
    subscription.update(
        status="active",
        trial_end=end,
        metadata={**subscription["metadata"], **CHOSEN},
        items={
            "data": [{"id": "si_1", "price": {"id": "price_max"}, "quantity": 1}],
            "has_more": False,
        },
        latest_invoice={
            "id": "in_max",
            "status": "paid",
            "created": end,
            "billing_reason": "subscription_update",
        },
    )
    return subscription


@pytest.mark.asyncio
@pytest.mark.parametrize("activation_ready", [False, True])
async def test_a_paid_switch_to_the_chosen_plan_converts_the_trial_onto_it(
    trial, switched, boundaries, monkeypatch, activation_ready
):
    publish = AsyncMock(return_value=activation_ready)
    monkeypatch.setattr(
        fulfillment.subscription_activation, "publish_initial_pro_activation", publish
    )
    boundaries.user.find_unique_or_raise = AsyncMock()
    boundaries.subscriptiontrial.find_unique_or_raise = AsyncMock(
        return_value=SimpleNamespace(
            convertedAt=datetime.now(UTC) if activation_ready else None,
            stripeConversionInvoiceId="in_max" if activation_ready else None,
        )
    )
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)

    expected = SubscriptionTier.MAX if activation_ready else SubscriptionTier.NO_TIER
    assert result is not None and result[1] == expected
    publish.assert_awaited_once()
    assert publish.await_args.args[2] == "price_max"
    saved = boundaries.subscriptiontrial.update.await_args.kwargs["data"]
    assert (saved["convertedAt"] is not None) == activation_ready
    assert saved["stripeConversionInvoiceId"] == (
        "in_max" if activation_ready else None
    )
    granted = boundaries.user.update_many.await_args.kwargs["data"]
    assert granted["subscriptionTier"] == expected
    # A plan clears the product-mail win-back date; ending the trial stamps it.
    assert (granted["subscriptionEndedAt"] is None) == activation_ready


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", ["BASIC", "BUSINESS"])
async def test_other_chosen_plans_keep_existing_conversion_without_a_usage_reset(
    trial, switched, boundaries, monkeypatch, tier
):
    switched["metadata"]["trial_conversion_tier"] = tier
    publish = AsyncMock()
    monkeypatch.setattr(
        fulfillment.subscription_activation, "publish_initial_pro_activation", publish
    )
    result = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
    assert result is not None and result[1] == SubscriptionTier(tier)
    publish.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        # Still trialing: the switch has not been paid for.
        {"status": "trialing"},
        # Items that no longer carry the chosen price.
        {
            "items": {
                "data": [{"id": "si_1", "price": {"id": "price_other"}, "quantity": 1}],
                "has_more": False,
            }
        },
        {
            "items": {
                "data": [{"id": "si_1", "price": {"id": "price_max"}, "quantity": 2}],
                "has_more": False,
            }
        },
    ],
)
async def test_the_chosen_price_is_granted_only_as_paid_for(
    trial, switched, boundaries, change
):
    switched.update(change)

    with pytest.raises(ValueError, match="accepted"):
        await fulfillment._reconcile_locked(trial, "sub_1", boundaries)

    boundaries.user.update_many.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("tier", ["TRIAL", "NO_TIER", "ENTERPRISE", "max", None])
async def test_a_chosen_plan_names_a_self_serve_tier(trial, switched, boundaries, tier):
    metadata = {**switched["metadata"], "trial_conversion_tier": tier}
    switched["metadata"] = {key: value for key, value in metadata.items() if value}

    with pytest.raises(ValueError, match="accepted"):
        await fulfillment._reconcile_locked(trial, "sub_1", boundaries)

    boundaries.user.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_switch_that_lost_its_record_of_the_choice_is_refused(
    trial, switched, boundaries
):
    switched["metadata"] = {
        key: value
        for key, value in switched["metadata"].items()
        if key != "trial_conversion_price_id"
    }

    with pytest.raises(ValueError, match="accepted"):
        await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
