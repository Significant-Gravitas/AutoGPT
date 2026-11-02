"""Existing cancel-pending conversions retain their owned, settled target."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import SubscriptionTier

from backend.data import subscription_activation_stripe as billing
from backend.data import subscription_activation_target as target
from backend.data import subscription_trial_fixtures as fixtures

trial = fixtures.trial
subscription = fixtures.subscription


@pytest.fixture
def chosen(trial, subscription, monkeypatch):
    trial.subscription_id = subscription["id"]
    subscription["status"] = "active"
    subscription["items"]["data"][0]["price"]["id"] = "price_selected"
    subscription["metadata"].update(
        trial_conversion_price_id="price_selected", trial_conversion_tier="MAX"
    )
    monkeypatch.setattr(target, "query_raw_with_schema", AsyncMock(return_value=[]))
    return subscription


@pytest.mark.asyncio
@pytest.mark.parametrize("plan", ["PRO", "MAX"])
async def test_chosen_paid_target_preserves_retired_selected_price(trial, chosen, plan):
    chosen["metadata"]["trial_conversion_tier"] = plan
    assert await target.accepted_conversion_target(trial, chosen, MagicMock()) == (
        SubscriptionTier(plan),
        "price_selected",
    )
    target.query_raw_with_schema.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("user_id", "foreign"),
        ("trial_enrollment_id", "foreign"),
        ("trial_checkout_attempt", "1"),
        ("trial_conversion_price_id", "price_other"),
        ("trial_conversion_tier", "BASIC"),
        ("trial_conversion_tier", "BUSINESS"),
        ("trial_conversion_tier", "ENTERPRISE"),
        ("trial_conversion_tier", "max"),
        ("trial_conversion_tier", ""),
        ("pro_activation_attempt_id", "unknown-confirmation"),
    ],
)
async def test_chosen_metadata_cannot_bypass_ownership_or_confirmation(
    trial, chosen, field, value
):
    chosen["metadata"][field] = value
    assert await target.accepted_conversion_target(trial, chosen, MagicMock()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["trialing", "incomplete", "past_due", "canceled"])
async def test_pending_chosen_target_does_not_authorize_activation(
    trial, chosen, status
):
    chosen["status"] = status
    assert await target.accepted_conversion_target(trial, chosen, MagicMock()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("quantity", [0, 2, None])
async def test_chosen_target_requires_the_owned_single_item(trial, chosen, quantity):
    chosen["items"]["data"][0]["quantity"] = quantity
    assert await target.accepted_conversion_target(trial, chosen, MagicMock()) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("plan", ["PRO", "MAX"])
async def test_polling_recognizes_owned_chosen_price_after_catalog_repricing(
    trial, billable_chosen, plan
):
    billable_chosen.metadata["trial_conversion_tier"] = plan
    assert (
        await billing.is_paid_activation_subscription(trial.user_id, billable_chosen)
        is True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value",
    [
        ("user_id", "foreign"),
        ("trial_enrollment_id", "foreign"),
        ("trial_checkout_attempt", "1"),
        ("trial_conversion_price_id", "price_other"),
        ("trial_conversion_tier", "BASIC"),
        ("pro_activation_attempt_id", "unknown-confirmation"),
    ],
)
async def test_polling_cannot_recognize_foreign_or_unconfirmed_chosen_plan(
    trial, billable_chosen, field, value
):
    billable_chosen.metadata[field] = value
    assert (
        await billing.is_paid_activation_subscription(trial.user_id, billable_chosen)
        is None
    )


@pytest.fixture
def billable_chosen(trial, chosen, monkeypatch):
    chosen["items"]["data"][0]["price"].update(
        unit_amount=10000,
        currency="usd",
        recurring={"interval": "month", "interval_count": 1},
    )
    subscription = billing.BillingSubscription.model_validate(chosen)
    monkeypatch.setattr(
        billing, "get_subscription_trial", AsyncMock(return_value=trial)
    )
    monkeypatch.setattr(
        billing.credit, "build_price_to_tier_map", AsyncMock(return_value={})
    )
    return subscription
