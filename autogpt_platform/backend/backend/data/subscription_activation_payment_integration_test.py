"""Settled payment evidence is required for activation and trial reconciliation."""

import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, uuid5

import pytest
from prisma.models import PaidUsageActivation, SubscriptionTrial, User
from redis.exceptions import ResponseError

from backend.copilot.usage_activation import usage_keys
from backend.data import db
from backend.data import subscription_activation_integration_fixtures as fixtures
from backend.data.credit import sync_subscription_from_stripe
from backend.data.pro_activation import get_usage_activation_state
from backend.data.subscription_activation import reconcile_paid_activation
from backend.data.subscription_trial_stripe import reconcile_trial_subscription

activation_case = fixtures.activation_case
pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires disposable DB and Redis cluster",
)


async def test_malformed_matching_first_invoice_stays_unactivated(activation_case):
    await activation_case.add_trial(100)
    activation_case.invoice["lines"]["data"][0]["period"] = {}
    assert not await activation_case.activate()
    state = await get_usage_activation_state(activation_case.user_id)
    assert state.generation is None and state.tier == "TRIAL"


@pytest.mark.parametrize("status", ["open", "draft", "uncollectible", "void"])
@pytest.mark.parametrize("tier", ["PRO", "MAX"])
async def test_unsettled_invoice_does_not_publish_entitlement(
    activation_case, status, tier
):
    case = activation_case
    await case.add_trial(100, tier)
    await case.add_cost(None, 137)
    case.invoice["status"] = status
    case.invoice["amount_remaining"] = 2000
    assert not await case.activate()
    state = await get_usage_activation_state(case.user_id)
    assert state.generation is None and state.tier == "TRIAL"
    assert await case.counters(None) == (137, 137)


@pytest.mark.parametrize(
    "status",
    [
        "trialing",
        "incomplete",
        "incomplete_expired",
        "past_due",
        "canceled",
        "unpaid",
        "paused",
    ],
)
async def test_inactive_subscription_does_not_publish_entitlement(
    activation_case, status
):
    case = activation_case
    await case.add_trial(100)
    case.subscription["status"] = status
    assert not await case.activate()
    assert (await get_usage_activation_state(case.user_id)).generation is None


async def test_zero_dollar_trial_setup_is_not_paid_activation(activation_case):
    case = activation_case
    await case.add_trial(100)
    case.invoice["amount_paid"] = 0
    case.invoice["lines"]["data"][0]["amount"] = 0
    assert not await case.activate()
    assert (await get_usage_activation_state(case.user_id)).generation is None


async def test_settled_payment_using_credit_or_discount_qualifies(activation_case):
    case = activation_case
    await case.add_trial(100)
    case.invoice["amount_paid"] = 0
    assert await case.activate()
    assert (await get_usage_activation_state(case.user_id)).generation


@pytest.mark.parametrize("cost", [25, 100])
@pytest.mark.parametrize("early", [False, True])
@pytest.mark.parametrize("tier", ["PRO", "MAX"])
async def test_full_trial_reconciliation_commits_entitlement_and_generation(
    activation_case, cost, early, tier
):
    case = activation_case
    trial = await case.add_trial(cost, tier)
    if early:
        await SubscriptionTrial.prisma().update(
            where={"id": trial.id},
            data={"endsAt": datetime.now(UTC) + timedelta(days=4)},
        )
    await case.add_cost(None, cost)
    result = await reconcile_trial_subscription(case.user_id, case.subscription["id"])
    assert result and result[1] == tier
    state = await get_usage_activation_state(case.user_id)
    assert state.ready and state.tier == tier and state.generation
    row = await SubscriptionTrial.prisma().find_unique_or_raise(where={"id": trial.id})
    assert row.costMicrodollars == cost and row.consumedAt == trial.consumed_at
    assert row.offer == trial.offer.model_dump(mode="json")
    assert row.convertedAt and row.stripeConversionInvoiceId == case.invoice["id"]
    assert await case.counters(state.generation) == (0, 0)
    await case.add_cost(state.generation, 109)
    await reconcile_trial_subscription(case.user_id, case.subscription["id"])
    assert await get_usage_activation_state(case.user_id) == state
    assert await case.counters(state.generation) == (109, 109)


async def test_full_reconciliation_redis_error_leaves_paid_activation_recoverable(
    activation_case,
):
    case = activation_case
    trial = await case.add_trial(100)
    generation = str(
        uuid5(
            NAMESPACE_URL,
            f"autogpt:initial-pro:{case.user_id}:{case.subscription['id']}:{case.invoice['id']}",
        )
    )
    daily, weekly = usage_keys(case.user_id, generation, datetime.now(UTC))
    await case.redis.lpush(daily, "unavailable-counter")
    await case.redis.set(weekly, 113)
    with pytest.raises(ResponseError, match="WRONGTYPE"):
        await reconcile_trial_subscription(case.user_id, case.subscription["id"])
    row = await SubscriptionTrial.prisma().find_unique_or_raise(where={"id": trial.id})
    assert row.convertedAt is None and row.costMicrodollars == 100
    assert row.status == "trialing"
    state = await get_usage_activation_state(case.user_id)
    assert state.tier == "TRIAL" and state.generation is None
    await case.redis.delete(daily)
    result = await reconcile_trial_subscription(case.user_id, case.subscription["id"])
    assert result and result[1] == "PRO"
    assert await case.counters(generation) == (0, 113)


async def test_delayed_conversion_retains_first_invoice_when_latest_is_renewal(
    activation_case,
):
    case = activation_case
    trial = await case.add_trial(100)
    renewal = deepcopy(case.invoice)
    renewal.update(
        id=f"renewal_{case.invoice['id']}", created=case.invoice["created"] + 60
    )
    renewal["status_transitions"]["paid_at"] += 60
    renewal["lines"]["data"][0]["period"]["start"] += 60
    case.invoices.append(renewal)
    case.subscription["latest_invoice"] = renewal["id"]
    assert await reconcile_trial_subscription(case.user_id, case.subscription["id"])
    row = await SubscriptionTrial.prisma().find_unique_or_raise(where={"id": trial.id})
    activation = await PaidUsageActivation.prisma().find_unique_or_raise(
        where={"userId": case.user_id}
    )
    assert (
        row.stripeConversionInvoiceId
        == activation.stripeInvoiceId
        == case.invoice["id"]
    )
    assert (
        int(row.convertedAt.timestamp())
        == case.invoice["status_transitions"]["paid_at"]
    )


@pytest.mark.parametrize(
    "field,value", [("status", "open"), ("amount_remaining", 2000)]
)
async def test_full_reconciliation_rejects_unsettled_invoice(
    activation_case, field, value
):
    case = activation_case
    trial = await case.add_trial(100)
    case.invoice[field] = value
    result = await reconcile_trial_subscription(case.user_id, case.subscription["id"])
    assert result and result[1] == "NO_TIER"
    state = await get_usage_activation_state(case.user_id)
    assert state.generation is None and state.tier == "NO_TIER"
    row = await SubscriptionTrial.prisma().find_unique_or_raise(where={"id": trial.id})
    assert row.convertedAt is None and row.costMicrodollars == 100


async def test_stale_webhook_payload_cannot_override_current_stripe_state(
    activation_case,
):
    case = activation_case
    await case.add_trial(100)
    stale_event = {**case.subscription, "status": "canceled"}
    await sync_subscription_from_stripe(stale_event, track_lifecycle=False)
    state = await get_usage_activation_state(case.user_id)
    assert state.ready and state.tier == "PRO" and state.generation
    await case.add_cost(state.generation, 127)
    case.subscription["status"] = "canceled"
    await sync_subscription_from_stripe(
        {**stale_event, "status": "active"}, track_lifecycle=False
    )
    ended = await get_usage_activation_state(case.user_id)
    assert ended.tier == "NO_TIER" and ended.generation == state.generation
    assert await case.counters(state.generation) == (127, 127)


async def test_status_does_not_treat_admin_pro_label_as_completed_activation(
    activation_case,
):
    case = activation_case
    await case.add_trial(100)
    enrollment_id = case.subscription["metadata"].pop("trial_enrollment_id")
    await User.prisma().update(
        where={"id": case.user_id}, data={"subscriptionTier": "PRO"}
    )
    with pytest.raises(ValueError, match="matching enrollment"):
        await reconcile_paid_activation(case.user_id, case.subscription["id"])
    case.subscription["metadata"]["trial_enrollment_id"] = enrollment_id
    period = case.invoice["lines"]["data"][0].pop("period")
    assert not await reconcile_paid_activation(case.user_id, case.subscription["id"])
    assert (await get_usage_activation_state(case.user_id)).generation is None
    case.invoice["lines"]["data"][0]["period"] = period
    assert await reconcile_paid_activation(case.user_id, case.subscription["id"])
    state = await get_usage_activation_state(case.user_id)
    assert state.ready and state.generation


@pytest.mark.parametrize("tier", ["PRO", "MAX"])
@pytest.mark.parametrize("history", ["returning", "prepolicy"])
async def test_ineligible_reset_still_records_the_paid_trial_conversion(
    activation_case, tier, history
):
    case = activation_case
    if history == "prepolicy":
        policy = await db.prisma.paidusageactivationpolicy.find_unique_or_raise(
            where={"id": "initial-pro-v1"}
        )
        paid_at = int(policy.startsAt.timestamp()) - 1
        case.invoice["created"] = paid_at
        case.invoice["status_transitions"]["paid_at"] = paid_at
    trial = await case.add_trial(100, tier)
    if history == "returning":
        previous = deepcopy(case.invoice)
        previous.update(id="in_previous", subscription="sub_previous")
        previous["lines"]["data"][0]["subscription"] = "sub_previous"
        previous["created"] -= 60
        previous["status_transitions"]["paid_at"] -= 60
        case.invoices.append(previous)
    await case.add_cost(None, 199)

    result = await reconcile_trial_subscription(case.user_id, case.subscription["id"])

    assert result and result[1] == tier
    state = await get_usage_activation_state(case.user_id)
    assert state.ready and state.generation is None
    assert await case.counters(None) == (199, 199)
    row = await SubscriptionTrial.prisma().find_unique_or_raise(where={"id": trial.id})
    assert row.convertedAt and row.stripeConversionInvoiceId == case.invoice["id"]
    assert row.costMicrodollars == 100 and row.consumedAt == trial.consumed_at
    assert row.offer == trial.offer.model_dump(mode="json")


@pytest.mark.parametrize("tier", ["BASIC", "BUSINESS"])
async def test_other_scheduled_trial_plans_keep_usage_without_activation(
    activation_case, tier
):
    case = activation_case
    trial = await case.add_trial(100, tier)
    await case.add_cost(None, 211)

    result = await reconcile_trial_subscription(case.user_id, case.subscription["id"])

    assert result and result[1] == tier
    state = await get_usage_activation_state(case.user_id)
    assert state.ready and state.tier == tier and state.generation is None
    assert await case.counters(None) == (211, 211)
    row = await SubscriptionTrial.prisma().find_unique_or_raise(where={"id": trial.id})
    assert row.costMicrodollars == 100 and row.consumedAt == trial.consumed_at
    assert row.offer == trial.offer.model_dump(mode="json")
