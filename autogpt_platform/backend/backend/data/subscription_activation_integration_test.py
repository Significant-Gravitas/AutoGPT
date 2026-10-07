"""Activation atomicity and replay safety on PostgreSQL and Redis Cluster."""

import asyncio
import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, uuid5

import pytest
from prisma.enums import SubscriptionTier
from prisma.errors import RawQueryError
from prisma.models import PaidUsageActivation, SubscriptionTrial, User
from redis.exceptions import ResponseError

from backend.copilot.usage_activation import usage_keys
from backend.data import db
from backend.data import subscription_activation_integration_fixtures as fixtures
from backend.data.credit import set_subscription_tier
from backend.data.pro_activation import get_usage_activation_state
from backend.data.subscription_trial import record_subscription_trial_cost

activation_case = fixtures.activation_case
pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires disposable DB and Redis cluster",
)


@pytest.mark.parametrize("tier", ["PRO", "MAX"])
async def test_initial_signup_preserves_calendar_usage(activation_case, tier):
    case = activation_case
    case.select_plan(tier)
    await case.add_cost(None, 900)
    assert await case.activate()
    state = await get_usage_activation_state(case.user_id)
    assert state.ready and state.tier == tier and state.generation is None
    assert await case.counters(None) == (900, 900)
    assert await PaidUsageActivation.prisma().count(where={"userId": case.user_id}) == 0


@pytest.mark.parametrize("cost", [25, 100, 250])
@pytest.mark.parametrize("tier", ["PRO", "MAX"])
async def test_trial_conversion_preserves_lifetime_ledger_and_account_history(
    activation_case, cost, tier
):
    case = activation_case
    trial = await case.add_trial(cost, tier)
    before = await SubscriptionTrial.prisma().find_unique_or_raise(
        where={"userId": case.user_id}
    )
    claim = await db.prisma.subscriptiontrialclaim.create(
        data={"key": f"claim:{trial.id}", "trialId": trial.id}
    )
    balance = await db.prisma.userbalance.create(
        data={"userId": case.user_id, "balance": 321}
    )
    history = await db.prisma.credittransaction.create(
        data={"userId": case.user_id, "amount": 123, "type": "GRANT"}
    )
    chat = await db.prisma.chatsession.create(
        data={
            "userId": case.user_id,
            "totalPromptTokens": 1234,
            "title": "Preserved conversation",
        }
    )
    await case.add_cost(None, cost)
    assert await case.activate()
    after = await SubscriptionTrial.prisma().find_unique_or_raise(
        where={"userId": case.user_id}
    )
    assert after.convertedAt and after.stripeConversionInvoiceId == case.invoice["id"]
    assert after.costMicrodollars == cost
    for name in ("id", "offer", "consumedAt", "startedAt", "endsAt", "cardVerifiedAt"):
        assert before.model_dump()[name] == after.model_dump()[name]
    assert (
        await db.prisma.subscriptiontrialclaim.find_unique(where={"key": claim.key})
        == claim
    )
    assert (
        await db.prisma.userbalance.find_unique(where={"userId": case.user_id})
        == balance
    )
    assert await db.prisma.credittransaction.find_many(
        where={"userId": case.user_id}
    ) == [history]
    assert await db.prisma.chatsession.find_unique(where={"id": chat.id}) == chat
    state = await get_usage_activation_state(case.user_id)
    assert state.ready and state.tier == tier and state.generation
    assert state.trial_id is None
    assert await case.counters(state.generation) == (0, 0)


@pytest.mark.parametrize("tier", ["PRO", "MAX"])
async def test_concurrent_duplicate_activation_publishes_one_generation(
    activation_case, tier
):
    case = activation_case
    await case.add_trial(100, tier)
    assert all(await asyncio.gather(*(case.activate() for _ in range(4))))
    state = await get_usage_activation_state(case.user_id)
    assert state.ready and state.generation
    assert await PaidUsageActivation.prisma().count(where={"userId": case.user_id}) == 1
    await case.add_cost(state.generation, 37)
    assert all(await asyncio.gather(*(case.activate() for _ in range(4))))
    assert await get_usage_activation_state(case.user_id) == state
    assert await case.counters(state.generation) == (37, 37)


async def test_legacy_conversion_marker_preserves_calendar_usage(activation_case):
    case = activation_case
    await case.add_trial(100)
    await SubscriptionTrial.prisma().update(
        where={"userId": case.user_id},
        data={"convertedAt": datetime.now(UTC)},
    )
    await case.add_cost(None, 100)
    assert await case.activate()
    state = await get_usage_activation_state(case.user_id)
    assert state.ready and state.generation is None
    assert await case.counters(None) == (100, 100)


async def test_database_failure_rolls_back_entitlement_and_retry_keeps_paid_cost(
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
    await case.add_cost(generation, 29)
    with pytest.raises(RawQueryError):
        await case.activate(abort=True)
    assert await PaidUsageActivation.prisma().count(where={"userId": case.user_id}) == 0
    pending_trial = await SubscriptionTrial.prisma().find_unique_or_raise(
        where={"id": trial.id}
    )
    assert pending_trial.convertedAt is None and pending_trial.costMicrodollars == 100
    assert (await get_usage_activation_state(case.user_id)).tier == "TRIAL"
    assert await case.activate()
    assert (await get_usage_activation_state(case.user_id)).generation == generation
    assert await case.counters(generation) == (29, 29)


async def test_redis_failure_rolls_back_database_and_recovers_without_erasing_usage(
    activation_case,
):
    case = activation_case
    await case.add_trial(100)
    generation = str(
        uuid5(
            NAMESPACE_URL,
            f"autogpt:initial-pro:{case.user_id}:{case.subscription['id']}:{case.invoice['id']}",
        )
    )
    daily, weekly = usage_keys(case.user_id, generation, datetime.now(UTC))
    await case.redis.lpush(daily, "invalid-counter")
    await case.redis.set(weekly, 41)
    with pytest.raises(ResponseError, match="WRONGTYPE"):
        await case.activate()
    assert await PaidUsageActivation.prisma().count(where={"userId": case.user_id}) == 0
    assert (await get_usage_activation_state(case.user_id)).tier == "TRIAL"
    await case.redis.delete(daily)
    await case.redis.set(daily, 17)
    assert await case.activate()
    assert await case.counters(generation) == (17, 41)


async def test_incomplete_activation_is_processing_then_finishes_without_reset(
    activation_case,
):
    case = activation_case
    await case.add_trial(100)
    row = await PaidUsageActivation.prisma().create(
        data={
            "userId": case.user_id,
            "stripeSubscriptionId": case.subscription["id"],
            "stripeInvoiceId": case.invoice["id"],
        }
    )
    await case.add_cost(row.id, 53)
    state = await get_usage_activation_state(case.user_id)
    assert state.generation == row.id and not state.ready
    assert await case.activate()
    ready = await get_usage_activation_state(case.user_id)
    assert ready.generation == row.id and ready.ready and ready.tier == "PRO"
    assert await case.counters(row.id) == (53, 53)


@pytest.mark.parametrize("tier", ["PRO", "MAX", "ENTERPRISE"])
async def test_existing_paid_history_never_resets(activation_case, tier):
    case = activation_case
    await User.prisma().update(
        where={"id": case.user_id}, data={"subscriptionTier": tier}
    )
    await case.add_cost(None, 73)
    policy = await db.prisma.paidusageactivationpolicy.find_unique_or_raise(
        where={"id": "initial-pro-v1"}
    )
    case.invoice["status_transitions"]["paid_at"] = int(policy.startsAt.timestamp()) - 1
    case.invoice["created"] = int(policy.startsAt.timestamp()) - 2
    assert await case.activate()
    assert (await get_usage_activation_state(case.user_id)).generation is None
    assert await case.counters(None) == (73, 73)


async def test_admin_tier_edits_and_nontrial_payment_preserve_usage(
    activation_case,
):
    case = activation_case
    await case.add_cost(None, 71)
    for tier in (SubscriptionTier.PRO, SubscriptionTier.MAX, SubscriptionTier.PRO):
        await set_subscription_tier(case.user_id, tier, track_lifecycle=False)
        assert (await get_usage_activation_state(case.user_id)).generation is None
        assert await case.counters(None) == (71, 71)
    assert await case.activate()
    state = await get_usage_activation_state(case.user_id)
    assert state.generation is None and state.ready
    assert await case.counters(None) == (71, 71)


async def test_returning_subscriber_never_receives_another_generation(activation_case):
    case = activation_case
    await case.add_trial(100)
    previous = deepcopy(case.invoice)
    previous.update(id=f"old_{case.invoice['id']}", subscription="sub_previous")
    previous["lines"]["data"][0]["subscription"] = "sub_previous"
    previous["created"] -= 60
    previous["status_transitions"]["paid_at"] -= 60
    case.invoices.append(previous)
    await case.add_cost(None, 79)
    assert await case.activate()
    assert (await get_usage_activation_state(case.user_id)).generation is None
    assert await case.counters(None) == (79, 79)


async def test_payment_before_cutover_never_resets_on_delayed_event(activation_case):
    case = activation_case
    policy = await db.prisma.paidusageactivationpolicy.find_unique_or_raise(
        where={"id": "initial-pro-v1"}
    )
    case.invoice["status_transitions"]["paid_at"] = int(
        (policy.startsAt - timedelta(seconds=1)).timestamp()
    )
    case.invoice["created"] = case.invoice["status_transitions"]["paid_at"] - 1
    await case.add_trial(100)
    await case.add_cost(None, 83)
    assert await case.activate()
    assert (await get_usage_activation_state(case.user_id)).generation is None
    assert await case.counters(None) == (83, 83)


async def test_renewal_replay_preserves_current_generation_and_paid_usage(
    activation_case,
):
    case = activation_case
    await case.add_trial(100)
    assert await case.activate()
    state = await get_usage_activation_state(case.user_id)
    await case.add_cost(state.generation, 89)
    renewal = deepcopy(case.invoice)
    renewal.update(
        id=f"renewal_{case.invoice['id']}", billing_reason="subscription_cycle"
    )
    renewal["created"] += 30 * 86400
    renewal["status_transitions"]["paid_at"] += 30 * 86400
    case.invoices.append(renewal)
    case.subscription["latest_invoice"] = renewal["id"]
    assert await case.activate()
    case.subscription["latest_invoice"] = case.invoice["id"]
    assert await case.activate()
    assert await get_usage_activation_state(case.user_id) == state
    assert await case.counters(state.generation) == (89, 89)


@pytest.mark.parametrize("field", ["customer", "metadata"])
async def test_subscription_ownership_is_enforced_before_any_reset(
    activation_case, field
):
    case = activation_case
    case.subscription[field] = (
        "cus_other" if field == "customer" else {"user_id": "someone_else"}
    )
    with pytest.raises(ValueError, match="does not belong"):
        await case.activate()
    assert await PaidUsageActivation.prisma().count(where={"userId": case.user_id}) == 0


async def test_late_trial_ledger_cost_does_not_consume_paid_generation(activation_case):
    case = activation_case
    trial = await case.add_trial(100)
    assert await case.activate()
    state = await get_usage_activation_state(case.user_id)
    await case.add_cost(state.generation, 97)
    await asyncio.gather(
        *(record_subscription_trial_cost(case.user_id, 13, trial.id) for _ in range(8))
    )
    row = await SubscriptionTrial.prisma().find_unique_or_raise(where={"id": trial.id})
    assert row.costMicrodollars == 204 and row.consumedAt == trial.consumed_at
    assert await case.counters(state.generation) == (97, 97)
