"""Paid upgrade recovery must preserve the initial activation boundary."""

import os
from copy import deepcopy

import pytest
from prisma.enums import SubscriptionTier
from prisma.models import PaidUsageActivation, SubscriptionTrial, User

from backend.data import credit, db
from backend.data import subscription_activation_integration_fixtures as fixtures
from backend.data.pro_activation import get_usage_activation_state
from backend.data.subscription_activation import reconcile_paid_activation

activation_case = fixtures.activation_case
pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires disposable DB and Redis cluster",
)


@pytest.fixture(autouse=True)
def billing_boundaries(mocker):
    mocker.patch.object(
        credit,
        "build_price_to_tier_map",
        return_value={
            "price_pro": SubscriptionTier.PRO,
            "price_max": SubscriptionTier.MAX,
        },
    )
    mocker.patch.object(credit, "schedule_posthog_lifecycle_sync")


def add_proration(case):
    invoice = deepcopy(case.invoice)
    invoice.update(
        id=f"proration_{case.invoice['id']}", billing_reason="subscription_update"
    )
    invoice["created"] += 60
    invoice["status_transitions"]["paid_at"] += 60
    invoice["lines"]["data"][0].update(
        proration=True,
        amount=1000,
        price=deepcopy(case.subscription["items"]["data"][0]["price"]),
    )
    case.invoices.append(invoice)
    case.subscription["latest_invoice"] = invoice["id"]
    return invoice


async def test_paid_basic_upgrade_recovers_without_a_new_usage_generation(
    activation_case,
):
    case = activation_case
    case.invoice["lines"]["data"][0]["price"]["id"] = "price_basic"
    await User.prisma().update(
        where={"id": case.user_id}, data={"subscriptionTier": "BASIC"}
    )
    await case.add_cost(None, 149)
    add_proration(case)

    await credit.sync_subscription_from_stripe(case.subscription, track_lifecycle=False)

    state = await get_usage_activation_state(case.user_id)
    assert state.tier == "PRO" and state.generation is None
    assert await case.counters(None) == (149, 149)
    result = await reconcile_paid_activation(case.user_id, case.subscription["id"])
    assert (
        result is not None and not result.usage_reset and result.activation_id is None
    )
    assert await PaidUsageActivation.prisma().count(where={"userId": case.user_id}) == 0


async def test_paid_upgrade_preserves_existing_generation(activation_case):
    case = activation_case
    await case.add_trial(100)
    assert await case.activate()
    before = await get_usage_activation_state(case.user_id)
    await case.add_cost(before.generation, 151)
    await User.prisma().update(
        where={"id": case.user_id}, data={"subscriptionTier": "BASIC"}
    )
    add_proration(case)

    await credit.sync_subscription_from_stripe(case.subscription, track_lifecycle=False)

    assert await get_usage_activation_state(case.user_id) == before
    assert await case.counters(before.generation) == (151, 151)
    result = await reconcile_paid_activation(case.user_id, case.subscription["id"])
    assert (
        result is not None and not result.usage_reset and result.activation_id is None
    )


@pytest.mark.parametrize("history", [False, True])
async def test_proration_without_initial_paid_service_does_not_activate(
    activation_case, history
):
    case = activation_case
    if history:
        case.invoice["lines"]["data"][0]["amount"] = 0
        add_proration(case)
    else:
        case.invoice["billing_reason"] = "subscription_update"
        case.invoice["lines"]["data"][0]["proration"] = True

    await credit.sync_subscription_from_stripe(case.subscription, track_lifecycle=False)

    state = await get_usage_activation_state(case.user_id)
    assert state.tier == "NO_TIER" and state.generation is None


@pytest.mark.parametrize(
    "invalid", ["open", "foreign_price", "foreign_subscription", "quantity"]
)
async def test_paid_history_cannot_legitimize_invalid_current_proration(
    activation_case, invalid
):
    case = activation_case
    case.invoice["lines"]["data"][0]["price"]["id"] = "price_basic"
    await User.prisma().update(
        where={"id": case.user_id}, data={"subscriptionTier": "BASIC"}
    )
    invoice = add_proration(case)
    line = invoice["lines"]["data"][0]
    if invalid == "open":
        invoice.update(status="open", amount_remaining=1000)
    elif invalid == "foreign_price":
        line["price"]["id"] = "price_other"
    elif invalid == "foreign_subscription":
        line["subscription"] = "sub_other"
    else:
        line["quantity"] = 2

    await credit.sync_subscription_from_stripe(case.subscription, track_lifecycle=False)

    state = await get_usage_activation_state(case.user_id)
    assert state.tier == "BASIC" and state.generation is None


async def test_known_different_trial_subscription_does_not_block_paid_signup(
    activation_case,
):
    case = activation_case
    trial = await case.add_trial(100)
    await SubscriptionTrial.prisma().update(
        where={"id": trial.id},
        data={"stripeSubscriptionId": "sub_previous_trial", "status": "canceled"},
    )
    del case.subscription["metadata"]["trial_enrollment_id"]
    before = await SubscriptionTrial.prisma().find_unique_or_raise(
        where={"id": trial.id}
    )
    await case.add_cost(None, 153)

    await credit.sync_subscription_from_stripe(case.subscription, track_lifecycle=False)

    state = await get_usage_activation_state(case.user_id)
    assert state.tier == "PRO" and state.generation is None
    assert await case.counters(None) == (153, 153)
    assert (
        await SubscriptionTrial.prisma().find_unique_or_raise(where={"id": trial.id})
        == before
    )


@pytest.mark.parametrize("identity", ["same", "unknown"])
async def test_missing_trial_metadata_stays_closed_without_distinct_identity(
    activation_case, identity
):
    case = activation_case
    trial = await case.add_trial(100)
    if identity == "unknown":
        await SubscriptionTrial.prisma().update(
            where={"id": trial.id}, data={"stripeSubscriptionId": None}
        )
    del case.subscription["metadata"]["trial_enrollment_id"]

    with pytest.raises(ValueError, match="matching enrollment"):
        await credit.sync_subscription_from_stripe(
            case.subscription, track_lifecycle=False
        )
    assert (await get_usage_activation_state(case.user_id)).generation is None


async def test_repeated_paid_result_identifies_current_activation_without_erasing_usage(
    activation_case,
):
    case = activation_case
    await case.add_trial(100)
    result = await reconcile_paid_activation(case.user_id, case.subscription["id"])
    state = await get_usage_activation_state(case.user_id)
    assert result is not None and result.usage_reset
    assert result.activation_id == state.generation and state.generation
    await case.add_cost(state.generation, 157)

    assert (
        await reconcile_paid_activation(case.user_id, case.subscription["id"]) == result
    )
    assert await get_usage_activation_state(case.user_id) == state
    assert await case.counters(state.generation) == (157, 157)


async def test_renewal_paid_result_does_not_announce_initial_reset(activation_case):
    case = activation_case
    await case.add_trial(100)
    assert await case.activate()
    before = await get_usage_activation_state(case.user_id)
    await case.add_cost(before.generation, 159)
    renewal = deepcopy(case.invoice)
    renewal.update(id="in_renewal", billing_reason="subscription_cycle")
    renewal["created"] += 30 * 86400
    renewal["status_transitions"]["paid_at"] += 30 * 86400
    renewal["lines"]["data"][0]["period"]["start"] += 30 * 86400
    renewal["lines"]["data"][0]["period"]["end"] += 30 * 86400
    case.invoices.append(renewal)
    case.subscription["latest_invoice"] = renewal["id"]

    result = await reconcile_paid_activation(case.user_id, case.subscription["id"])

    assert (
        result is not None and not result.usage_reset and result.activation_id is None
    )
    assert await get_usage_activation_state(case.user_id) == before
    assert await case.counters(before.generation) == (159, 159)


@pytest.mark.parametrize("history", ["returning", "prepolicy"])
async def test_paid_result_without_activation_does_not_announce_a_reset(
    activation_case, history
):
    case = activation_case
    if history == "returning":
        previous = deepcopy(case.invoice)
        previous.update(id="in_previous", subscription="sub_previous")
        previous["lines"]["data"][0]["subscription"] = "sub_previous"
        previous["created"] -= 60
        previous["status_transitions"]["paid_at"] -= 60
        case.invoices.append(previous)
    else:
        policy = await db.prisma.paidusageactivationpolicy.find_unique_or_raise(
            where={"id": "initial-pro-v1"}
        )
        case.invoice["status_transitions"]["paid_at"] = (
            int(policy.startsAt.timestamp()) - 1
        )
    await case.add_cost(None, 163)

    result = await reconcile_paid_activation(case.user_id, case.subscription["id"])

    assert (
        result is not None and not result.usage_reset and result.activation_id is None
    )
    assert (await get_usage_activation_state(case.user_id)).generation is None
    assert await case.counters(None) == (163, 163)


async def test_returning_subscription_does_not_announce_old_lifetime_generation(
    activation_case,
):
    case = activation_case
    await case.add_trial(100)
    assert await case.activate()
    before = await get_usage_activation_state(case.user_id)
    await case.add_cost(before.generation, 167)
    new_invoice = deepcopy(case.invoice)
    new_invoice.update(id="in_returning", subscription="sub_returning")
    new_invoice["lines"]["data"][0]["subscription"] = "sub_returning"
    new_invoice["created"] += 60
    new_invoice["status_transitions"]["paid_at"] += 60
    case.invoices.append(new_invoice)
    case.subscription.update(id="sub_returning", latest_invoice="in_returning")
    del case.subscription["metadata"]["trial_enrollment_id"]
    await User.prisma().update(
        where={"id": case.user_id}, data={"subscriptionTier": "NO_TIER"}
    )

    result = await reconcile_paid_activation(case.user_id, case.subscription["id"])

    assert (
        result is not None and not result.usage_reset and result.activation_id is None
    )
    assert await get_usage_activation_state(case.user_id) == before
    assert await case.counters(before.generation) == (167, 167)
