"""Accepted trial prices remain reconcilable after leaving the live price map."""

import os
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest
import stripe
from prisma.enums import SubscriptionTier
from prisma.models import User

from backend.data import credit, stripe_reconciliation
from backend.data import subscription_activation_integration_fixtures as fixtures
from backend.data.pro_activation import get_usage_activation_state
from backend.data.subscription_activation import reconcile_paid_activation

activation_case = fixtures.activation_case
pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires disposable DB and Redis cluster",
)


@pytest.mark.parametrize("via", ["current", "sweep"])
async def test_settled_renewal_restores_retired_accepted_plan_without_reset(
    activation_case, mocker, via
):
    case = activation_case
    await case.add_trial(100)
    assert await case.activate()
    before = await get_usage_activation_state(case.user_id)
    await case.add_cost(before.generation, 173)
    renewal = deepcopy(case.invoice)
    renewal.update(id="in_legacy_renewal", status="open", amount_remaining=2000)
    renewal["created"] += 60
    renewal["status_transitions"]["paid_at"] += 60
    renewal["lines"]["data"][0]["period"]["start"] += 60
    case.invoices.append(renewal)
    case.subscription.update(status="past_due", latest_invoice=renewal["id"])
    mocker.patch.object(credit, "build_price_to_tier_map", return_value={})
    mocker.patch.object(
        stripe_reconciliation, "build_price_to_tier_map", return_value={}
    )
    mocker.patch.object(credit, "schedule_posthog_lifecycle_sync")

    await credit.sync_subscription_from_stripe(case.subscription, track_lifecycle=False)
    assert (await get_usage_activation_state(case.user_id)).tier == "NO_TIER"
    case.subscription["status"] = "active"
    await credit.sync_subscription_from_stripe(case.subscription, track_lifecycle=False)
    assert (await get_usage_activation_state(case.user_id)).tier == "NO_TIER"

    renewal.update(status="paid", amount_remaining=0)
    if via == "current":
        result = await reconcile_paid_activation(case.user_id, case.subscription["id"])
        assert (
            result is not None
            and not result.usage_reset
            and result.activation_id is None
        )
    else:
        await run_customer_sweep(case, mocker)

    assert await get_usage_activation_state(case.user_id) == before
    assert await case.counters(before.generation) == (173, 173)


async def run_customer_sweep(case, mocker):
    async def subscriptions(*, status, **kwargs):
        return stripe.ListObject.construct_from(
            {
                "object": "list",
                "data": [case.subscription] if status == "active" else [],
                "has_more": False,
            },
            None,
        )

    async def candidates(**kwargs):
        return [await User.prisma().find_unique_or_raise(where={"id": case.user_id})]

    queries = MagicMock(wraps=User.prisma())
    queries.find_many = AsyncMock(side_effect=candidates)
    mocker.patch.object(
        stripe_reconciliation, "User", MagicMock(prisma=MagicMock(return_value=queries))
    )
    mocker.patch.object(stripe.Subscription, "list_async", side_effect=subscriptions)
    mocker.patch.object(stripe_reconciliation, "alert_tier_reconciliation_discrepancy")

    summary = await stripe_reconciliation.reconcile_all_stripe_tiers()

    assert summary.candidate_users == 1 and summary.errors == 0
    assert summary.stripe_active_subscriptions == 1


@pytest.mark.parametrize("mapped", [True, False])
async def test_converted_trial_can_recover_a_paid_pro_price_change(
    activation_case, mocker, mapped
):
    case = activation_case
    await case.add_trial(100)
    assert await case.activate()
    before = await get_usage_activation_state(case.user_id)
    await case.add_cost(before.generation, 179)
    case.subscription["items"]["data"][0]["price"]["id"] = "price_pro_yearly"
    invoice = deepcopy(case.invoice)
    invoice.update(id="in_yearly_upgrade", billing_reason="subscription_update")
    invoice["created"] += 60
    invoice["status_transitions"]["paid_at"] += 60
    invoice["lines"]["data"][0].update(proration=True, price={"id": "price_pro_yearly"})
    case.invoices.append(invoice)
    case.subscription["latest_invoice"] = invoice["id"]
    mocker.patch.object(
        credit,
        "build_price_to_tier_map",
        return_value={"price_pro_yearly": SubscriptionTier.PRO} if mapped else {},
    )
    mocker.patch.object(credit, "schedule_posthog_lifecycle_sync")

    result = await reconcile_paid_activation(case.user_id, case.subscription["id"])

    if mapped:
        assert (
            result is not None
            and not result.usage_reset
            and result.activation_id is None
        )
    else:
        assert result is None
    assert await get_usage_activation_state(case.user_id) == before
    assert await case.counters(before.generation) == (179, 179)
