"""Upstream chosen-plan conversions use the same durable usage generation."""

import os

import pytest
from prisma.models import PaidUsageActivation, SubscriptionTrial

from backend.data import subscription_activation_integration_fixtures as fixtures

activation_case = fixtures.activation_case
pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires disposable DB and Redis cluster",
)


@pytest.mark.parametrize("plan", ["PRO", "MAX"])
async def test_chosen_conversion_preserves_trial_and_paid_costs_during_replay(
    activation_case, plan
):
    case = activation_case
    trial = await case.add_trial(73, tier="BASIC")
    case.select_plan(plan)
    case.subscription["metadata"].update(
        trial_conversion_price_id=f"price_{plan.lower()}",
        trial_conversion_tier=plan,
    )

    assert await case.activate()
    activation = await PaidUsageActivation.prisma().find_unique_or_raise(
        where={"userId": case.user_id}
    )
    assert activation.readyAt is not None
    assert await case.counters(activation.id) == (0, 0)
    await case.add_cost(activation.id, 29)
    assert await case.activate()
    assert await case.counters(activation.id) == (29, 29)
    assert await PaidUsageActivation.prisma().count(where={"userId": case.user_id}) == 1
    saved = await SubscriptionTrial.prisma().find_unique_or_raise(
        where={"id": trial.id}
    )
    assert saved.convertedAt and saved.stripeConversionInvoiceId == case.invoice["id"]
    assert saved.consumedAt == trial.consumed_at
    assert saved.costMicrodollars == 73
    assert saved.offer == trial.offer.model_dump(mode="json")
