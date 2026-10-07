"""Full recovery reads publish one reset and preserve subsequent paid usage."""

import os

import pytest
import stripe
from prisma.enums import SubscriptionTier
from prisma.models import SubscriptionTrial

from backend.data import credit
from backend.data import subscription_activation_integration_fixtures as fixtures
from backend.data.subscription_activation_checkout import current_activation

activation_case = fixtures.activation_case
pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires explicitly selected disposable database and Redis",
)


@pytest.mark.parametrize("trial_cost", [None, 25, 100])
async def test_current_completes_activation_and_keeps_paid_usage_on_repeated_reads(
    activation_case, mocker, trial_cost
):
    case = activation_case
    trial = await case.add_trial(trial_cost) if trial_cost is not None else None
    await case.add_cost(None, 173)
    case.subscription["metadata"]["pro_activation_return_to"] = "/chat/resume?kept=1"
    case.subscription["items"]["data"][0]["price"].update(
        unit_amount=2000,
        currency="usd",
        recurring={"interval": "month", "interval_count": 1},
    )
    mocker.patch.object(
        credit,
        "build_price_to_tier_map",
        return_value={"price_pro": SubscriptionTier.PRO},
    )
    mocker.patch.object(credit, "schedule_posthog_lifecycle_sync")
    mocker.patch(
        "stripe.Subscription.list_async",
        return_value=stripe.ListObject.construct_from(
            {"object": "list", "data": [case.subscription], "has_more": False}, None
        ),
    )
    charge = mocker.patch("stripe.Subscription.modify_async")
    response = await current_activation(case.user_id)
    assert response.status == "ready" and response.usage_reset
    assert response.activation_id
    assert response.return_to == "/chat/resume?kept=1"
    assert await case.counters(response.activation_id) == (0, 0)
    await case.add_cost(response.activation_id, 179)

    for _ in range(2):
        assert await current_activation(case.user_id) == response
    assert await case.counters(response.activation_id) == (179, 179)
    assert await case.counters(None) == (173, 173)
    charge.assert_not_called()
    if trial:
        saved = await SubscriptionTrial.prisma().find_unique_or_raise(
            where={"id": trial.id}
        )
        assert saved.convertedAt and saved.costMicrodollars == trial_cost
        assert saved.consumedAt == trial.consumed_at
