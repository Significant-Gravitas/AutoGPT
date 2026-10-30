"""Paid plan changes preserve usage, including pre-activation trial history."""

import os
from copy import deepcopy
from datetime import UTC, datetime

import pytest
from prisma.enums import SubscriptionTier
from prisma.models import PaidUsageActivation, SubscriptionTrial, User

from backend.data import credit
from backend.data import subscription_activation_integration_fixtures as fixtures
from backend.data.pro_activation import get_usage_activation_state
from backend.data.subscription_activation import reconcile_paid_activation

activation_case = fixtures.activation_case
pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires disposable DB and Redis cluster",
)


@pytest.mark.parametrize("source,target", [("PRO", "MAX"), ("MAX", "PRO")])
@pytest.mark.parametrize("history", ["no_trial", "converted_legacy", "activation"])
@pytest.mark.parametrize("new_invoice", [False, True])
async def test_paid_plan_change_never_resets_usage(
    activation_case, mocker, source, target, history, new_invoice
):
    case = activation_case
    mocker.patch.object(
        credit,
        "build_price_to_tier_map",
        return_value={
            "price_pro": SubscriptionTier.PRO,
            "price_max": SubscriptionTier.MAX,
        },
    )
    mocker.patch.object(credit, "schedule_posthog_lifecycle_sync")
    await prepare_paid_user(case, source, history)
    before = await get_usage_activation_state(case.user_id)
    await case.add_cost(before.generation, 191)
    trial_before = await SubscriptionTrial.prisma().find_unique(
        where={"userId": case.user_id}
    )
    case.subscription["items"]["data"][0]["price"]["id"] = f"price_{target.lower()}"
    if new_invoice:
        add_plan_change_invoice(case)

    for _ in range(2):
        result = await reconcile_paid_activation(case.user_id, case.subscription["id"])
        assert result is not None and not result.usage_reset
        assert result.activation_id is None
        state = await get_usage_activation_state(case.user_id)
        assert state.ready and state.tier == target
        assert state.generation == before.generation
        assert await case.counters(state.generation) == (191, 191)
    assert await PaidUsageActivation.prisma().count(where={"userId": case.user_id}) == (
        1 if history == "activation" else 0
    )
    trial_after = await SubscriptionTrial.prisma().find_unique(
        where={"userId": case.user_id}
    )
    if trial_before:
        assert trial_after
        for field in (
            "id",
            "userId",
            "offer",
            "consumedAt",
            "costMicrodollars",
            "convertedAt",
            "stripeConversionInvoiceId",
            "stripeSubscriptionId",
        ):
            assert trial_after.model_dump()[field] == trial_before.model_dump()[field]
    else:
        assert trial_after is None


@pytest.mark.parametrize("tier", ["PRO", "MAX"])
async def test_legacy_conversion_invoice_replay_never_resets_usage(
    activation_case, mocker, tier
):
    case = activation_case
    mocker.patch.object(
        credit,
        "build_price_to_tier_map",
        return_value={
            "price_pro": SubscriptionTier.PRO,
            "price_max": SubscriptionTier.MAX,
        },
    )
    mocker.patch.object(credit, "schedule_posthog_lifecycle_sync")
    await prepare_paid_user(case, tier, "converted_legacy")
    await case.add_cost(None, 193)
    for _ in range(2):
        result = await reconcile_paid_activation(case.user_id, case.subscription["id"])
        assert result is not None and not result.usage_reset
        assert result.activation_id is None
    state = await get_usage_activation_state(case.user_id)
    assert state.ready and state.tier == tier and state.generation is None
    assert await case.counters(None) == (193, 193)
    assert await PaidUsageActivation.prisma().count(where={"userId": case.user_id}) == 0


async def prepare_paid_user(case, tier, history):
    case.select_plan(tier)
    if history != "no_trial":
        trial = await case.add_trial(100, tier)
        if history == "activation":
            assert await case.activate()
            return
        await SubscriptionTrial.prisma().update(
            where={"id": trial.id},
            data={
                "convertedAt": datetime.now(UTC),
                "stripeConversionInvoiceId": case.invoice["id"],
                "status": "active",
            },
        )
    await User.prisma().update(
        where={"id": case.user_id}, data={"subscriptionTier": tier}
    )


def add_plan_change_invoice(case):
    invoice = deepcopy(case.invoice)
    invoice.update(
        id=f"change_{case.invoice['id']}", billing_reason="subscription_update"
    )
    invoice["created"] += 60
    invoice["status_transitions"]["paid_at"] += 60
    invoice["lines"]["data"][0].update(
        proration=True,
        price=deepcopy(case.subscription["items"]["data"][0]["price"]),
    )
    case.invoices.append(invoice)
    case.subscription["latest_invoice"] = invoice["id"]
