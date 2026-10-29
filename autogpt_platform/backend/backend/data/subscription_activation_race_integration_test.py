"""A later paid plan cannot complete an earlier plan's activation response."""

import os
from datetime import UTC, datetime, timedelta

import pytest
from prisma.enums import SubscriptionTier
from prisma.models import SubscriptionTrial, User

from backend.data import credit
from backend.data import subscription_activation as activation
from backend.data import subscription_activation_checkout as checkout
from backend.data import subscription_activation_integration_fixtures as fixtures
from backend.data.pro_activation import get_usage_activation_state
from backend.data.subscription_activation_attempt import save_confirmation, save_quote
from backend.data.subscription_activation_models import ActivationTerms
from backend.data.subscription_activation_transition_integration_test import (
    prepare_paid_user,
)
from backend.data.subscription_trial import TrialState

activation_case = fixtures.activation_case
pytestmark = pytest.mark.skipif(
    os.environ.get("TRIAL_TEST_DATABASE") != "1",
    reason="Requires disposable DB and Redis cluster",
)


@pytest.mark.parametrize("source,target", [("PRO", "MAX"), ("MAX", "PRO")])
async def test_attempt_price_change_during_reconciliation_stays_pending(
    activation_case, mocker, source, target
):
    case = activation_case
    trial, generation = await prepare_case(case, mocker, source)
    attempt = await save_confirmation(
        await save_quote(
            case.user_id,
            case.subscription["id"],
            case.subscription["customer"],
            ActivationTerms(
                plan=source,
                price_id=f"price_{source.lower()}",
                accepted_offer_token=trial.offer.token,
                amount_due=2000,
                currency="usd",
                billing_interval="month",
                renewal_unit_amount=2000,
                renewal_terms="Monthly",
                expires_at=datetime.now(UTC) + timedelta(minutes=10),
            ),
            "/chat/kept",
        )
    )
    reconcile = checkout.reconcile_paid_activation

    async def changed_plan(user_id, subscription_id, **kwargs):
        case.subscription["items"]["data"][0]["price"]["id"] = f"price_{target.lower()}"
        return await reconcile(user_id, subscription_id, **kwargs)

    mocker.patch.object(checkout, "reconcile_paid_activation", side_effect=changed_plan)
    result = await checkout.get_activation(case.user_id, attempt.id)
    assert result.status == "processing" and not result.usage_reset
    assert result.activation_id is None
    assert result.terms and result.terms.plan == source
    result = await checkout.get_activation(case.user_id, attempt.id)
    assert result.status == "not_applicable" and result.error_code == "plan_changed"
    assert await case.counters(generation) == (191, 191)


@pytest.mark.parametrize("source,target", [("PRO", "MAX"), ("MAX", "PRO")])
async def test_price_change_after_sync_requires_matching_locked_entitlement(
    activation_case, mocker, source, target
):
    case = activation_case
    _, generation = await prepare_case(case, mocker, source)
    sync = credit.sync_subscription_from_stripe

    async def changed_plan(subscription):
        await sync(subscription)
        case.subscription["items"]["data"][0]["price"]["id"] = f"price_{target.lower()}"

    mocker.patch.object(
        credit, "sync_subscription_from_stripe", side_effect=changed_plan
    )
    assert (
        await activation.reconcile_paid_activation(
            case.user_id, case.subscription["id"]
        )
        is None
    )
    user = await User.prisma().find_unique_or_raise(where={"id": case.user_id})
    assert user.subscriptionTier == source
    result = await activation.reconcile_paid_activation(
        case.user_id, case.subscription["id"]
    )
    assert result is not None and not result.usage_reset
    user = await User.prisma().find_unique_or_raise(where={"id": case.user_id})
    assert user.subscriptionTier == target
    assert await case.counters(generation) == (191, 191)


async def prepare_case(case, mocker, source):
    mocker.patch.object(
        credit,
        "build_price_to_tier_map",
        return_value={
            "price_pro": SubscriptionTier.PRO,
            "price_max": SubscriptionTier.MAX,
        },
    )
    mocker.patch.object(credit, "schedule_posthog_lifecycle_sync")
    await prepare_paid_user(case, source, "activation")
    case.subscription["items"]["data"][0]["price"].update(
        unit_amount=2000,
        currency="usd",
        recurring={"interval": "month", "interval_count": 1},
    )
    row = await SubscriptionTrial.prisma().find_unique_or_raise(
        where={"userId": case.user_id}
    )
    state = await get_usage_activation_state(case.user_id)
    assert state.generation
    await case.add_cost(state.generation, 191)
    return TrialState.from_db(row), state.generation
