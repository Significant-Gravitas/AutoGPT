from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import subscription_trial_conversion as conversion
from backend.data.subscription_trial import TrialState

pytest_plugins = ("backend.data.subscription_trial_fixtures",)


@pytest.fixture
def pending_trial(trial: TrialState) -> TrialState:
    now = datetime.now(UTC)
    return trial.model_copy(
        update={
            "subscription_id": "sub_1",
            "status": "trialing",
            "card_verified_at": now - timedelta(days=2),
            "started_at": now - timedelta(days=2),
            "ends_at": now + timedelta(days=5),
            "consumed_at": now - timedelta(days=2),
            "cancel_at_period_end": True,
        }
    )


@pytest.fixture
def live(subscription: dict) -> dict:
    return {**subscription, "cancel_at_period_end": True}


@pytest.fixture
def boundaries(live: dict):
    calls: list[str] = []
    converted = stripe.Subscription.construct_from(
        {**live, "status": "active", "cancel_at_period_end": False}, "test-key"
    )

    def recorded(name: str, result: object = None) -> AsyncMock:
        async def effect(*args, **kwargs):
            calls.append(name)
            return result

        return AsyncMock(side_effect=effect)

    @asynccontextmanager
    async def lock(user_id: str):
        calls.append(f"lock:{user_id}")
        try:
            yield
        finally:
            calls.append("unlock")

    retrieve = recorded(
        "retrieve", stripe.Subscription.construct_from(live, "test-key")
    )
    with (
        patch.object(conversion, "subscription_checkout_lock", lock),
        patch.object(stripe.Subscription, "retrieve_async", retrieve),
        patch.object(
            stripe.Subscription, "modify_async", recorded("modify", converted)
        ) as modify,
        patch.object(
            conversion, "expire_other_subscription_checkouts", recorded("expire")
        ) as expire,
        patch.object(
            conversion, "sync_subscription_from_stripe", recorded("sync")
        ) as sync,
    ):
        yield MagicMock(
            calls=calls,
            retrieve=retrieve,
            modify=modify,
            expire=expire,
            sync=sync,
            converted=converted,
        )


@pytest.mark.asyncio
async def test_converts_cancel_pending_trial_in_place_under_checkout_lock(
    pending_trial, boundaries
):
    await conversion.convert_cancel_pending_trial(pending_trial)

    assert boundaries.calls == [
        "lock:user-1",
        "retrieve",
        "expire",
        "modify",
        "sync",
        "unlock",
    ]
    boundaries.retrieve.assert_awaited_once_with("sub_1")
    boundaries.expire.assert_awaited_once_with("cus_1")
    boundaries.modify.assert_awaited_once_with(
        "sub_1",
        cancel_at_period_end=False,
        trial_end="now",
        proration_behavior="none",
        payment_behavior="error_if_incomplete",
    )
    boundaries.sync.assert_awaited_once_with(dict(boundaries.converted))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"customer": "cus_other"},
        {"metadata": {"trial_enrollment_id": "trial-x", "user_id": "user-1"}},
        {"metadata": {"trial_enrollment_id": "trial-1", "user_id": "user-x"}},
        {"metadata": None},
    ],
)
async def test_refuses_a_subscription_the_trial_does_not_own(
    pending_trial, live, boundaries, change
):
    live.update(change)
    boundaries.retrieve.side_effect = None
    boundaries.retrieve.return_value = stripe.Subscription.construct_from(
        live, "test-key"
    )

    with pytest.raises(conversion.TrialConversionRefused) as refused:
        await conversion.convert_cancel_pending_trial(pending_trial)

    assert str(refused.value) == "This trial has ended. Manage the plan in billing."
    boundaries.expire.assert_not_awaited()
    boundaries.modify.assert_not_awaited()
    boundaries.sync.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"status": "active"},
        {"status": "past_due"},
        {"trial_end": int((datetime.now(UTC) - timedelta(minutes=1)).timestamp())},
    ],
)
async def test_refuses_a_trial_that_is_no_longer_running(
    pending_trial, live, boundaries, change
):
    live.update(change)
    boundaries.retrieve.side_effect = None
    boundaries.retrieve.return_value = stripe.Subscription.construct_from(
        live, "test-key"
    )

    with pytest.raises(conversion.TrialConversionRefused, match="has ended"):
        await conversion.convert_cancel_pending_trial(pending_trial)

    boundaries.expire.assert_not_awaited()
    boundaries.modify.assert_not_awaited()
    boundaries.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_canceled_trial_is_reconciled_and_refused(
    pending_trial, live, boundaries
):
    live["status"] = "canceled"
    canceled = stripe.Subscription.construct_from(live, "test-key")
    boundaries.retrieve.side_effect = None
    boundaries.retrieve.return_value = canceled

    with pytest.raises(conversion.TrialConversionRefused, match="has ended"):
        await conversion.convert_cancel_pending_trial(pending_trial)

    boundaries.sync.assert_awaited_once_with(dict(canceled))
    boundaries.expire.assert_not_awaited()
    boundaries.modify.assert_not_awaited()


@pytest.mark.asyncio
async def test_resumed_trial_is_refused_like_any_running_trial(
    pending_trial, live, boundaries
):
    live["cancel_at_period_end"] = False
    boundaries.retrieve.side_effect = None
    boundaries.retrieve.return_value = stripe.Subscription.construct_from(
        live, "test-key"
    )

    with pytest.raises(conversion.TrialConversionRefused) as refused:
        await conversion.convert_cancel_pending_trial(pending_trial)

    assert str(refused.value) == (
        "Your accepted plan starts after your trial. Manage the trial in billing."
    )
    boundaries.expire.assert_not_awaited()
    boundaries.modify.assert_not_awaited()
    boundaries.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_card_decline_leaves_the_trial_cancel_pending(pending_trial, boundaries):
    boundaries.modify.side_effect = stripe.CardError(
        "Your card was declined.", param="card", code="card_declined"
    )

    with pytest.raises(stripe.CardError):
        await conversion.convert_cancel_pending_trial(pending_trial)

    boundaries.sync.assert_not_awaited()
    assert boundaries.calls[-1] == "unlock"


@pytest.mark.asyncio
async def test_finds_a_live_cancel_pending_trial(pending_trial):
    with patch.object(
        conversion, "get_subscription_trial", AsyncMock(return_value=pending_trial)
    ):
        assert await conversion.get_cancel_pending_trial("user-1") == pending_trial


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        {"cancel_at_period_end": False},
        {"converted_at": datetime.now(UTC)},
        {"consumed_at": None},
        {"subscription_id": None},
        {"status": "canceled"},
        {"ends_at": datetime.now(UTC) - timedelta(minutes=1)},
        {"ends_at": None},
    ],
)
async def test_other_trials_are_not_cancel_pending(pending_trial, change):
    trial = pending_trial.model_copy(update=change)
    with patch.object(
        conversion, "get_subscription_trial", AsyncMock(return_value=trial)
    ):
        assert await conversion.get_cancel_pending_trial("user-1") is None


@pytest.mark.asyncio
async def test_no_trial_is_not_cancel_pending():
    with patch.object(
        conversion, "get_subscription_trial", AsyncMock(return_value=None)
    ):
        assert await conversion.get_cancel_pending_trial("user-1") is None


@pytest.mark.parametrize(
    "tier,billing_cycle,price_id,expected",
    [
        (SubscriptionTier.PRO, "monthly", "price_pro", True),
        (SubscriptionTier.MAX, "monthly", "price_max", False),
        (SubscriptionTier.PRO, "yearly", "price_pro", False),
        (SubscriptionTier.PRO, "monthly", "price_pro_v2", False),
    ],
)
def test_only_the_accepted_plan_converts_in_place(
    pending_trial, tier, billing_cycle, price_id, expected
):
    assert (
        conversion.is_trial_plan(pending_trial, tier, billing_cycle, price_id)
        is expected
    )
