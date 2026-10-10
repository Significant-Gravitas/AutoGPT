from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import credit
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


def _subscriptions(*subscriptions: dict) -> stripe.ListObject:
    return stripe.ListObject.construct_from(
        {"data": list(subscriptions), "has_more": False}, "test-key"
    )


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
    others = recorded("others", _subscriptions(live))
    with (
        patch.object(conversion, "subscription_checkout_lock", lock),
        patch.object(stripe.Subscription, "retrieve_async", retrieve),
        patch.object(stripe.Subscription, "list_async", others),
        patch.object(
            stripe.Subscription, "modify_async", recorded("modify", converted)
        ) as modify,
        patch.object(
            conversion, "expire_other_subscription_checkouts", recorded("expire")
        ) as expire,
        patch.object(
            conversion, "sync_subscription_from_stripe", recorded("sync")
        ) as sync,
        patch.object(
            conversion,
            "subscription_card_can_be_charged",
            AsyncMock(return_value=True),
        ) as card,
    ):
        yield MagicMock(
            calls=calls,
            retrieve=retrieve,
            others=others,
            modify=modify,
            expire=expire,
            sync=sync,
            card=card,
            converted=converted,
        )


@pytest.mark.asyncio
async def test_converts_cancel_pending_trial_in_place_under_checkout_lock(
    pending_trial, boundaries
):
    assert await conversion.convert_cancel_pending_trial(pending_trial)

    assert boundaries.calls == [
        "lock:user-1",
        "retrieve",
        "expire",
        "others",
        "modify",
        "sync",
        "unlock",
    ]
    boundaries.retrieve.assert_awaited_once_with("sub_1")
    assert boundaries.card.await_args.args[0] == "sub_1"
    boundaries.expire.assert_awaited_once_with("cus_1")
    boundaries.others.assert_awaited_once_with(
        customer="cus_1", status="all", limit=100
    )
    boundaries.modify.assert_awaited_once_with(
        "sub_1",
        cancel_at_period_end=False,
        trial_end="now",
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
        {"status": "past_due"},
        {"trial_end": int((datetime.now(UTC) - timedelta(minutes=1)).timestamp())},
    ],
)
async def test_refuses_a_trial_that_is_no_longer_running(
    pending_trial, live, boundaries, change
):
    """Stripe's state is saved first, so the refreshed page shows why."""
    live.update(change)
    ended = stripe.Subscription.construct_from(live, "test-key")
    boundaries.retrieve.side_effect = None
    boundaries.retrieve.return_value = ended

    with pytest.raises(conversion.TrialConversionRefused, match="has ended"):
        await conversion.convert_cancel_pending_trial(pending_trial)

    boundaries.sync.assert_awaited_once_with(dict(ended))
    boundaries.expire.assert_not_awaited()
    boundaries.modify.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_trial_stripe_already_converted_is_saved_and_succeeds(
    pending_trial, live, boundaries
):
    """A retry after a conversion whose answer was lost: the plan the person
    asked for is already on, so it is saved and reported, never charged again."""
    live.update(status="active", cancel_at_period_end=False)
    converted = stripe.Subscription.construct_from(live, "test-key")
    boundaries.retrieve.side_effect = None
    boundaries.retrieve.return_value = converted

    assert await conversion.convert_cancel_pending_trial(pending_trial)

    boundaries.sync.assert_awaited_once_with(dict(converted))
    boundaries.modify.assert_not_awaited()


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
    boundaries.sync.assert_awaited_once()
    boundaries.expire.assert_not_awaited()
    boundaries.modify.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_card_stripe_cannot_charge_sends_the_conversion_to_checkout(
    pending_trial, boundaries
):
    """Only Checkout can take a new card, and ending a card-less trial would
    let Stripe cancel it instead of billing it."""
    boundaries.card.return_value = False

    assert not await conversion.convert_cancel_pending_trial(pending_trial)

    boundaries.expire.assert_not_awaited()
    boundaries.modify.assert_not_awaited()
    boundaries.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_conversion_stripe_did_not_activate_is_saved_and_refused(
    pending_trial, live, boundaries
):
    canceled = stripe.Subscription.construct_from(
        {**live, "status": "canceled"}, "test-key"
    )
    boundaries.modify.side_effect = None
    boundaries.modify.return_value = canceled

    with pytest.raises(conversion.TrialConversionRefused, match="has ended"):
        await conversion.convert_cancel_pending_trial(pending_trial)

    boundaries.sync.assert_awaited_once_with(dict(canceled))


@pytest.mark.asyncio
async def test_a_failed_sync_after_the_charge_still_reports_the_conversion(
    pending_trial, boundaries
):
    """The card is charged and the plan is live in Stripe; its webhooks save
    it, so the paying customer is not told the change failed."""
    boundaries.sync.side_effect = stripe.APIConnectionError("Stripe is unreachable")

    assert await conversion.convert_cancel_pending_trial(pending_trial)

    boundaries.modify.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_busy_checkout_lock_is_a_retryable_conflict(pending_trial, boundaries):
    @asynccontextmanager
    async def busy(user_id: str):
        raise conversion.SubscriptionCheckoutUnavailable("busy")
        yield

    with patch.object(conversion, "subscription_checkout_lock", busy):
        with pytest.raises(conversion.TrialConversionRefused) as refused:
            await conversion.convert_cancel_pending_trial(pending_trial)

    assert str(refused.value) == "Your trial is already being updated. Please retry."
    boundaries.retrieve.assert_not_awaited()
    boundaries.modify.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["active", "trialing", "past_due", "incomplete"])
async def test_refuses_while_another_plan_is_live(
    pending_trial, live, boundaries, status
):
    """A plan bought through Checkout ends the trial when its webhook lands;
    converting first would bill the customer for both plans."""
    boundaries.others.side_effect = None
    boundaries.others.return_value = _subscriptions(
        live, {"id": "sub_max", "status": status}
    )

    with pytest.raises(conversion.TrialConversionRefused) as refused:
        await conversion.convert_cancel_pending_trial(pending_trial)

    assert str(refused.value) == (
        "Another plan is already active. Manage it in billing."
    )
    boundaries.expire.assert_awaited_once_with("cus_1")
    boundaries.modify.assert_not_awaited()
    boundaries.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_ended_plans_do_not_block_the_conversion(pending_trial, live, boundaries):
    boundaries.others.side_effect = None
    boundaries.others.return_value = _subscriptions(
        live,
        {"id": "sub_old", "status": "canceled"},
        {"id": "sub_abandoned", "status": "incomplete_expired"},
    )

    await conversion.convert_cancel_pending_trial(pending_trial)

    boundaries.modify.assert_awaited_once()
    boundaries.sync.assert_awaited_once_with(dict(boundaries.converted))


@pytest.fixture
def subscription_lookup():
    """The short-lived active-subscription lookup the status route reads."""
    lookup = credit._get_active_subscription_cached
    lookup.cache_delete("cus_1")
    yield lookup
    lookup.cache_delete("cus_1")


@pytest.mark.asyncio
async def test_conversion_refreshes_the_cached_subscription_lookup(
    pending_trial, live, boundaries, subscription_lookup
):
    """The status returned right after shows the paid period, not the trial's."""
    trialing = stripe.Subscription.construct_from(live, "test-key")
    with patch.object(
        credit,
        "_get_active_subscription",
        AsyncMock(side_effect=[trialing, boundaries.converted]),
    ):
        assert await subscription_lookup("cus_1") is trialing
        await conversion.convert_cancel_pending_trial(pending_trial)
        assert await subscription_lookup("cus_1") is boundaries.converted


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
    "tier,billing_cycle,expected",
    [
        (SubscriptionTier.PRO, "monthly", True),
        (SubscriptionTier.MAX, "monthly", False),
        (SubscriptionTier.PRO, "yearly", False),
    ],
)
def test_only_the_accepted_plan_converts_in_place(
    pending_trial, tier, billing_cycle, expected
):
    """The plan is matched on tier and cycle: the conversion bills the price
    the trial accepted, as the plan card shows, even after a re-price."""
    assert conversion.is_trial_plan(pending_trial, tier, billing_cycle) is expected
