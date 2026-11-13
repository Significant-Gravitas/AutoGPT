from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import credit
from backend.data import subscription_trial_conversion as conversion
from backend.data.subscription_trial import TrialState

pytest_plugins = ("backend.data.subscription_trial_fixtures",)

PRO = SubscriptionTier.PRO


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
    assert await conversion.convert_cancel_pending_trial(pending_trial, PRO) is None

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
        await conversion.convert_cancel_pending_trial(pending_trial, PRO)

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
        await conversion.convert_cancel_pending_trial(pending_trial, PRO)

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

    assert await conversion.convert_cancel_pending_trial(pending_trial, PRO) is None

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
        await conversion.convert_cancel_pending_trial(pending_trial, PRO)

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
        await conversion.convert_cancel_pending_trial(pending_trial, PRO)

    assert str(refused.value) == (
        "Your accepted plan starts after your trial. Manage the trial in billing."
    )
    boundaries.sync.assert_awaited_once()
    boundaries.expire.assert_not_awaited()
    boundaries.modify.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_card_stripe_cannot_charge_is_refused_until_it_is_updated(
    pending_trial, boundaries
):
    """Ending a card-less trial would let Stripe cancel it instead of billing
    it, and Checkout can't take a new card beside the trial's subscription."""
    boundaries.card.return_value = False

    with pytest.raises(conversion.TrialConversionRefused) as refused:
        await conversion.convert_cancel_pending_trial(pending_trial, PRO)

    assert str(refused.value) == conversion.CARD_NEEDED
    boundaries.expire.assert_not_awaited()
    boundaries.modify.assert_not_awaited()
    boundaries.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_another_plan_moves_the_trials_own_subscription_onto_its_price(
    pending_trial, boundaries
):
    """Stripe limits a customer to one subscription, so Max is bought by
    changing the trial's, with the choice recorded for the reconcile."""
    page = await conversion.convert_cancel_pending_trial(
        pending_trial, SubscriptionTier.MAX, "price_max"
    )

    assert page is None
    boundaries.modify.assert_awaited_once_with(
        "sub_1",
        cancel_at_period_end=False,
        trial_end="now",
        items=[{"id": "si_1", "price": "price_max"}],
        metadata={
            "trial_conversion_price_id": "price_max",
            "trial_conversion_tier": "MAX",
        },
        payment_behavior="error_if_incomplete",
    )
    boundaries.sync.assert_awaited_once_with(dict(boundaries.converted))


@pytest.mark.asyncio
async def test_another_plan_needs_a_subscription_with_one_plan(
    pending_trial, live, boundaries
):
    live["items"] = {
        "data": [
            {"id": "si_1", "price": {"id": "price_pro"}, "quantity": 1},
            {"id": "si_2", "price": {"id": "price_seat"}, "quantity": 1},
        ],
        "has_more": False,
    }
    boundaries.retrieve.side_effect = None
    boundaries.retrieve.return_value = stripe.Subscription.construct_from(
        live, "test-key"
    )

    with pytest.raises(ValueError, match="one plan"):
        await conversion.convert_cancel_pending_trial(
            pending_trial, SubscriptionTier.MAX, "price_max"
        )

    boundaries.modify.assert_not_awaited()


def _bank_wants_to_confirm() -> stripe.CardError:
    return stripe.CardError(
        "Authentication required.",
        param="card",
        code="authentication_required",
        json_body={"error": {"code": "authentication_required"}},
    )


def _held(live: dict, **changes) -> stripe.Subscription:
    return stripe.Subscription.construct_from(
        {
            **live,
            "pending_update": {"trial_end": 1, "expires_at": 2},
            "latest_invoice": {
                "id": "in_held",
                "status": "open",
                "created": 1,
                "hosted_invoice_url": "https://invoice.stripe.com/i/test_pay",
            },
            **changes,
        },
        "test-key",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tier,price_id,plan",
    [
        (PRO, None, {}),
        (
            SubscriptionTier.MAX,
            "price_max",
            {
                "items": [{"id": "si_1", "price": "price_max"}],
                "metadata": {
                    "trial_conversion_price_id": "price_max",
                    "trial_conversion_tier": "MAX",
                },
            },
        ),
    ],
)
async def test_a_bank_that_wants_to_confirm_gets_stripes_invoice_page(
    pending_trial, live, boundaries, tier, price_id, plan
):
    """The same change is held until its invoice is paid; meanwhile the trial
    stays cancel-pending, so nothing is saved. The hold names an API version
    that takes metadata and cancel_at_period_end in a pending update."""
    boundaries.modify.side_effect = [_bank_wants_to_confirm(), _held(live)]

    page = await conversion.convert_cancel_pending_trial(pending_trial, tier, price_id)

    assert page == "https://invoice.stripe.com/i/test_pay"
    first, hold = boundaries.modify.await_args_list
    assert "stripe_version" not in first.kwargs
    assert hold.kwargs == {
        "cancel_at_period_end": False,
        "trial_end": "now",
        **plan,
        "payment_behavior": "pending_if_incomplete",
        "expand": ["latest_invoice"],
        "stripe_version": "2026-09-30.endive",
    }
    boundaries.sync.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_held_change_that_was_paid_at_once_is_read_back_and_saved(
    pending_trial, live, boundaries
):
    """The hold's answer is in the newer API version's shape, so the plan is
    read back in the SDK's own version before it is saved."""
    boundaries.modify.side_effect = [
        _bank_wants_to_confirm(),
        _held(live, status="active", pending_update=None),
    ]
    active = stripe.Subscription.construct_from(
        {**live, "status": "active", "cancel_at_period_end": False}, "test-key"
    )
    boundaries.retrieve.side_effect = [
        stripe.Subscription.construct_from(live, "test-key"),
        active,
    ]

    assert await conversion.convert_cancel_pending_trial(pending_trial, PRO) is None

    assert boundaries.retrieve.await_args_list[-1] == call(live["id"])
    boundaries.sync.assert_awaited_once_with(dict(active))


@pytest.mark.asyncio
async def test_a_held_change_without_a_page_to_pay_on_is_an_error(
    pending_trial, live, boundaries
):
    held = _held(live, latest_invoice={"id": "in_held", "created": 1})
    boundaries.modify.side_effect = [_bank_wants_to_confirm(), held]

    with pytest.raises(ValueError, match="no page"):
        await conversion.convert_cancel_pending_trial(pending_trial, PRO)

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
        await conversion.convert_cancel_pending_trial(pending_trial, PRO)

    boundaries.sync.assert_awaited_once_with(dict(canceled))


@pytest.mark.asyncio
async def test_a_failed_sync_after_the_charge_still_reports_the_conversion(
    pending_trial, boundaries
):
    """The card is charged and the plan is live in Stripe; its webhooks save
    it, so the paying customer is not told the change failed."""
    boundaries.sync.side_effect = stripe.APIConnectionError("Stripe is unreachable")

    assert await conversion.convert_cancel_pending_trial(pending_trial, PRO) is None

    boundaries.modify.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_busy_checkout_lock_is_a_retryable_conflict(pending_trial, boundaries):
    @asynccontextmanager
    async def busy(user_id: str):
        raise conversion.SubscriptionCheckoutUnavailable("busy")
        yield

    with patch.object(conversion, "subscription_checkout_lock", busy):
        with pytest.raises(conversion.TrialConversionRefused) as refused:
            await conversion.convert_cancel_pending_trial(pending_trial, PRO)

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
        await conversion.convert_cancel_pending_trial(pending_trial, PRO)

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

    await conversion.convert_cancel_pending_trial(pending_trial, PRO)

    boundaries.modify.assert_awaited_once()
    boundaries.sync.assert_awaited_once_with(dict(boundaries.converted))


@pytest.mark.asyncio
async def test_conversion_refreshes_the_cached_subscription_lookup(
    pending_trial, live, boundaries
):
    """The status returned right after shows the paid period, not the trial's."""
    trialing = stripe.Subscription.construct_from(live, "test-key")
    with patch.object(
        credit,
        "_get_active_subscription",
        AsyncMock(side_effect=[trialing, boundaries.converted]),
    ):
        before = await credit._get_active_subscription_cached("cus_1")
        assert before is not None and before.status == "trialing"
        await conversion.convert_cancel_pending_trial(pending_trial, PRO)
        after = await credit._get_active_subscription_cached("cus_1")
        assert after is not None and after.status == "active"


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_fails", [False, True])
async def test_the_paid_plan_is_saved_before_the_lookup_is_dropped(
    pending_trial, boundaries, sync_fails
):
    """The charge has gone through: saving the plan comes first, and dropping
    the cached lookup follows even when the save fails."""

    async def failing_sync(*args, **kwargs):
        boundaries.calls.append("sync")
        raise RuntimeError("database down")

    if sync_fails:
        boundaries.sync.side_effect = failing_sync

    async def invalidate(customer_id: str) -> None:
        boundaries.calls.append(f"invalidate:{customer_id}")

    with patch.object(conversion, "invalidate_billing_lookups", invalidate):
        assert await conversion.convert_cancel_pending_trial(pending_trial, PRO) is None

    assert boundaries.calls[-4:] == ["modify", "sync", "invalidate:cus_1", "unlock"]


@pytest.mark.asyncio
async def test_card_decline_leaves_the_trial_cancel_pending(pending_trial, boundaries):
    boundaries.modify.side_effect = stripe.CardError(
        "Your card was declined.", param="card", code="card_declined"
    )

    with pytest.raises(stripe.CardError):
        await conversion.convert_cancel_pending_trial(pending_trial, PRO)

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
