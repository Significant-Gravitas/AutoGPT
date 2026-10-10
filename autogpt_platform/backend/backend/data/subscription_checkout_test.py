import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import credit, stripe_client, subscription_checkout
from backend.data.credit import _expire_open_subscription_sessions


@pytest.mark.asyncio
async def test_expiration_list_failure_stops_new_checkout():
    with patch(
        "backend.data.credit.stripe.checkout.Session.list_async",
        AsyncMock(side_effect=stripe.APIConnectionError("unavailable")),
    ):
        with pytest.raises(stripe.APIConnectionError):
            await _expire_open_subscription_sessions("cus_test")


@pytest.mark.asyncio
async def test_in_place_change_cannot_bypass_trial_with_stale_local_tier():
    subscription = stripe.Subscription.construct_from(
        {
            "id": "sub_trial",
            "status": "trialing",
            "schedule": None,
            "cancel_at_period_end": False,
            "items": {"data": [{"id": "si_trial"}]},
            "metadata": {"trial_enrollment_id": "trial-1"},
        },
        "test-key",
    )
    user = MagicMock(
        stripe_customer_id="cus_test", subscription_tier=SubscriptionTier.NO_TIER
    )
    with (
        patch.object(
            credit, "get_subscription_price_id", AsyncMock(return_value="price_max")
        ),
        patch.object(credit, "get_user_by_id", AsyncMock(return_value=user)),
        patch.object(
            credit, "_get_active_subscription", AsyncMock(return_value=subscription)
        ),
        patch.object(credit.stripe.Subscription, "modify_async", AsyncMock()) as modify,
        patch.object(credit, "set_subscription_tier", AsyncMock()) as promote,
        patch.object(credit, "_track_billing_event"),
    ):
        with pytest.raises(ValueError, match="trial"):
            await credit.modify_stripe_subscription_for_tier(
                "user-1", SubscriptionTier.MAX
            )
    modify.assert_not_awaited()
    promote.assert_not_awaited()


@pytest.mark.asyncio
async def test_session_completed_during_expiration_stops_new_checkout():
    sessions = stripe.ListObject.construct_from(
        {"data": [{"id": "cs_old", "mode": "subscription"}], "has_more": False},
        "test-key",
    )
    with (
        patch(
            "backend.data.credit.stripe.checkout.Session.list_async",
            AsyncMock(return_value=sessions),
        ),
        patch(
            "backend.data.credit.stripe.checkout.Session.expire_async",
            AsyncMock(side_effect=stripe.InvalidRequestError("already complete", "id")),
        ),
    ):
        with pytest.raises(stripe.InvalidRequestError):
            await _expire_open_subscription_sessions("cus_test")


@pytest.mark.asyncio
async def test_checkout_history_pagination_has_a_bounded_timeout():
    first = stripe.ListObject.construct_from(
        {
            "data": [{"id": "sub_old", "status": "canceled", "metadata": {}}],
            "has_more": True,
        },
        "test-key",
    )
    second = stripe.ListObject.construct_from(
        {
            "data": [
                {
                    "id": "sub_trial",
                    "status": "trialing",
                    "metadata": {"trial_enrollment_id": "trial-1"},
                }
            ],
            "has_more": False,
        },
        "test-key",
    )

    async def delayed_page():
        await asyncio.sleep(0.05)
        return second

    with (
        patch.object(stripe_client, "DEFAULT_TIMEOUT_SECONDS", 0.01),
        patch.object(
            subscription_checkout,
            "get_subscription_trial",
            AsyncMock(return_value=MagicMock(id="trial-1", converted_at=None)),
        ),
        patch.object(stripe.Subscription, "list_async", AsyncMock(return_value=first)),
        patch.object(
            stripe.ListObject, "next_page_async", AsyncMock(side_effect=delayed_page)
        ) as next_page,
    ):
        with pytest.raises(stripe.APIConnectionError):
            await subscription_checkout.ensure_no_unconverted_trial(
                "user-1", "cus_test"
            )
    next_page.assert_awaited_once()


def _trial_history(**subscription) -> stripe.ListObject:
    return stripe.ListObject.construct_from(
        {
            "data": [
                {
                    "id": "sub_trial",
                    "metadata": {"trial_enrollment_id": "trial-1"},
                    **subscription,
                }
            ],
            "has_more": False,
        },
        "test-key",
    )


@pytest.mark.asyncio
async def test_cancel_pending_trial_does_not_block_another_plan():
    with (
        patch.object(
            subscription_checkout,
            "get_subscription_trial",
            AsyncMock(return_value=MagicMock(id="trial-1", converted_at=None)),
        ),
        patch.object(
            stripe.Subscription,
            "list_async",
            AsyncMock(
                return_value=_trial_history(
                    status="trialing", cancel_at_period_end=True
                )
            ),
        ),
    ):
        await subscription_checkout.ensure_no_unconverted_trial("user-1", "cus_test")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "subscription",
    [
        {"status": "trialing", "cancel_at_period_end": False},
        {"status": "trialing"},
        {"status": "active", "cancel_at_period_end": True},
        {"status": "past_due", "cancel_at_period_end": True},
    ],
)
async def test_running_or_unconverted_trial_still_blocks_another_plan(subscription):
    with (
        patch.object(
            subscription_checkout,
            "get_subscription_trial",
            AsyncMock(return_value=MagicMock(id="trial-1", converted_at=None)),
        ),
        patch.object(
            stripe.Subscription,
            "list_async",
            AsyncMock(return_value=_trial_history(**subscription)),
        ),
    ):
        with pytest.raises(
            subscription_checkout.SubscriptionCheckoutUnavailable,
            match="already has a trial subscription",
        ):
            await subscription_checkout.ensure_no_unconverted_trial(
                "user-1", "cus_test"
            )


def _subscriptions(*subscriptions: dict, has_more: bool = False) -> stripe.ListObject:
    return stripe.ListObject.construct_from(
        {"data": list(subscriptions), "has_more": has_more}, "test-key"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", ["active", "trialing", "past_due", "incomplete", "unpaid", "paused"]
)
async def test_another_plan_that_has_not_ended_is_live(status):
    listed = _subscriptions(
        {"id": "sub_trial", "status": "trialing"}, {"id": "sub_max", "status": status}
    )
    with patch.object(
        stripe.Subscription, "list_async", AsyncMock(return_value=listed)
    ) as list_async:
        assert await subscription_checkout.other_plan_is_live("cus_test", "sub_trial")
    list_async.assert_awaited_once_with(customer="cus_test", status="all", limit=100)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "others",
    [
        [],
        [{"id": "sub_old", "status": "canceled"}],
        [{"id": "sub_abandoned", "status": "incomplete_expired"}],
    ],
    ids=["only-this-one", "canceled", "incomplete-expired"],
)
async def test_this_plan_and_ended_plans_are_not_another_live_plan(others):
    listed = _subscriptions({"id": "sub_trial", "status": "trialing"}, *others)
    with patch.object(
        stripe.Subscription, "list_async", AsyncMock(return_value=listed)
    ):
        assert not await subscription_checkout.other_plan_is_live(
            "cus_test", "sub_trial"
        )


@pytest.mark.asyncio
async def test_a_live_plan_on_a_later_page_is_found():
    first = _subscriptions({"id": "sub_old", "status": "canceled"}, has_more=True)
    second = _subscriptions({"id": "sub_max", "status": "active"})
    with (
        patch.object(stripe.Subscription, "list_async", AsyncMock(return_value=first)),
        patch.object(
            stripe.ListObject, "next_page_async", AsyncMock(return_value=second)
        ) as next_page,
    ):
        assert await subscription_checkout.other_plan_is_live("cus_test", "sub_trial")
    next_page.assert_awaited_once()
