"""Deleting an account ends its Stripe subscriptions now, and marks the cancel
so the webhooks it sets off keep the tier sync but skip analytics."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import credit

CUSTOMER = "cus_123"


def _user(stripe_customer_id: str | None) -> MagicMock:
    user = MagicMock()
    user.stripe_customer_id = stripe_customer_id
    return user


def _page(*subscriptions: dict) -> MagicMock:
    page = MagicMock()
    page.data = [
        stripe.Subscription.construct_from(s, "sk_test") for s in subscriptions
    ]
    page.has_more = False
    return page


def _stripe_lists(active: MagicMock, trialing: MagicMock):
    def by_status(*_args, **kwargs):
        return trialing if kwargs.get("status") == "trialing" else active

    return by_status


async def _cancelled(sub_id: str, **params) -> stripe.Subscription:
    """Stripe echoes the cancel's comment back on the cancelled subscription."""
    details = {"reason": "cancellation_requested"}
    details.update(params.get("cancellation_details") or {})
    return stripe.Subscription.construct_from(
        {
            "id": sub_id,
            "customer": CUSTOMER,
            "status": "canceled",
            "metadata": {"trial_enrollment_id": "trial-1", "user_id": "user-1"},
            "cancellation_details": details,
        },
        "sk_test",
    )


@pytest.mark.asyncio
async def test_a_user_without_a_stripe_customer_cancels_nothing():
    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_user(None),
        ),
        patch("backend.data.credit.stripe.Subscription.list_async") as listed,
        patch("backend.data.credit.stripe.Subscription.cancel_async") as cancel,
        patch("backend.data.credit.stripe.Customer.create_async") as created,
    ):
        assert await credit.cancel_all_subscriptions_now("user-1") == 0

    listed.assert_not_called()
    cancel.assert_not_called()
    created.assert_not_called()


@pytest.mark.asyncio
async def test_paid_and_trial_subscriptions_end_now_without_an_invoice():
    active = _page({"id": "sub_paid", "schedule": None})
    trialing = _page({"id": "sub_trial", "schedule": None})
    comment = {"comment": credit.ACCOUNT_DELETION_CANCELLATION_COMMENT}
    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_user(CUSTOMER),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            side_effect=_stripe_lists(active, trialing),
        ) as listed,
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as modify,
        patch(
            "backend.data.credit.stripe.Subscription.cancel_async",
            new_callable=AsyncMock,
        ) as cancel,
    ):
        assert await credit.cancel_all_subscriptions_now("user-1") == 2

    assert {call.kwargs["customer"] for call in listed.call_args_list} == {CUSTOMER}
    cancel.assert_any_await("sub_paid", cancellation_details=comment)
    cancel.assert_any_await(
        "sub_trial", invoice_now=False, prorate=False, cancellation_details=comment
    )
    assert cancel.await_count == 2
    for call in cancel.await_args_list:
        assert call.kwargs.get("invoice_now") is not True
        assert call.kwargs.get("prorate") is not True
    # Never the at-period-end road: the account is closing now.
    modify.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["list_async", "cancel_async"])
async def test_a_stripe_error_reaches_the_caller(failing):
    active = _page({"id": "sub_paid", "schedule": None})
    trialing = _page()
    stripe_methods = {
        "list_async": AsyncMock(side_effect=_stripe_lists(active, trialing)),
        "cancel_async": AsyncMock(),
    }
    stripe_methods[failing] = AsyncMock(side_effect=stripe.StripeError("down"))
    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_user(CUSTOMER),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            stripe_methods["list_async"],
        ),
        patch(
            "backend.data.credit.stripe.Subscription.cancel_async",
            stripe_methods["cancel_async"],
        ),
    ):
        with pytest.raises(stripe.StripeError):
            await credit.cancel_all_subscriptions_now("user-1")


@pytest.mark.asyncio
async def test_an_enrolled_trial_still_syncs_its_tier_but_not_its_person():
    """The inline trial sync keeps the drop to NO_TIER, but must not refresh
    the PostHog person that the erasure deletes."""
    trialing = _page(
        {
            "id": "sub_trial",
            "schedule": None,
            "metadata": {"trial_enrollment_id": "trial-1", "user_id": "user-1"},
        }
    )
    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_user(CUSTOMER),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            side_effect=_stripe_lists(_page(), trialing),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.cancel_async",
            side_effect=_cancelled,
        ),
        patch(
            "backend.data.credit._sync_subscription_tier_from_stripe",
            new_callable=AsyncMock,
        ) as tier_sync,
        patch("backend.data.credit.schedule_posthog_lifecycle_sync") as person_sync,
    ):
        assert await credit.cancel_all_subscriptions_now("user-1") == 1

    tier_sync.assert_awaited_once()
    assert tier_sync.await_args.args[0]["id"] == "sub_trial"
    person_sync.assert_not_called()


def _deleted_event(comment: str | None) -> dict:
    details: dict = {"reason": "cancellation_requested"}
    if comment:
        details["comment"] = comment
    return {
        "id": "sub_paid",
        "customer": CUSTOMER,
        "status": "canceled",
        "items": {"data": []},
        "cancellation_details": details,
    }


@pytest.mark.asyncio
async def test_the_deletion_webhook_drops_the_tier_without_touching_the_person():
    user = MagicMock(id="user-1", subscriptionTier=SubscriptionTier.PRO)
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=user)),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            return_value=_page(),
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as set_tier,
        patch("backend.data.credit.schedule_posthog_lifecycle_sync") as person_sync,
    ):
        await credit.sync_subscription_from_stripe(
            _deleted_event(credit.ACCOUNT_DELETION_CANCELLATION_COMMENT)
        )

    set_tier.assert_awaited_once_with(
        "user-1", SubscriptionTier.NO_TIER, track_lifecycle=False
    )
    person_sync.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "comment", [None, credit.PAYMENT_FAILURE_CANCELLATION_COMMENT, "other"]
)
async def test_any_other_cancel_still_refreshes_the_person(comment):
    with (
        patch(
            "backend.data.credit._sync_subscription_tier_from_stripe",
            new_callable=AsyncMock,
        ),
        patch("backend.data.credit.schedule_posthog_lifecycle_sync") as person_sync,
    ):
        await credit.sync_subscription_from_stripe(_deleted_event(comment))

    person_sync.assert_called_once_with(stripe_customer_id=CUSTOMER)


def test_only_the_account_deletion_comment_marks_a_deletion():
    assert credit.ACCOUNT_DELETION_CANCELLATION_COMMENT == "autogpt:account_deleted"
    assert credit.is_account_deletion_cancellation(
        _deleted_event(credit.ACCOUNT_DELETION_CANCELLATION_COMMENT)
    )
    assert not credit.is_account_deletion_cancellation(
        _deleted_event(credit.PAYMENT_FAILURE_CANCELLATION_COMMENT)
    )
    assert not credit.is_account_deletion_cancellation({"id": "sub_paid"})
    assert not credit.is_account_deletion_cancellation(
        {"id": "sub_paid", "cancellation_details": None}
    )
