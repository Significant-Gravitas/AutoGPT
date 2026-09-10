from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data.credit import _cancel_customer_subscriptions
from backend.data.stripe_reconciliation import _collect_status_page


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [ValueError("Missing enrollment"), stripe.APIConnectionError("Unavailable")],
)
async def test_stale_trial_does_not_abort_later_pages(error: Exception):
    stale = stripe.Subscription.construct_from(
        {
            "id": "sub_stale",
            "customer": "cus_stale",
            "metadata": {"trial_enrollment_id": "missing"},
        },
        "test",
    )
    paid = stripe.Subscription.construct_from(
        {
            "id": "sub_paid",
            "customer": "cus_paid",
            "items": {"data": [{"price": {"id": "price_pro"}}]},
        },
        "test",
    )
    pages = [
        MagicMock(data=[stale], has_more=True),
        MagicMock(data=[paid], has_more=False),
    ]
    with (
        patch(
            "backend.data.stripe_reconciliation.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            side_effect=pages,
        ) as listing,
        patch(
            "backend.data.stripe_reconciliation.sync_subscription_from_stripe",
            new_callable=AsyncMock,
            side_effect=error,
        ),
    ):
        tiers: dict[str, SubscriptionTier] = {}
        incomplete = await _collect_status_page(
            "active", {"price_pro": SubscriptionTier.PRO}, tiers
        )
    assert incomplete is True
    assert tiers == {"cus_paid": SubscriptionTier.PRO}
    assert listing.await_args_list[1].kwargs["starting_after"] == "sub_stale"


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["active", "trialing"])
async def test_cancellation_follows_all_pages(status: str):
    subscriptions = [
        stripe.Subscription.construct_from(
            {
                "id": f"sub_{index}",
                "schedule": None,
                "metadata": (
                    {"trial_enrollment_id": "trial"} if status == "trialing" else {}
                ),
            },
            "test",
        )
        for index in range(11)
    ]
    second = MagicMock(data=subscriptions[10:], has_more=False)
    first = MagicMock(data=subscriptions[:10], has_more=True)
    first.next_page_async = AsyncMock(return_value=second)
    empty = MagicMock(data=[], has_more=False)
    with (
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            side_effect=[first, empty] if status == "active" else [empty, first],
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as modify,
        patch(
            "backend.data.credit.stripe.Subscription.cancel_async",
            new_callable=AsyncMock,
            side_effect=subscriptions,
        ) as cancel,
        patch(
            "backend.data.credit.sync_subscription_from_stripe", new_callable=AsyncMock
        ) as sync,
    ):
        count = await _cancel_customer_subscriptions("cus_test", at_period_end=True)
    assert count == 11
    first.next_page_async.assert_awaited_once_with()
    if status == "active":
        assert modify.await_count == 11
        modify.assert_any_await("sub_10", cancel_at_period_end=True)
        cancel.assert_not_awaited()
    else:
        assert cancel.await_count == 11
        cancel.assert_any_await("sub_10", invoice_now=False, prorate=False)
        assert sync.await_count == 11
        modify.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancellation_propagates_later_page_failure():
    first = MagicMock(
        data=[
            stripe.Subscription.construct_from(
                {"id": "sub_first", "schedule": None}, "test"
            )
        ],
        has_more=True,
    )
    first.next_page_async = AsyncMock(
        side_effect=stripe.APIConnectionError("Next page unavailable")
    )
    with (
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=first,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ),
        pytest.raises(stripe.APIConnectionError, match="Next page unavailable"),
    ):
        await _cancel_customer_subscriptions("cus_test", at_period_end=True)
