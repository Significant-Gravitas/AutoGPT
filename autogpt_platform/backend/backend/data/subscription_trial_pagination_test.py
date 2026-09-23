from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe

from backend.data import subscription_trial_stripe as fulfillment

pytest_plugins = ("backend.data.subscription_trial_fixtures",)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["active", "trialing"])
@pytest.mark.parametrize("page_error", [False, True])
async def test_tier_clear_checks_later_subscription_pages(
    trial, subscription, boundaries, status, page_error
):
    subscription.update(status=status, default_payment_method=None)
    first = MagicMock(
        data=[stripe.Subscription.construct_from(subscription, "test")],
        has_more=True,
    )
    second = MagicMock(
        data=[
            stripe.Subscription.construct_from(
                {"id": "sub_other", "status": status}, "test"
            )
        ],
        has_more=False,
    )
    first.next_page_async = AsyncMock(
        return_value=second,
        side_effect=(
            stripe.APIConnectionError("Later page unavailable") if page_error else None
        ),
    )
    empty = MagicMock(data=[], has_more=False)
    with patch.object(
        fulfillment.stripe.Subscription,
        "list_async",
        AsyncMock(side_effect=[first, empty] if status == "active" else [empty, first]),
    ):
        if page_error:
            with pytest.raises(stripe.APIConnectionError, match="Later page"):
                await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
        else:
            _, tier = await fulfillment._reconcile_locked(trial, "sub_1", boundaries)
            assert tier is None

    first.next_page_async.assert_awaited_once_with()
    boundaries.user.update_many.assert_not_awaited()
