"""Reviewed payment retries preserve Stripe's idempotency guarantees."""

from unittest.mock import AsyncMock, Mock

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.data import credit
from backend.data import subscription_upgrade as billing
from backend.data import subscription_upgrade_cache as cache
from backend.data.subscription_upgrade_models import (
    SubscriptionUpgradePreviewRequest,
    UpgradePreviewUnavailable,
)

pytest_plugins = ("backend.data.subscription_upgrade_fixtures",)
REQUEST = SubscriptionUpgradePreviewRequest(tier="MAX")


@pytest.fixture(autouse=True)
def graph_cleanup():
    """No graph/database resources are used by these Stripe boundary tests."""
    yield


@pytest.fixture
def tier_change(monkeypatch):
    update = AsyncMock()
    monkeypatch.setattr(credit, "set_subscription_tier", update)
    return update


@pytest.fixture
def payment(setup, monkeypatch, tier_change):
    monkeypatch.setattr(cache, "get_redis_async", AsyncMock(return_value=setup.redis))
    monkeypatch.setattr(
        credit, "get_subscription_price_id", AsyncMock(return_value="price_max")
    )
    monkeypatch.setattr(credit, "get_user_by_id", AsyncMock(return_value=setup.user))
    monkeypatch.setattr(
        credit,
        "_get_active_subscription",
        AsyncMock(return_value=stripe.Subscription.construct_from(setup.sub, "k")),
    )
    monkeypatch.setattr(credit, "_track_billing_event", Mock())
    monkeypatch.setattr(credit, "invalidate_billing_lookups", AsyncMock())
    modify = AsyncMock()
    monkeypatch.setattr(stripe.Subscription, "modify_async", modify)
    return modify


async def confirm(preview, user_id="user-1"):
    quote = await billing.validate_subscription_upgrade_preview(
        user_id, REQUEST, preview.preview_id, preview.proration_date
    )
    return await credit.modify_stripe_subscription_for_tier(
        user_id,
        SubscriptionTier.MAX,
        proration_date=preview.proration_date,
        upgrade_preview=quote,
    )


@pytest.mark.parametrize(
    "failure",
    [
        stripe.CardError("Declined", "payment_method", "card_declined"),
        stripe.CardError(
            "Authentication required",
            "payment_method",
            "subscription_payment_intent_requires_action",
        ),
        stripe.InvalidRequestError(
            "No payment method",
            "default_payment_method",
            code="resource_missing",
            http_status=400,
        ),
    ],
)
async def test_rejected_payment_requires_new_review_before_retry(
    setup, payment, failure
):
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    payment.side_effect = failure
    with pytest.raises(type(failure)) as error:
        await confirm(preview)
    assert error.value is failure
    with pytest.raises(UpgradePreviewUnavailable):
        await confirm(preview)
    assert payment.await_count == 1

    fresh = await billing.preview_subscription_upgrade("user-1", REQUEST)
    payment.side_effect = None
    assert await confirm(fresh) is True
    assert fresh.preview_id != preview.preview_id
    assert [call.kwargs["idempotency_key"] for call in payment.await_args_list] == [
        f"subscription-upgrade:{preview.preview_id}",
        f"subscription-upgrade:{fresh.preview_id}",
    ]


@pytest.mark.parametrize(
    "failure",
    [
        stripe.APIConnectionError("Connection lost"),
        stripe.APIError("Internal server error", http_status=500),
        stripe.RateLimitError("Rate limited", http_status=429),
        stripe.InvalidRequestError("Request in progress", None, http_status=409),
        stripe.IdempotencyError("Request in progress", http_status=409),
    ],
)
async def test_ambiguous_or_retryable_errors_keep_the_review_and_key(
    setup, payment, failure
):
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    payment.side_effect = failure
    with pytest.raises(type(failure)):
        await confirm(preview)
    payment.side_effect = None
    assert await confirm(preview) is True
    assert payment.await_args_list[0] == payment.await_args_list[1]
    setup.redis.delete.assert_not_awaited()


async def test_other_user_cannot_retry_or_retire_an_owned_review(setup, payment):
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    with pytest.raises(UpgradePreviewUnavailable):
        await confirm(preview, "user-2")
    payment.assert_not_awaited()
    setup.redis.delete.assert_not_awaited()
    assert await confirm(preview) is True


async def test_cache_failure_preserves_the_stripe_failure_and_does_not_change_tier(
    setup, payment, tier_change
):
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    failure = stripe.CardError("Declined", "payment_method", "card_declined")
    payment.side_effect = failure
    setup.redis.delete.side_effect = ConnectionError("Redis unavailable")
    with pytest.raises(ConnectionError) as error:
        await confirm(preview)
    assert error.value.__context__ is failure
    tier_change.assert_not_awaited()
    quote = await billing.validate_subscription_upgrade_preview(
        "user-1", REQUEST, preview.preview_id, preview.proration_date
    )
    assert quote.preview == preview


async def test_retirement_cannot_delete_another_users_review(setup, payment):
    preview = await billing.preview_subscription_upgrade("user-1", REQUEST)
    await cache.discard_subscription_upgrade_preview("user-2", preview.preview_id)
    assert await confirm(preview) is True
