"""
Tests for Stripe-based subscription tier billing.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier
from prisma.models import User

from backend.data.credit import (
    cancel_stripe_subscription,
    create_subscription_checkout,
    get_proration_credit_cents,
    handle_subscription_payment_failure,
    modify_stripe_subscription_for_tier,
    set_subscription_tier,
    sync_subscription_from_stripe,
)


@pytest.mark.asyncio
async def test_set_subscription_tier_updates_db():
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(update=AsyncMock()),
        ) as mock_prisma,
        patch("backend.data.credit.get_user_by_id"),
    ):
        await set_subscription_tier("user-1", SubscriptionTier.PRO)
        mock_prisma.return_value.update.assert_awaited_once_with(
            where={"id": "user-1"},
            data={"subscriptionTier": SubscriptionTier.PRO},
        )


@pytest.mark.asyncio
async def test_set_subscription_tier_downgrade():
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(update=AsyncMock()),
        ),
        patch("backend.data.credit.get_user_by_id"),
    ):
        # Downgrade to FREE should not raise
        await set_subscription_tier("user-1", SubscriptionTier.FREE)


def _make_user(user_id: str = "user-1", tier: SubscriptionTier = SubscriptionTier.FREE):
    mock_user = MagicMock(spec=User)
    mock_user.id = user_id
    mock_user.subscriptionTier = tier
    return mock_user


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_active():
    mock_user = _make_user()
    stripe_sub = {
        "id": "sub_new",
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
    }

    async def mock_price_id(tier: SubscriptionTier) -> str | None:
        if tier == SubscriptionTier.PRO:
            return "price_pro_monthly"
        if tier == SubscriptionTier.BUSINESS:
            return "price_biz_monthly"
        return None

    empty_list = MagicMock()
    empty_list.data = []
    empty_list.has_more = False

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=empty_list,
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        mock_set.assert_awaited_once_with("user-1", SubscriptionTier.PRO)


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_idempotent_no_write_if_unchanged():
    """Stripe retries webhooks; re-sending the same event must not re-write the DB."""
    mock_user = _make_user(tier=SubscriptionTier.PRO)
    stripe_sub = {
        "id": "sub_new",
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
    }

    async def mock_price_id(tier: SubscriptionTier) -> str | None:
        if tier == SubscriptionTier.PRO:
            return "price_pro_monthly"
        if tier == SubscriptionTier.BUSINESS:
            return "price_biz_monthly"
        return None

    empty_list = MagicMock()
    empty_list.data = []
    empty_list.has_more = False

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=empty_list,
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        mock_set.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_enterprise_not_overwritten():
    """Webhook events must never overwrite an ENTERPRISE tier (admin-managed)."""
    mock_user = _make_user(tier=SubscriptionTier.ENTERPRISE)
    stripe_sub = {
        "id": "sub_new",
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
    }

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        mock_set.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_cancelled():
    """When the only active sub is cancelled, the user is downgraded to FREE."""
    mock_user = _make_user(tier=SubscriptionTier.PRO)
    stripe_sub = {
        "id": "sub_old",
        "customer": "cus_123",
        "status": "canceled",
        "items": {"data": []},
    }
    empty_list = MagicMock()
    empty_list.data = []
    empty_list.has_more = False
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=empty_list,
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        mock_set.assert_awaited_once_with("user-1", SubscriptionTier.FREE)


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_cancelled_but_other_active_sub_exists():
    """Cancelling sub_old must NOT downgrade the user if sub_new is still active.

    This covers the race condition where `customer.subscription.deleted` for
    the old sub arrives after `customer.subscription.created` for the new sub
    was already processed. Unconditionally downgrading to FREE here would
    immediately undo the user's upgrade.
    """
    mock_user = _make_user(tier=SubscriptionTier.BUSINESS)
    stripe_sub = {
        "id": "sub_old",
        "customer": "cus_123",
        "status": "canceled",
        "items": {"data": []},
    }
    # Stripe still shows sub_new as active for this customer.
    active_list = MagicMock()
    active_list.data = [{"id": "sub_new"}]
    active_list.has_more = False
    empty_list = MagicMock()
    empty_list.data = []
    empty_list.has_more = False

    def list_side_effect(*args, **kwargs):
        if kwargs.get("status") == "active":
            return active_list
        return empty_list

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            side_effect=list_side_effect,
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        # Must NOT write FREE — another active sub is still present.
        mock_set.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_trialing():
    """status='trialing' should map to the paid tier, same as 'active'."""
    mock_user = _make_user()
    stripe_sub = {
        "id": "sub_new",
        "customer": "cus_123",
        "status": "trialing",
        "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
    }

    async def mock_price_id(tier: SubscriptionTier) -> str | None:
        if tier == SubscriptionTier.PRO:
            return "price_pro_monthly"
        if tier == SubscriptionTier.BUSINESS:
            return "price_biz_monthly"
        return None

    empty_list = MagicMock()
    empty_list.data = []
    empty_list.has_more = False

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=empty_list,
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        mock_set.assert_awaited_once_with("user-1", SubscriptionTier.PRO)


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_unknown_customer():
    stripe_sub = {
        "customer": "cus_unknown",
        "status": "active",
        "items": {"data": []},
    }
    with patch(
        "backend.data.credit.User.prisma",
        return_value=MagicMock(find_first=AsyncMock(return_value=None)),
    ):
        # Should not raise even if user not found
        await sync_subscription_from_stripe(stripe_sub)


def _make_user_with_stripe(stripe_customer_id: str | None = "cus_123") -> MagicMock:
    """Return a mock model.User with the given stripe_customer_id."""
    mock_user = MagicMock()
    mock_user.stripe_customer_id = stripe_customer_id
    return mock_user


@pytest.mark.asyncio
async def test_cancel_stripe_subscription_cancels_active():
    mock_subscriptions = MagicMock()
    mock_subscriptions.data = [{"id": "sub_abc123"}]
    mock_subscriptions.has_more = False

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_make_user_with_stripe("cus_123"),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=mock_subscriptions,
        ),
        patch("backend.data.credit.stripe.Subscription.modify") as mock_modify,
    ):
        await cancel_stripe_subscription("user-1")
        mock_modify.assert_called_once_with("sub_abc123", cancel_at_period_end=True)


@pytest.mark.asyncio
async def test_cancel_stripe_subscription_no_customer_id_returns_false():
    """Users with no stripe_customer_id return False without creating a Stripe customer."""
    result = False
    with patch(
        "backend.data.credit.get_user_by_id",
        new_callable=AsyncMock,
        return_value=_make_user_with_stripe(stripe_customer_id=None),
    ):
        result = await cancel_stripe_subscription("user-1")
    assert result is False


@pytest.mark.asyncio
async def test_cancel_stripe_subscription_multi_partial_failure():
    """First modify raises → error propagates and subsequent subs are not scheduled."""
    mock_subscriptions = MagicMock()
    mock_subscriptions.data = [{"id": "sub_first"}, {"id": "sub_second"}]
    mock_subscriptions.has_more = False

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_make_user_with_stripe("cus_123"),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=mock_subscriptions,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify",
            side_effect=stripe.StripeError("first modify failed"),
        ) as mock_modify,
        patch(
            "backend.data.credit.set_subscription_tier",
            new_callable=AsyncMock,
        ) as mock_set_tier,
    ):
        with pytest.raises(stripe.StripeError):
            await cancel_stripe_subscription("user-1")
        # Only the first modify should have been attempted.
        # _cancel_customer_subscriptions has no per-cancel try/except, so the
        # StripeError propagates immediately, aborting the loop before sub_second
        # is attempted. This is intentional fail-fast behaviour — the caller
        # (cancel_stripe_subscription) re-raises and the API handler returns 502.
        mock_modify.assert_called_once_with("sub_first", cancel_at_period_end=True)
        # DB tier must NOT be updated on the error path — the caller raises
        # before reaching set_subscription_tier.
        mock_set_tier.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_stripe_subscription_no_active():
    mock_subscriptions = MagicMock()
    mock_subscriptions.data = []
    mock_subscriptions.has_more = False

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_make_user_with_stripe("cus_123"),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=mock_subscriptions,
        ),
        patch("backend.data.credit.stripe.Subscription.cancel") as mock_cancel,
    ):
        await cancel_stripe_subscription("user-1")
        mock_cancel.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_stripe_subscription_raises_on_list_failure():
    """stripe.Subscription.list() failure propagates so DB tier is not updated."""
    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_make_user_with_stripe("cus_123"),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            side_effect=stripe.StripeError("network error"),
        ),
    ):
        with pytest.raises(stripe.StripeError):
            await cancel_stripe_subscription("user-1")


@pytest.mark.asyncio
async def test_cancel_stripe_subscription_cancels_trialing():
    """Trialing subs must also be scheduled for cancellation, else users get billed after trial end."""
    active_subs = MagicMock()
    active_subs.data = []
    active_subs.has_more = False
    trialing_subs = MagicMock()
    trialing_subs.data = [{"id": "sub_trial_123"}]
    trialing_subs.has_more = False

    def list_side_effect(*args, **kwargs):
        return trialing_subs if kwargs.get("status") == "trialing" else active_subs

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_make_user_with_stripe("cus_123"),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            side_effect=list_side_effect,
        ),
        patch("backend.data.credit.stripe.Subscription.modify") as mock_modify,
    ):
        await cancel_stripe_subscription("user-1")
        mock_modify.assert_called_once_with("sub_trial_123", cancel_at_period_end=True)


@pytest.mark.asyncio
async def test_cancel_stripe_subscription_cancels_active_and_trialing():
    """Both active AND trialing subs present → both get scheduled for cancellation, no duplicates."""
    active_subs = MagicMock()
    active_subs.data = [{"id": "sub_active_1"}]
    active_subs.has_more = False
    trialing_subs = MagicMock()
    trialing_subs.data = [{"id": "sub_trial_2"}]
    trialing_subs.has_more = False

    def list_side_effect(*args, **kwargs):
        return trialing_subs if kwargs.get("status") == "trialing" else active_subs

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_make_user_with_stripe("cus_123"),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            side_effect=list_side_effect,
        ),
        patch("backend.data.credit.stripe.Subscription.modify") as mock_modify,
    ):
        await cancel_stripe_subscription("user-1")
        modified_ids = {call.args[0] for call in mock_modify.call_args_list}
        assert modified_ids == {"sub_active_1", "sub_trial_2"}


@pytest.mark.asyncio
async def test_get_proration_credit_cents_no_stripe_customer_returns_zero():
    """Admin-granted tier users without stripe_customer_id get 0 without creating a customer."""
    with patch(
        "backend.data.credit.get_user_by_id",
        new_callable=AsyncMock,
        return_value=_make_user_with_stripe(stripe_customer_id=None),
    ) as mock_user:
        result = await get_proration_credit_cents("user-1", monthly_cost_cents=2000)
    assert result == 0
    mock_user.assert_awaited_once_with("user-1")


@pytest.mark.asyncio
async def test_get_proration_credit_cents_zero_cost_returns_zero():
    """FREE tier users (cost=0) return 0 without calling get_user_by_id."""
    with patch(
        "backend.data.credit.get_user_by_id", new_callable=AsyncMock
    ) as mock_get_user:
        result = await get_proration_credit_cents("user-1", monthly_cost_cents=0)
    assert result == 0
    mock_get_user.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_proration_credit_cents_with_active_sub():
    """User with active sub returns prorated credit based on remaining billing period."""
    import time

    now = int(time.time())
    period_start = now - 15 * 24 * 3600  # 15 days ago
    period_end = now + 15 * 24 * 3600  # 15 days ahead
    mock_sub = {
        "id": "sub_abc",
        "current_period_start": period_start,
        "current_period_end": period_end,
    }
    mock_subs = MagicMock()
    mock_subs.data = [mock_sub]

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_make_user_with_stripe("cus_123"),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=mock_subs,
        ),
    ):
        result = await get_proration_credit_cents("user-1", monthly_cost_cents=2000)
    assert result > 0
    assert result < 2000


@pytest.mark.asyncio
async def test_create_subscription_checkout_returns_url():
    mock_session = MagicMock()
    mock_session.url = "https://checkout.stripe.com/pay/cs_test_abc123"
    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value="price_pro_monthly",
        ),
        patch(
            "backend.data.credit.get_stripe_customer_id",
            new_callable=AsyncMock,
            return_value="cus_123",
        ),
        patch(
            "backend.data.credit.stripe.checkout.Session.create",
            return_value=mock_session,
        ),
    ):
        url = await create_subscription_checkout(
            user_id="user-1",
            tier=SubscriptionTier.PRO,
            success_url="https://app.example.com/success",
            cancel_url="https://app.example.com/cancel",
        )
        assert url == "https://checkout.stripe.com/pay/cs_test_abc123"


@pytest.mark.asyncio
async def test_create_subscription_checkout_no_price_raises():
    with patch(
        "backend.data.credit.get_subscription_price_id",
        new_callable=AsyncMock,
        return_value=None,
    ):
        with pytest.raises(ValueError, match="not available"):
            await create_subscription_checkout(
                user_id="user-1",
                tier=SubscriptionTier.PRO,
                success_url="https://app.example.com/success",
                cancel_url="https://app.example.com/cancel",
            )


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_missing_customer_key_returns_early():
    """A webhook payload missing 'customer' must not raise KeyError — returns early with a warning."""
    stripe_sub = {
        # Omit "customer" entirely — simulates a valid HMAC but malformed payload
        "status": "active",
        "id": "sub_xyz",
        "items": {"data": [{"price": {"id": "price_pro"}}]},
    }

    with (
        patch("backend.data.credit.User.prisma") as mock_prisma,
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        # Should return early without querying the DB or writing a tier
        await sync_subscription_from_stripe(stripe_sub)
        mock_prisma.assert_not_called()
        mock_set.assert_not_called()


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_unknown_price_id_preserves_current_tier():
    """Unknown price_id should preserve the current tier, not default to FREE (no DB write)."""
    mock_user = _make_user(tier=SubscriptionTier.PRO)
    stripe_sub = {
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_unknown"}}]},
    }

    async def mock_price_id(tier: SubscriptionTier) -> str | None:
        return "price_pro_monthly" if tier == SubscriptionTier.PRO else None

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        # Unknown price → preserve current tier (early return, no DB write)
        mock_set.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_unconfigured_ld_price_preserves_current_tier():
    """When LD flags are unconfigured (None price IDs), the current tier should be preserved, not defaulted to FREE."""
    mock_user = _make_user(tier=SubscriptionTier.PRO)
    stripe_sub = {
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
    }

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value=None,  # LD flags unconfigured
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        # None from LD → comparison guards prevent match → preserve current tier
        mock_set.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_business_tier():
    """BUSINESS price_id should map to BUSINESS tier."""
    mock_user = _make_user()
    stripe_sub = {
        "id": "sub_new",
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_biz_monthly"}}]},
    }

    async def mock_price_id(tier: SubscriptionTier) -> str | None:
        if tier == SubscriptionTier.PRO:
            return "price_pro_monthly"
        if tier == SubscriptionTier.BUSINESS:
            return "price_biz_monthly"
        return None

    empty_list = MagicMock()
    empty_list.data = []
    empty_list.has_more = False

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=empty_list,
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        mock_set.assert_awaited_once_with("user-1", SubscriptionTier.BUSINESS)


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_cancels_stale_subs():
    """When a new subscription becomes active, older active subs are cancelled.

    Covers the paid-to-paid upgrade case (e.g. PRO → BUSINESS) where Stripe
    Checkout creates a new subscription without touching the previous one,
    leaving the customer double-billed.
    """
    mock_user = _make_user(tier=SubscriptionTier.PRO)
    stripe_sub = {
        "id": "sub_new",
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_biz_monthly"}}]},
    }

    async def mock_price_id(tier: SubscriptionTier) -> str | None:
        if tier == SubscriptionTier.PRO:
            return "price_pro_monthly"
        if tier == SubscriptionTier.BUSINESS:
            return "price_biz_monthly"
        return None

    existing = MagicMock()
    existing.data = [{"id": "sub_old"}, {"id": "sub_new"}]
    existing.has_more = False

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=existing,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.cancel",
        ) as mock_cancel,
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        mock_set.assert_awaited_once_with("user-1", SubscriptionTier.BUSINESS)
        # Only the stale sub should be cancelled — never the new one.
        mock_cancel.assert_called_once_with("sub_old")


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_stale_cancel_errors_swallowed():
    """Errors cancelling stale subs must not block DB tier update for new sub."""
    import stripe as stripe_mod

    mock_user = _make_user(tier=SubscriptionTier.BUSINESS)
    stripe_sub = {
        "id": "sub_new",
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
    }

    async def mock_price_id(tier: SubscriptionTier) -> str | None:
        if tier == SubscriptionTier.PRO:
            return "price_pro_monthly"
        if tier == SubscriptionTier.BUSINESS:
            return "price_biz_monthly"
        return None

    existing = MagicMock()
    existing.data = [{"id": "sub_old"}]
    existing.has_more = False

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=existing,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.cancel",
            side_effect=stripe_mod.StripeError("cancel failed"),
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        # Must not raise — tier update proceeds even if cleanup cancel fails.
        await sync_subscription_from_stripe(stripe_sub)
        mock_set.assert_awaited_once_with("user-1", SubscriptionTier.PRO)


@pytest.mark.asyncio
async def test_get_subscription_price_id_pro():
    from backend.data.credit import get_subscription_price_id

    # Clear cached state from other tests to ensure a fresh LD flag lookup.
    get_subscription_price_id.cache_clear()  # type: ignore[attr-defined]
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value="price_pro_monthly",
    ):
        price_id = await get_subscription_price_id(SubscriptionTier.PRO)
        assert price_id == "price_pro_monthly"
    get_subscription_price_id.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_get_subscription_price_id_free_returns_none():
    from backend.data.credit import get_subscription_price_id

    # FREE tier bypasses the LD flag lookup entirely (returns None before fetch).
    price_id = await get_subscription_price_id(SubscriptionTier.FREE)
    assert price_id is None


@pytest.mark.asyncio
async def test_get_subscription_price_id_empty_flag_returns_none():
    from backend.data.credit import get_subscription_price_id

    get_subscription_price_id.cache_clear()  # type: ignore[attr-defined]
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value="",  # LD flag not set
    ):
        price_id = await get_subscription_price_id(SubscriptionTier.BUSINESS)
        assert price_id is None
    get_subscription_price_id.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_get_subscription_price_id_none_not_cached():
    """None returns from transient LD failures are not cached (cache_none=False).

    Without cache_none=False a single LD hiccup would block upgrades for the
    full 60-second TTL window because the ``None`` sentinel would be served from
    cache on every subsequent call.
    """
    from backend.data.credit import get_subscription_price_id

    get_subscription_price_id.cache_clear()  # type: ignore[attr-defined]
    mock_ld = AsyncMock(side_effect=["", "price_pro_monthly"])
    with patch("backend.data.credit.get_feature_flag_value", mock_ld):
        # First call: LD returns empty string → None (transient failure)
        first = await get_subscription_price_id(SubscriptionTier.PRO)
        assert first is None
        # Second call: LD returns the real price ID — must NOT be blocked by cached None
        second = await get_subscription_price_id(SubscriptionTier.PRO)
        assert second == "price_pro_monthly"
        assert mock_ld.call_count == 2  # both calls hit LD (None was not cached)
    get_subscription_price_id.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_cancel_stripe_subscription_raises_on_cancel_error():
    """Stripe errors during period-end scheduling are re-raised so the DB tier is not updated."""
    import stripe as stripe_mod

    mock_subscriptions = MagicMock()
    mock_subscriptions.data = [{"id": "sub_abc123"}]
    mock_subscriptions.has_more = False

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=_make_user_with_stripe("cus_123"),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=mock_subscriptions,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify",
            side_effect=stripe_mod.StripeError("network error"),
        ),
    ):
        with pytest.raises(stripe_mod.StripeError):
            await cancel_stripe_subscription("user-1")


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_metadata_user_id_matches():
    """metadata.user_id matching the DB user is accepted and the tier is updated normally."""
    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.FREE)
    stripe_sub = {
        "id": "sub_new",
        "customer": "cus_123",
        "status": "active",
        "metadata": {"user_id": "user-1"},
        "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
    }

    async def mock_price_id(tier: SubscriptionTier) -> str | None:
        return "price_pro_monthly" if tier == SubscriptionTier.PRO else None

    empty_list = MagicMock()
    empty_list.data = []
    empty_list.has_more = False

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=empty_list,
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        mock_set.assert_awaited_once_with("user-1", SubscriptionTier.PRO)


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_metadata_user_id_mismatch_blocked():
    """metadata.user_id mismatching the DB user must block the tier update.

    A customer↔user mapping inconsistency (e.g. a customer ID reassigned or
    a corrupted DB row) must never silently update the wrong user's tier.
    """
    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.FREE)
    stripe_sub = {
        "id": "sub_new",
        "customer": "cus_123",
        "status": "active",
        "metadata": {"user_id": "user-different"},
        "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
    }

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        # Mismatch → must not update any tier
        mock_set.assert_not_awaited()


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_no_metadata_user_id_skips_check():
    """Absence of metadata.user_id (e.g. subs created outside Checkout) skips the cross-check."""
    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.FREE)
    stripe_sub = {
        "id": "sub_new",
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
        # No "metadata" key at all
    }

    async def mock_price_id(tier: SubscriptionTier) -> str | None:
        return "price_pro_monthly" if tier == SubscriptionTier.PRO else None

    empty_list = MagicMock()
    empty_list.data = []
    empty_list.has_more = False

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=empty_list,
        ),
        patch(
            "backend.data.credit.set_subscription_tier", new_callable=AsyncMock
        ) as mock_set,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        # No metadata → cross-check skipped → tier updated normally
        mock_set.assert_awaited_once_with("user-1", SubscriptionTier.PRO)


@pytest.mark.asyncio
async def test_handle_subscription_payment_failure_balance_covers_pays_invoice():
    """When balance covers the invoice, Stripe Invoice.pay is called to stop retries."""
    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.PRO)
    invoice = {
        "id": "in_abc123",
        "customer": "cus_123",
        "subscription": "sub_abc123",
        "amount_due": 2000,
    }

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.UserCredit._add_transaction",
            new_callable=AsyncMock,
        ),
        patch("backend.data.credit.stripe.Invoice.pay") as mock_pay,
    ):
        await handle_subscription_payment_failure(invoice)
        mock_pay.assert_called_once_with("in_abc123")


@pytest.mark.asyncio
async def test_handle_subscription_payment_failure_invoice_pay_error_does_not_raise():
    """Failure to mark the invoice as paid is logged but does not propagate."""
    import stripe as stripe_mod

    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.PRO)
    invoice = {
        "id": "in_abc123",
        "customer": "cus_123",
        "subscription": "sub_abc123",
        "amount_due": 2000,
    }

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.UserCredit._add_transaction",
            new_callable=AsyncMock,
        ),
        patch(
            "backend.data.credit.stripe.Invoice.pay",
            side_effect=stripe_mod.StripeError("network error"),
        ),
    ):
        # Must not raise — the pay failure is only logged as a warning
        await handle_subscription_payment_failure(invoice)


@pytest.mark.asyncio
async def test_handle_subscription_payment_failure_passes_invoice_id_as_transaction_key():
    """invoice_id is used as the idempotency key to prevent double-charging on webhook retries."""
    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.PRO)
    invoice = {
        "id": "in_idempotency_test",
        "customer": "cus_123",
        "subscription": "sub_abc123",
        "amount_due": 2000,
    }

    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.UserCredit._add_transaction",
            new_callable=AsyncMock,
        ) as mock_add_tx,
        patch("backend.data.credit.stripe.Invoice.pay"),
    ):
        await handle_subscription_payment_failure(invoice)
        mock_add_tx.assert_called_once()
        _, kwargs = mock_add_tx.call_args
        assert kwargs.get("transaction_key") == "in_idempotency_test"


@pytest.mark.asyncio
async def test_modify_stripe_subscription_for_tier_modifies_existing_sub():
    """modify_stripe_subscription_for_tier calls Subscription.modify and returns True."""
    mock_sub = {
        "id": "sub_abc",
        "items": {"data": [{"id": "si_abc"}]},
    }
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"

    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value="price_pro_monthly",
        ),
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify",
        ) as mock_modify,
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.PRO
        )

    assert result is True
    mock_modify.assert_called_once_with(
        "sub_abc",
        items=[{"id": "si_abc", "price": "price_pro_monthly"}],
        proration_behavior="create_prorations",
    )


@pytest.mark.asyncio
async def test_modify_stripe_subscription_for_tier_returns_false_when_no_customer_id():
    """modify_stripe_subscription_for_tier returns False when user has no Stripe customer ID.

    Admin-granted paid tiers have no Stripe customer record.  Calling
    get_stripe_customer_id would create an orphaned customer if a subsequent API call
    fails, so the function returns False early and the API layer falls back to Checkout.
    """
    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = None

    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value="price_pro_monthly",
        ),
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.PRO
        )

    assert result is False


@pytest.mark.asyncio
async def test_modify_stripe_subscription_for_tier_returns_false_when_no_sub():
    """modify_stripe_subscription_for_tier returns False when no active subscription exists."""
    mock_list = MagicMock()
    mock_list.data = []

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"

    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value="price_pro_monthly",
        ),
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list",
            return_value=mock_list,
        ),
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.PRO
        )

    assert result is False


@pytest.mark.asyncio
async def test_modify_stripe_subscription_for_tier_raises_on_missing_price_id():
    """modify_stripe_subscription_for_tier raises ValueError when no price ID is configured."""
    with patch(
        "backend.data.credit.get_subscription_price_id",
        new_callable=AsyncMock,
        return_value=None,
    ):
        with pytest.raises(ValueError, match="No Stripe price ID configured"):
            await modify_stripe_subscription_for_tier("user-1", SubscriptionTier.PRO)
