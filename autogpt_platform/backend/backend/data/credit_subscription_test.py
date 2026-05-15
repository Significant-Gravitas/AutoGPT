"""
Tests for Stripe-based subscription tier billing.
"""

from typing import Protocol
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier
from prisma.errors import UniqueViolationError
from prisma.models import User

from backend.data.credit import (
    cancel_stripe_subscription,
    create_subscription_checkout,
    get_pending_subscription_change,
    get_proration_credit_cents,
    handle_subscription_payment_failure,
    handle_subscription_payment_success,
    is_tier_downgrade,
    is_tier_upgrade,
    modify_stripe_subscription_for_tier,
    release_pending_subscription_schedule,
    set_subscription_tier,
    sync_subscription_from_stripe,
    sync_subscription_schedule_from_stripe,
)


class _CacheClearable(Protocol):
    """Type-only view of the cache_clear attribute attached to functions
    decorated with ``@cached`` in ``backend.util.cache``. Lets the test file
    invoke ``cache_clear`` without ``# type: ignore[attr-defined]`` while
    still checking the call site for typos at type-check time."""

    def cache_clear(self) -> None: ...


def _clear_cache(fn: _CacheClearable) -> None:
    fn.cache_clear()


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
        # Downgrade to BASIC should not raise
        await set_subscription_tier("user-1", SubscriptionTier.BASIC)


def _make_user(
    user_id: str = "user-1", tier: SubscriptionTier = SubscriptionTier.BASIC
):
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

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        if billing_cycle != "monthly":
            return None
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
async def test_sync_subscription_from_stripe_yearly_pro_maps_to_pro():
    """A user on a yearly Pro plan still maps to SubscriptionTier.PRO."""
    mock_user = _make_user()
    stripe_sub = {
        "id": "sub_yearly",
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_pro_yearly"}}]},
    }

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        if tier == SubscriptionTier.PRO and billing_cycle == "monthly":
            return "price_pro_monthly"
        if tier == SubscriptionTier.PRO and billing_cycle == "yearly":
            return "price_pro_yearly"
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

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        if billing_cycle != "monthly":
            return None
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
    """When the only active sub is cancelled, the user is downgraded to NO_TIER."""
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
        mock_set.assert_awaited_once_with("user-1", SubscriptionTier.NO_TIER)


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_cancelled_applies_no_tier_storage_limit():
    """After unsubscribe takes effect, workspace storage resolves against NO_TIER."""
    from backend.copilot.rate_limit import get_workspace_storage_limit_bytes

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

    async def _set_tier(_user_id: str, tier: SubscriptionTier) -> None:
        mock_user.subscriptionTier = tier

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
            "backend.data.credit.set_subscription_tier",
            new_callable=AsyncMock,
            side_effect=_set_tier,
        ),
        patch(
            "backend.copilot.rate_limit.get_user_tier",
            new_callable=AsyncMock,
            side_effect=lambda _user_id: mock_user.subscriptionTier,
        ),
        patch(
            "backend.copilot.rate_limit.get_workspace_storage_limits_mb",
            new_callable=AsyncMock,
            return_value={
                "NO_TIER": 250,
                "BASIC": 500,
                "PRO": 1024,
                "MAX": 5 * 1024,
                "BUSINESS": 15 * 1024,
                "ENTERPRISE": 15 * 1024,
            },
        ),
        patch.object(
            get_pending_subscription_change,
            "cache_delete",
        ) as mock_pending_cache_delete,
    ):
        await sync_subscription_from_stripe(stripe_sub)
        result = await get_workspace_storage_limit_bytes("user-1")

    assert result == 250 * 1024 * 1024
    mock_pending_cache_delete.assert_called_once_with("user-1")


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_cancelled_but_other_active_sub_exists():
    """Cancelling sub_old must NOT downgrade the user if sub_new is still active.

    This covers the race condition where `customer.subscription.deleted` for
    the old sub arrives after `customer.subscription.created` for the new sub
    was already processed. Unconditionally downgrading to BASIC here would
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
        # Must NOT write BASIC — another active sub is still present.
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

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
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
    mock_subscriptions.data = [
        stripe.Subscription.construct_from(
            {"id": "sub_abc123", "schedule": None}, "sk_test"
        )
    ]
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
    mock_subscriptions.data = [
        stripe.Subscription.construct_from(
            {"id": "sub_first", "schedule": None}, "sk_test"
        ),
        stripe.Subscription.construct_from(
            {"id": "sub_second", "schedule": None}, "sk_test"
        ),
    ]
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
    trialing_subs.data = [
        stripe.Subscription.construct_from(
            {"id": "sub_trial_123", "schedule": None}, "sk_test"
        )
    ]
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
    active_subs.data = [
        stripe.Subscription.construct_from(
            {"id": "sub_active_1", "schedule": None}, "sk_test"
        )
    ]
    active_subs.has_more = False
    trialing_subs = MagicMock()
    trialing_subs.data = [
        stripe.Subscription.construct_from(
            {"id": "sub_trial_2", "schedule": None}, "sk_test"
        )
    ]
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
async def test_cancel_stripe_subscription_releases_attached_schedule_first():
    """Pre-existing Subscription Schedule must be released before cancel_at_period_end.

    Stripe rejects ``modify(cancel_at_period_end=True)`` with HTTP 400 when the
    subscription has an attached schedule (e.g. user queued a BUSINESS→PRO
    downgrade and now clicks "Downgrade to BASIC"). Without the pre-release,
    the API handler would surface a 502 to the user.
    """
    mock_subscriptions = MagicMock()
    mock_subscriptions.data = [
        stripe.Subscription.construct_from(
            {"id": "sub_abc123", "schedule": "sub_sched_abc"}, "sk_test"
        )
    ]
    mock_subscriptions.has_more = False

    call_order: list[str] = []

    async def record_release(schedule_id):
        call_order.append(f"release:{schedule_id}")

    def record_modify(sub_id, **kwargs):
        call_order.append(f"modify:{sub_id}:{kwargs}")

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
            "backend.data.credit.stripe.SubscriptionSchedule.release_async",
            new_callable=AsyncMock,
            side_effect=record_release,
        ) as mock_release,
        patch(
            "backend.data.credit.stripe.Subscription.modify",
            side_effect=record_modify,
        ) as mock_modify,
    ):
        await cancel_stripe_subscription("user-1")

    mock_release.assert_awaited_once_with("sub_sched_abc")
    mock_modify.assert_called_once_with("sub_abc123", cancel_at_period_end=True)
    # Release must happen before modify, else Stripe returns 400.
    assert call_order == [
        "release:sub_sched_abc",
        "modify:sub_abc123:{'cancel_at_period_end': True}",
    ]


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
    """BASIC tier users (cost=0) return 0 without calling get_user_by_id."""
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
    """Unknown price_id should preserve the current tier, not default to BASIC (no DB write)."""
    mock_user = _make_user(tier=SubscriptionTier.PRO)
    stripe_sub = {
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_unknown"}}]},
    }

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
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
    """When LD flags are unconfigured (None price IDs), the current tier should be preserved, not defaulted to BASIC."""
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

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
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
async def test_sync_subscription_from_stripe_basic_tier_via_ld_price():
    """BASIC price_id via LD should reconcile the user to BASIC.

    Protects the new stripe-price-id-basic reconciliation path — webhooks for a
    priced-BASIC sub must flip the DB tier back to BASIC when the active Stripe
    item matches the configured basic price.
    """
    mock_user = _make_user(tier=SubscriptionTier.PRO)
    stripe_sub = {
        "id": "sub_new",
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_basic_monthly"}}]},
    }

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        if tier == SubscriptionTier.BASIC:
            return "price_basic_monthly"
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
        mock_set.assert_awaited_once_with("user-1", SubscriptionTier.BASIC)


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

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
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

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
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
    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value={"PRO": "price_pro_monthly", "MAX": "price_max_monthly"},
    ):
        price_id = await get_subscription_price_id(SubscriptionTier.PRO)
        assert price_id == "price_pro_monthly"
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_basic_returns_ld_flag():
    from backend.data.credit import get_subscription_price_id

    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value={"BASIC": "price_basic_monthly"},
    ):
        price_id = await get_subscription_price_id(SubscriptionTier.BASIC)
        assert price_id == "price_basic_monthly"
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_max():
    from backend.data.credit import get_subscription_price_id

    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value={"MAX": "price_max_monthly"},
    ):
        price_id = await get_subscription_price_id(SubscriptionTier.MAX)
        assert price_id == "price_max_monthly"
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_enterprise_returns_none():
    from backend.data.credit import get_subscription_price_id

    # ENTERPRISE is never a key in the JSON flag → resolves to None.
    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value={"PRO": "price_pro_monthly"},
    ):
        price_id = await get_subscription_price_id(SubscriptionTier.ENTERPRISE)
        assert price_id is None
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_empty_flag_returns_none():
    """Empty dict payload → every tier resolves to None."""
    from backend.data.credit import get_subscription_price_id

    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value={},
    ):
        for tier in (
            SubscriptionTier.BASIC,
            SubscriptionTier.PRO,
            SubscriptionTier.MAX,
            SubscriptionTier.BUSINESS,
            SubscriptionTier.ENTERPRISE,
        ):
            assert await get_subscription_price_id(tier) is None
            _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_partial_dict():
    """Dict with only PRO resolves PRO, other tiers return None."""
    from backend.data.credit import get_subscription_price_id

    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value={"PRO": "price_pro_monthly"},
    ):
        assert (
            await get_subscription_price_id(SubscriptionTier.PRO) == "price_pro_monthly"
        )
        _clear_cache(get_subscription_price_id)
        assert await get_subscription_price_id(SubscriptionTier.MAX) is None
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_empty_string_value_returns_none():
    """Empty-string price_id in the JSON resolves to None (defensive)."""
    from backend.data.credit import get_subscription_price_id

    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value={"PRO": ""},
    ):
        assert await get_subscription_price_id(SubscriptionTier.PRO) is None
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_non_string_value_returns_none():
    """Non-string price_id (e.g. number, null) resolves to None."""
    from backend.data.credit import get_subscription_price_id

    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value={"PRO": 123, "MAX": None},
    ):
        assert await get_subscription_price_id(SubscriptionTier.PRO) is None
        _clear_cache(get_subscription_price_id)
        assert await get_subscription_price_id(SubscriptionTier.MAX) is None
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_non_dict_payload_returns_none(caplog):
    """A non-dict LD value logs a warning and returns None for every tier."""
    from backend.data.credit import get_subscription_price_id

    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value="not-a-dict",
    ):
        with caplog.at_level("WARNING"):
            price_id = await get_subscription_price_id(SubscriptionTier.PRO)
        assert price_id is None
        assert any(
            "copilot-tier-stripe-prices" in rec.message for rec in caplog.records
        )
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_ld_raises_returns_none():
    """Transient LD failure (here surfaced as a caught exception inside
    get_feature_flag_value → default None) returns None and does not raise."""
    from backend.data.credit import get_subscription_price_id

    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value=None,
    ):
        assert await get_subscription_price_id(SubscriptionTier.PRO) is None
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_none_not_cached():
    """None returns from transient LD failures are not cached (cache_none=False).

    Without cache_none=False a single LD hiccup would block upgrades for the
    full 60-second TTL window because the ``None`` sentinel would be served from
    cache on every subsequent call.
    """
    from backend.data.credit import get_subscription_price_id

    _clear_cache(get_subscription_price_id)
    mock_ld = AsyncMock(side_effect=[None, {"PRO": "price_pro_monthly"}])
    with patch("backend.data.credit.get_feature_flag_value", mock_ld):
        # First call: LD returns None (transient failure)
        first = await get_subscription_price_id(SubscriptionTier.PRO)
        assert first is None
        # Second call: LD returns the real price dict — must NOT be blocked by cached None
        second = await get_subscription_price_id(SubscriptionTier.PRO)
        assert second == "price_pro_monthly"
        assert mock_ld.call_count == 2  # both calls hit LD (None was not cached)
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_yearly_suffix_key():
    """Yearly request reads the <TIER>_YEARLY key."""
    from backend.data.credit import get_subscription_price_id

    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value={"PRO": "price_pro_m", "PRO_YEARLY": "price_pro_y"},
    ):
        assert (
            await get_subscription_price_id(SubscriptionTier.PRO, "yearly")
            == "price_pro_y"
        )
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_monthly_unchanged_after_yearly_addition():
    """Adding a yearly suffix key never changes the monthly lookup. Old code
    that only reads <TIER> keeps returning the monthly price unaffected."""
    from backend.data.credit import get_subscription_price_id

    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value={"PRO": "price_pro_m", "PRO_YEARLY": "price_pro_y"},
    ):
        assert (
            await get_subscription_price_id(SubscriptionTier.PRO, "monthly")
            == "price_pro_m"
        )
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_yearly_missing_returns_none():
    """No <TIER>_YEARLY key → yearly request fails closed (no silent monthly)."""
    from backend.data.credit import get_subscription_price_id

    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value={"PRO": "price_pro_m"},
    ):
        assert await get_subscription_price_id(SubscriptionTier.PRO, "yearly") is None
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_get_subscription_price_id_monthly_only_flag_serves_monthly():
    """Backward compat: a monthly-only flag value (pre-yearly rollout) keeps
    serving monthly requests exactly as before this PR."""
    from backend.data.credit import get_subscription_price_id

    _clear_cache(get_subscription_price_id)
    with patch(
        "backend.data.credit.get_feature_flag_value",
        new_callable=AsyncMock,
        return_value={"PRO": "price_pro_monthly_only"},
    ):
        assert (
            await get_subscription_price_id(SubscriptionTier.PRO, "monthly")
            == "price_pro_monthly_only"
        )
    _clear_cache(get_subscription_price_id)


@pytest.mark.asyncio
async def test_cancel_stripe_subscription_raises_on_cancel_error():
    """Stripe errors during period-end scheduling are re-raised so the DB tier is not updated."""
    import stripe as stripe_mod

    mock_subscriptions = MagicMock()
    mock_subscriptions.data = [
        stripe.Subscription.construct_from(
            {"id": "sub_abc123", "schedule": None}, "sk_test"
        )
    ]
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
    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.BASIC)
    stripe_sub = {
        "id": "sub_new",
        "customer": "cus_123",
        "status": "active",
        "metadata": {"user_id": "user-1"},
        "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
    }

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
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
    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.BASIC)
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
    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.BASIC)
    stripe_sub = {
        "id": "sub_new",
        "customer": "cus_123",
        "status": "active",
        "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
        # No "metadata" key at all
    }

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
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
    """When balance covers the invoice, Stripe Invoice.pay is called with
    paid_out_of_band=True so the card isn't double-charged on top of the
    balance debit (the card already failed; retrying it would let the
    success-handler webhook reverse the debit via the credit grant)."""
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
        mock_pay.assert_called_once_with("in_abc123", paid_out_of_band=True)


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


def _patch_credit_grant_config(enabled: bool):
    """Patch ``settings.config.enable_subscription_credit_grant`` for the
    success handler's gate check.

    The setting is OFF by default, so the success-handler tests that exercise
    the grant path must explicitly opt in; the disabled-path test below
    asserts the default behaviour.
    """
    return patch(
        "backend.data.credit.settings.config.enable_subscription_credit_grant",
        new=enabled,
    )


@pytest.mark.asyncio
async def test_handle_subscription_payment_success_grants_credits_when_enabled():
    """When the credit-grant config is on, a paid sub invoice grants credits
    equal to ``amount_paid`` keyed by invoice id."""
    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.PRO)
    invoice = {
        "id": "in_abc123",
        "customer": "cus_123",
        "subscription": "sub_abc123",
        "amount_paid": 5000,
        "billing_reason": "subscription_cycle",
    }

    add_tx_mock = AsyncMock()
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.UserCredit._add_transaction",
            new=add_tx_mock,
        ),
        _patch_credit_grant_config(True),
    ):
        await handle_subscription_payment_success(invoice)

    add_tx_mock.assert_awaited_once()
    kwargs = add_tx_mock.await_args.kwargs
    assert kwargs["amount"] == 5000
    assert kwargs["transaction_key"] == "INVOICE-in_abc123"


@pytest.mark.asyncio
async def test_handle_subscription_payment_success_skips_when_disabled():
    """Default-off: with the config disabled, an otherwise grant-eligible
    invoice is a no-op. Guards against an accidental re-enablement of the
    post-#12933 behaviour (Pro Monthly subscribers receiving matching
    automation credits).

    Asserts against the REAL default (no patching of the flag) so flipping
    ``enable_subscription_credit_grant`` to True in settings.py fails this
    test rather than silently passing.
    """
    from backend.util.settings import Settings

    assert Settings().config.enable_subscription_credit_grant is False, (
        "Default flipped — this regression test asserts the OFF default; "
        "either revert the default or update the assertion intentionally."
    )

    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.PRO)
    invoice = {
        "id": "in_flag_off",
        "customer": "cus_123",
        "subscription": "sub_abc123",
        "amount_paid": 5000,
        "billing_reason": "subscription_cycle",
    }
    add_tx_mock = AsyncMock()
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.UserCredit._add_transaction",
            new=add_tx_mock,
        ),
    ):
        await handle_subscription_payment_success(invoice)

    add_tx_mock.assert_not_called()


@pytest.mark.asyncio
async def test_handle_subscription_payment_success_skips_non_subscription_invoice():
    """Invoices with no subscription field (one-off invoices) are no-ops —
    short-circuit lands before the flag check, so no LD eval is needed."""
    invoice = {
        "id": "in_abc123",
        "customer": "cus_123",
        "amount_paid": 5000,
        # No 'subscription' field
    }

    prisma_mock = MagicMock()
    with patch("backend.data.credit.User.prisma", return_value=prisma_mock):
        await handle_subscription_payment_success(invoice)
    prisma_mock.find_first.assert_not_called()


@pytest.mark.asyncio
async def test_handle_subscription_payment_success_skips_paid_out_of_band():
    """When the failure handler covered the invoice from the user's balance and
    marked it ``paid_out_of_band=True``, the success-handler webhook that
    follows must NOT grant credits — doing so would reverse the balance debit
    and effectively give the user a free billing period."""
    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.PRO)
    invoice = {
        "id": "in_oob_123",
        "customer": "cus_123",
        "subscription": "sub_abc123",
        "amount_paid": 5000,
        "billing_reason": "subscription_cycle",
        "paid_out_of_band": True,
    }

    add_tx_mock = AsyncMock()
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.UserCredit._add_transaction",
            new=add_tx_mock,
        ),
        _patch_credit_grant_config(True),
    ):
        await handle_subscription_payment_success(invoice)
    add_tx_mock.assert_not_called()


@pytest.mark.asyncio
async def test_handle_subscription_payment_success_skips_zero_amount():
    """Zero-amount invoices (card validation, $0 trials) are no-ops."""
    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.PRO)
    invoice = {
        "id": "in_abc123",
        "customer": "cus_123",
        "subscription": "sub_abc123",
        "amount_paid": 0,
    }

    add_tx_mock = AsyncMock()
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.UserCredit._add_transaction",
            new=add_tx_mock,
        ),
        _patch_credit_grant_config(True),
    ):
        await handle_subscription_payment_success(invoice)
    add_tx_mock.assert_not_called()


@pytest.mark.asyncio
async def test_handle_subscription_payment_success_skips_missing_customer():
    """Invoices missing the customer field are dropped with a warning."""
    invoice = {
        "id": "in_abc",
        "subscription": "sub_abc",
        "amount_paid": 1000,
    }
    prisma_mock = MagicMock()
    with patch("backend.data.credit.User.prisma", return_value=prisma_mock):
        await handle_subscription_payment_success(invoice)
    prisma_mock.find_first.assert_not_called()


@pytest.mark.asyncio
async def test_handle_subscription_payment_success_skips_unknown_user():
    """Invoices for an unknown stripeCustomerId are dropped with a warning."""
    invoice = {
        "id": "in_abc",
        "customer": "cus_unknown",
        "subscription": "sub_abc",
        "amount_paid": 1000,
    }
    add_tx_mock = AsyncMock()
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=None)),
        ),
        patch(
            "backend.data.credit.UserCredit._add_transaction",
            new=add_tx_mock,
        ),
        _patch_credit_grant_config(True),
    ):
        await handle_subscription_payment_success(invoice)
    add_tx_mock.assert_not_called()


@pytest.mark.asyncio
async def test_handle_subscription_payment_success_skips_enterprise():
    """ENTERPRISE users don't get credit grants from Stripe invoices."""
    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.ENTERPRISE)
    invoice = {
        "id": "in_abc",
        "customer": "cus_123",
        "subscription": "sub_abc",
        "amount_paid": 5000,
    }
    add_tx_mock = AsyncMock()
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.UserCredit._add_transaction",
            new=add_tx_mock,
        ),
        _patch_credit_grant_config(True),
    ):
        await handle_subscription_payment_success(invoice)
    add_tx_mock.assert_not_called()


@pytest.mark.asyncio
async def test_handle_subscription_payment_success_idempotent_on_unique_violation():
    """If the GRANT transaction key already exists (Stripe webhook retry),
    UniqueViolationError is swallowed so the webhook returns 200 and Stripe
    stops retrying."""
    mock_user = _make_user(user_id="user-1", tier=SubscriptionTier.PRO)
    invoice = {
        "id": "in_abc",
        "customer": "cus_123",
        "subscription": "sub_abc",
        "amount_paid": 5000,
    }
    add_tx_mock = AsyncMock(side_effect=UniqueViolationError({"error": "dup"}))
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=mock_user)),
        ),
        patch(
            "backend.data.credit.UserCredit._add_transaction",
            new=add_tx_mock,
        ),
        _patch_credit_grant_config(True),
    ):
        await handle_subscription_payment_success(invoice)
    add_tx_mock.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_active_subscription_period_end_returns_unix_timestamp():
    """Happy path: returns int(current_period_end) for an active sub."""
    mock_sub = stripe.Subscription.construct_from(
        {"id": "sub_abc", "current_period_end": 1779340148}, "k"
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]
    user = MagicMock(spec=User)
    user.stripe_customer_id = "cus_abc"

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
    ):
        from backend.data.credit import get_active_subscription_period_end

        result = await get_active_subscription_period_end("user-1")
    assert result == 1779340148


@pytest.mark.asyncio
async def test_get_active_subscription_period_end_returns_none_without_customer():
    """Users without a Stripe customer ID return None — no Stripe API call."""
    user = MagicMock(spec=User)
    user.stripe_customer_id = None
    list_mock = AsyncMock()

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new=list_mock,
        ),
    ):
        from backend.data.credit import get_active_subscription_period_end

        result = await get_active_subscription_period_end("user-1")
    assert result is None
    list_mock.assert_not_called()


@pytest.mark.asyncio
async def test_get_active_subscription_period_end_swallows_stripe_errors():
    """A Stripe error during the lookup returns None instead of raising."""
    user = MagicMock(spec=User)
    user.stripe_customer_id = "cus_abc"

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            side_effect=stripe.StripeError("boom"),
        ),
    ):
        from backend.data.credit import get_active_subscription_period_end

        result = await get_active_subscription_period_end("user-1")
    assert result is None


@pytest.mark.asyncio
async def test_top_up_intent_uses_inline_product_data_when_flag_unset():
    """When STRIPE_PRODUCT_ID_TOPUP flag is undefined (default), top-up Checkout
    creates an ephemeral product per session via product_data."""
    from backend.data.credit import UserCredit

    mock_session = MagicMock()
    mock_session.id = "cs_test_topup"
    mock_session.url = "https://checkout.stripe.com/c/cs_test_topup"
    create_mock = MagicMock(return_value=mock_session)
    credit_system = UserCredit()
    with (
        patch(
            "backend.data.credit.get_stripe_customer_id",
            new_callable=AsyncMock,
            return_value="cus_123",
        ),
        patch(
            "backend.data.credit.get_feature_flag_value",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "backend.data.credit.stripe.checkout.Session.create",
            new=create_mock,
        ),
        patch.object(credit_system, "_add_transaction", new_callable=AsyncMock),
    ):
        await credit_system.top_up_intent(user_id="user-1", amount=500)

    price_data = create_mock.call_args.kwargs["line_items"][0]["price_data"]
    assert price_data == {
        "currency": "usd",
        "unit_amount": 500,
        "product_data": {"name": "AutoGPT Platform Credits"},
    }


@pytest.mark.asyncio
async def test_top_up_intent_references_product_id_when_flag_set():
    """When STRIPE_PRODUCT_ID_TOPUP flag returns a string, top-up Checkout
    references the canonical Product ID and keeps the per-session amount via
    unit_amount."""
    from backend.data.credit import UserCredit

    mock_session = MagicMock()
    mock_session.id = "cs_test_topup"
    mock_session.url = "https://checkout.stripe.com/c/cs_test_topup"
    create_mock = MagicMock(return_value=mock_session)
    credit_system = UserCredit()
    with (
        patch(
            "backend.data.credit.get_stripe_customer_id",
            new_callable=AsyncMock,
            return_value="cus_123",
        ),
        patch(
            "backend.data.credit.get_feature_flag_value",
            new_callable=AsyncMock,
            return_value="prod_abc123",
        ),
        patch(
            "backend.data.credit.stripe.checkout.Session.create",
            new=create_mock,
        ),
        patch.object(credit_system, "_add_transaction", new_callable=AsyncMock),
    ):
        await credit_system.top_up_intent(user_id="user-1", amount=2500)

    price_data = create_mock.call_args.kwargs["line_items"][0]["price_data"]
    assert price_data == {
        "currency": "usd",
        "unit_amount": 2500,
        "product": "prod_abc123",
    }
    # No product_data — that path is mutually exclusive with product reference.
    assert "product_data" not in price_data


@pytest.mark.asyncio
async def test_modify_stripe_subscription_for_tier_modifies_existing_sub():
    """modify_stripe_subscription_for_tier calls Subscription.modify and returns True."""
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_abc",
            "items": {"data": [{"id": "si_abc"}]},
            "schedule": None,
            "cancel_at_period_end": False,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.BASIC

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
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
        patch(
            "backend.data.credit.set_subscription_tier",
            new_callable=AsyncMock,
        ) as mock_set_tier,
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.PRO
        )

    assert result is True
    mock_modify.assert_called_once_with(
        "sub_abc",
        items=[{"id": "si_abc", "price": "price_pro_monthly"}],
        proration_behavior="always_invoice",
        payment_behavior="error_if_incomplete",
        metadata={
            "user_id": "user-1",
            "tier": SubscriptionTier.PRO.value,
            "billing_cycle": "monthly",
        },
    )
    mock_set_tier.assert_awaited_once_with("user-1", SubscriptionTier.PRO)


@pytest.mark.asyncio
async def test_modify_stripe_subscription_for_tier_clears_cancel_at_period_end_on_upgrade():
    """Upgrading from a sub with cancel_at_period_end=True clears the flag so the
    upgrade isn't silently cancelled at period end and the DB tier flips immediately."""
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_upgrading",
            "items": {"data": [{"id": "si_abc"}]},
            "schedule": None,
            "cancel_at_period_end": True,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.PRO

    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value="price_biz_monthly",
        ),
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
        patch(
            "backend.data.credit.set_subscription_tier",
            new_callable=AsyncMock,
        ) as mock_set_tier,
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.BUSINESS
        )

    assert result is True
    mock_modify.assert_called_once_with(
        "sub_upgrading",
        items=[{"id": "si_abc", "price": "price_biz_monthly"}],
        proration_behavior="always_invoice",
        payment_behavior="error_if_incomplete",
        metadata={
            "user_id": "user-1",
            "tier": SubscriptionTier.BUSINESS.value,
            "billing_cycle": "monthly",
        },
        cancel_at_period_end=False,
    )
    mock_set_tier.assert_awaited_once_with("user-1", SubscriptionTier.BUSINESS)


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
    mock_user.subscription_tier = SubscriptionTier.BASIC

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
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
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


def test_tier_order_helpers():
    assert is_tier_upgrade(SubscriptionTier.BASIC, SubscriptionTier.PRO) is True
    assert is_tier_upgrade(SubscriptionTier.PRO, SubscriptionTier.BUSINESS) is True
    assert is_tier_upgrade(SubscriptionTier.BUSINESS, SubscriptionTier.PRO) is False
    assert is_tier_downgrade(SubscriptionTier.BUSINESS, SubscriptionTier.PRO) is True
    assert is_tier_downgrade(SubscriptionTier.PRO, SubscriptionTier.BASIC) is True
    assert is_tier_downgrade(SubscriptionTier.PRO, SubscriptionTier.BUSINESS) is False


@pytest.mark.asyncio
async def test_modify_stripe_subscription_for_tier_downgrade_creates_schedule():
    """Paid→paid downgrade (BUSINESS→PRO) creates a Subscription Schedule rather than proration."""
    import time as time_mod

    now = int(time_mod.time())
    period_end = now + 27 * 24 * 3600
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_biz",
            "items": {"data": [{"id": "si_biz", "price": {"id": "price_biz_monthly"}}]},
            "current_period_start": now - 3 * 24 * 3600,
            "current_period_end": period_end,
            "schedule": None,
            "cancel_at_period_end": False,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.BUSINESS

    mock_schedule = stripe.SubscriptionSchedule.construct_from(
        {"id": "sub_sched_1"}, "k"
    )

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
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.create_async",
            new_callable=AsyncMock,
            return_value=mock_schedule,
        ) as mock_schedule_create,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.modify_async",
            new_callable=AsyncMock,
        ) as mock_schedule_modify,
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.PRO
        )

    assert result is True
    # Did NOT call Subscription.modify with proration (no immediate tier change).
    mock_modify.assert_not_called()
    mock_schedule_create.assert_called_once_with(from_subscription="sub_biz")
    assert mock_schedule_modify.call_count == 1
    _, kwargs = mock_schedule_modify.call_args
    phases = kwargs["phases"]
    assert phases[0]["items"][0]["price"] == "price_biz_monthly"
    assert phases[0]["end_date"] == period_end
    assert phases[1]["items"][0]["price"] == "price_pro_monthly"
    assert phases[0]["proration_behavior"] == "none"
    assert phases[1]["proration_behavior"] == "none"


@pytest.mark.asyncio
async def test_modify_stripe_subscription_for_tier_upgrade_immediate_proration():
    """PRO→BUSINESS upgrade still uses Subscription.modify with proration (no schedule)."""
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro",
            "items": {"data": [{"id": "si_pro", "price": {"id": "price_pro_monthly"}}]},
            "schedule": None,
            "cancel_at_period_end": False,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.PRO

    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value="price_biz_monthly",
        ),
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.create_async",
            new_callable=AsyncMock,
        ) as mock_schedule_create,
        patch(
            "backend.data.credit.set_subscription_tier",
            new_callable=AsyncMock,
        ),
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.BUSINESS
        )

    assert result is True
    mock_modify.assert_called_once_with(
        "sub_pro",
        items=[{"id": "si_pro", "price": "price_biz_monthly"}],
        proration_behavior="always_invoice",
        payment_behavior="error_if_incomplete",
        metadata={
            "user_id": "user-1",
            "tier": SubscriptionTier.BUSINESS.value,
            "billing_cycle": "monthly",
        },
    )
    mock_schedule_create.assert_not_called()


@pytest.mark.asyncio
async def test_modify_stripe_subscription_for_tier_pro_to_max_bills_immediately():
    """Pro→Max upgrade calls Stripe with always_invoice + error_if_incomplete so the
    user is charged the prorated amount immediately rather than at next cycle, and
    the DB tier flip lands once Stripe confirms payment success."""
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro",
            "items": {"data": [{"id": "si_pro", "price": {"id": "price_pro_monthly"}}]},
            "schedule": None,
            "cancel_at_period_end": False,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.PRO

    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value="price_max_monthly",
        ),
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
        patch(
            "backend.data.credit.set_subscription_tier",
            new_callable=AsyncMock,
        ) as mock_set_tier,
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.MAX
        )

    assert result is True
    mock_modify.assert_called_once_with(
        "sub_pro",
        items=[{"id": "si_pro", "price": "price_max_monthly"}],
        proration_behavior="always_invoice",
        payment_behavior="error_if_incomplete",
        metadata={
            "user_id": "user-1",
            "tier": SubscriptionTier.MAX.value,
            "billing_cycle": "monthly",
        },
    )
    mock_set_tier.assert_awaited_once_with("user-1", SubscriptionTier.MAX)


@pytest.mark.asyncio
async def test_modify_stripe_subscription_for_tier_pro_to_max_card_decline_does_not_flip_tier():
    """Pro→Max upgrade where Stripe raises CardError (auto-charge declined under
    payment_behavior=error_if_incomplete): the function must propagate the error
    AND must not call set_subscription_tier — the user stays on Pro."""
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro",
            "items": {"data": [{"id": "si_pro", "price": {"id": "price_pro_monthly"}}]},
            "schedule": None,
            "cancel_at_period_end": False,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.PRO

    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value="price_max_monthly",
        ),
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
            side_effect=stripe.CardError(
                "Your card was declined.", param="card", code="card_declined"
            ),
        ),
        patch(
            "backend.data.credit.set_subscription_tier",
            new_callable=AsyncMock,
        ) as mock_set_tier,
    ):
        with pytest.raises(stripe.CardError):
            await modify_stripe_subscription_for_tier("user-1", SubscriptionTier.MAX)

    mock_set_tier.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_pending_subscription_schedule_releases_downgrade_schedule():
    """release_pending_subscription_schedule releases the Stripe schedule if one is attached."""
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_biz",
            "schedule": "sub_sched_1",
            "cancel_at_period_end": False,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock()
    mock_user.stripe_customer_id = "cus_abc"

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.release_async",
            new_callable=AsyncMock,
        ) as mock_release,
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
    ):
        result = await release_pending_subscription_schedule("user-1")

    assert result is True
    mock_release.assert_called_once_with("sub_sched_1")
    mock_modify.assert_not_called()


@pytest.mark.asyncio
async def test_release_pending_subscription_schedule_clears_cancel_at_period_end():
    """release_pending_subscription_schedule reverts a pending paid→BASIC cancel."""
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro",
            "schedule": None,
            "cancel_at_period_end": True,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock()
    mock_user.stripe_customer_id = "cus_abc"

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.release_async",
            new_callable=AsyncMock,
        ) as mock_release,
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
    ):
        result = await release_pending_subscription_schedule("user-1")

    assert result is True
    mock_modify.assert_called_once_with("sub_pro", cancel_at_period_end=False)
    mock_release.assert_not_called()


@pytest.mark.asyncio
async def test_release_pending_subscription_schedule_no_pending_change_returns_false():
    """release_pending_subscription_schedule returns False when no schedule/cancel is set."""
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro",
            "schedule": None,
            "cancel_at_period_end": False,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock()
    mock_user.stripe_customer_id = "cus_abc"

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
    ):
        result = await release_pending_subscription_schedule("user-1")

    assert result is False


@pytest.mark.asyncio
async def test_release_pending_subscription_schedule_no_stripe_customer_returns_false():
    mock_user = MagicMock()
    mock_user.stripe_customer_id = None

    with patch(
        "backend.data.credit.get_user_by_id",
        new_callable=AsyncMock,
        return_value=mock_user,
    ):
        result = await release_pending_subscription_schedule("user-1")

    assert result is False


@pytest.mark.asyncio
async def test_get_pending_subscription_change_cancel_at_period_end():
    """cancel_at_period_end=True maps to pending NO_TIER at current_period_end."""
    import time as time_mod

    _clear_cache(get_pending_subscription_change)

    now = int(time_mod.time())
    period_end = now + 10 * 24 * 3600
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro",
            "current_period_end": period_end,
            "cancel_at_period_end": True,
            "schedule": None,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock()
    mock_user.stripe_customer_id = "cus_abc"

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        if tier == SubscriptionTier.PRO:
            return "price_pro_monthly"
        if tier == SubscriptionTier.BUSINESS:
            return "price_biz_monthly"
        return None

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
    ):
        result = await get_pending_subscription_change("user-1")

    assert result is not None
    pending_tier, effective_at, pending_cycle = result
    assert pending_tier == SubscriptionTier.NO_TIER
    assert int(effective_at.timestamp()) == period_end
    assert pending_cycle is None


@pytest.mark.asyncio
async def test_get_pending_subscription_change_from_schedule():
    """A schedule whose next phase uses the PRO price maps to pending_tier=PRO."""
    import time as time_mod

    _clear_cache(get_pending_subscription_change)

    now = int(time_mod.time())
    period_end = now + 10 * 24 * 3600
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_biz",
            "current_period_end": period_end,
            "cancel_at_period_end": False,
            "schedule": "sub_sched_1",
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_schedule = stripe.SubscriptionSchedule.construct_from(
        {
            "id": "sub_sched_1",
            "phases": [
                {
                    "start_date": now - 3 * 24 * 3600,
                    "end_date": period_end,
                    "items": [{"price": "price_biz_monthly"}],
                },
                {
                    "start_date": period_end,
                    "items": [{"price": "price_pro_monthly"}],
                },
            ],
        },
        "k",
    )

    mock_user = MagicMock()
    mock_user.stripe_customer_id = "cus_abc"

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        if billing_cycle == "yearly":
            return None
        if tier == SubscriptionTier.PRO:
            return "price_pro_monthly"
        if tier == SubscriptionTier.BUSINESS:
            return "price_biz_monthly"
        return None

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.retrieve_async",
            new_callable=AsyncMock,
            return_value=mock_schedule,
        ),
    ):
        result = await get_pending_subscription_change("user-1")

    assert result is not None
    pending_tier, effective_at, pending_cycle = result
    assert pending_tier == SubscriptionTier.PRO
    assert int(effective_at.timestamp()) == period_end
    assert pending_cycle == "monthly"


@pytest.mark.asyncio
async def test_get_pending_subscription_change_yearly_next_phase_maps_to_tier():
    """A schedule whose next phase uses a YEARLY price still resolves to the
    correct tier — yearly subscribers must see their pending downgrade in UI."""
    import time as time_mod

    _clear_cache(get_pending_subscription_change)

    now = int(time_mod.time())
    period_end = now + 10 * 24 * 3600
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_biz_yearly",
            "current_period_end": period_end,
            "cancel_at_period_end": False,
            "schedule": "sub_sched_yearly",
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_schedule = stripe.SubscriptionSchedule.construct_from(
        {
            "id": "sub_sched_yearly",
            "phases": [
                {
                    "start_date": now - 3 * 24 * 3600,
                    "end_date": period_end,
                    "items": [{"price": "price_biz_yearly"}],
                },
                {
                    "start_date": period_end,
                    "items": [{"price": "price_pro_yearly"}],
                },
            ],
        },
        "k",
    )

    mock_user = MagicMock()
    mock_user.stripe_customer_id = "cus_abc"

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        if billing_cycle == "yearly":
            if tier == SubscriptionTier.PRO:
                return "price_pro_yearly"
            if tier == SubscriptionTier.BUSINESS:
                return "price_biz_yearly"
            return None
        # Monthly entries also configured (mixed config) — confirms the lookup
        # picks up the yearly mapping rather than only seeing monthly.
        if tier == SubscriptionTier.PRO:
            return "price_pro_monthly"
        if tier == SubscriptionTier.BUSINESS:
            return "price_biz_monthly"
        return None

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.retrieve_async",
            new_callable=AsyncMock,
            return_value=mock_schedule,
        ),
    ):
        result = await get_pending_subscription_change("user-yearly")

    assert result is not None
    pending_tier, effective_at, pending_cycle = result
    assert pending_tier == SubscriptionTier.PRO
    assert int(effective_at.timestamp()) == period_end
    assert pending_cycle == "yearly"


@pytest.mark.asyncio
async def test_get_pending_subscription_change_from_schedule_to_basic():
    """A schedule whose next phase uses the BASIC price maps to pending_tier=BASIC."""
    import time as time_mod

    _clear_cache(get_pending_subscription_change)

    now = int(time_mod.time())
    period_end = now + 10 * 24 * 3600
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro",
            "current_period_end": period_end,
            "cancel_at_period_end": False,
            "schedule": "sub_sched_2",
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_schedule = stripe.SubscriptionSchedule.construct_from(
        {
            "id": "sub_sched_2",
            "phases": [
                {
                    "start_date": now - 3 * 24 * 3600,
                    "end_date": period_end,
                    "items": [{"price": "price_pro_monthly"}],
                },
                {
                    "start_date": period_end,
                    "items": [{"price": "price_basic_monthly"}],
                },
            ],
        },
        "k",
    )

    mock_user = MagicMock()
    mock_user.stripe_customer_id = "cus_abc"

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        if billing_cycle == "yearly":
            return None
        if tier == SubscriptionTier.BASIC:
            return "price_basic_monthly"
        if tier == SubscriptionTier.PRO:
            return "price_pro_monthly"
        if tier == SubscriptionTier.BUSINESS:
            return "price_biz_monthly"
        return None

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.retrieve_async",
            new_callable=AsyncMock,
            return_value=mock_schedule,
        ),
    ):
        result = await get_pending_subscription_change("user-1")

    assert result is not None
    pending_tier, effective_at, pending_cycle = result
    assert pending_tier == SubscriptionTier.BASIC
    assert int(effective_at.timestamp()) == period_end
    assert pending_cycle == "monthly"


@pytest.mark.asyncio
async def test_get_pending_subscription_change_none_when_no_schedule_or_cancel():
    """Returns None when neither a schedule nor cancel_at_period_end is set."""
    import time as time_mod

    _clear_cache(get_pending_subscription_change)

    now = int(time_mod.time())
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro",
            "current_period_end": now + 10 * 24 * 3600,
            "cancel_at_period_end": False,
            "schedule": None,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock()
    mock_user.stripe_customer_id = "cus_abc"

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        return {
            SubscriptionTier.PRO: "price_pro",
            SubscriptionTier.BUSINESS: "price_biz",
        }.get(tier)

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
    ):
        result = await get_pending_subscription_change("user-1")

    assert result is None


@pytest.mark.asyncio
async def test_get_pending_subscription_change_same_tier_yearly_to_monthly_reports_cycle():
    """Same-tier yearly→monthly schedule must surface the next-phase cycle so the
    UI can describe a cycle-only switch correctly (pending_tier == current tier
    would otherwise look like a no-op or a confusing same-tier "downgrade")."""
    import time as time_mod

    _clear_cache(get_pending_subscription_change)

    now = int(time_mod.time())
    period_end = now + 200 * 24 * 3600
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro_yearly",
            "current_period_end": period_end,
            "cancel_at_period_end": False,
            "schedule": "sub_sched_cycle",
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_schedule = stripe.SubscriptionSchedule.construct_from(
        {
            "id": "sub_sched_cycle",
            "phases": [
                {
                    "start_date": now - 5 * 24 * 3600,
                    "end_date": period_end,
                    "items": [{"price": "price_pro_yearly"}],
                },
                {
                    "start_date": period_end,
                    "items": [{"price": "price_pro_monthly"}],
                },
            ],
        },
        "k",
    )

    mock_user = MagicMock()
    mock_user.stripe_customer_id = "cus_abc"

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        if billing_cycle == "yearly":
            return {
                SubscriptionTier.PRO: "price_pro_yearly",
                SubscriptionTier.MAX: "price_max_yearly",
            }.get(tier)
        return {
            SubscriptionTier.PRO: "price_pro_monthly",
            SubscriptionTier.MAX: "price_max_monthly",
        }.get(tier)

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.retrieve_async",
            new_callable=AsyncMock,
            return_value=mock_schedule,
        ),
    ):
        result = await get_pending_subscription_change("user-yearly-to-monthly")

    assert result is not None
    pending_tier, _, pending_cycle = result
    assert pending_tier == SubscriptionTier.PRO
    assert pending_cycle == "monthly"


@pytest.mark.asyncio
async def test_sync_subscription_schedule_from_stripe_retrieves_and_delegates():
    """subscription_schedule.released triggers a sync via the active subscription object."""
    stripe_schedule = {"id": "sub_sched_1", "subscription": "sub_pro"}
    retrieved_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro",
            "customer": "cus_abc",
            "status": "active",
            "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
        },
        "k",
    )

    with (
        patch(
            "backend.data.credit.stripe.Subscription.retrieve_async",
            new_callable=AsyncMock,
            return_value=retrieved_sub,
        ) as mock_retrieve,
        patch(
            "backend.data.credit.sync_subscription_from_stripe",
            new_callable=AsyncMock,
        ) as mock_sync,
    ):
        await sync_subscription_schedule_from_stripe(stripe_schedule)

    mock_retrieve.assert_called_once_with("sub_pro")
    mock_sync.assert_awaited_once()
    forwarded = mock_sync.call_args.args[0]
    assert forwarded["id"] == "sub_pro"
    assert forwarded["customer"] == "cus_abc"


@pytest.mark.asyncio
async def test_sync_subscription_schedule_from_stripe_uses_released_subscription_fallback():
    """subscription_schedule.released events clear `subscription` and set
    `released_subscription`; the sync handler must fall back to that id."""
    stripe_schedule = {
        "id": "sub_sched_1",
        "subscription": None,
        "released_subscription": "sub_pro_released",
    }
    retrieved_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro_released",
            "customer": "cus_abc",
            "status": "active",
            "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
        },
        "k",
    )

    with (
        patch(
            "backend.data.credit.stripe.Subscription.retrieve_async",
            new_callable=AsyncMock,
            return_value=retrieved_sub,
        ) as mock_retrieve,
        patch(
            "backend.data.credit.sync_subscription_from_stripe",
            new_callable=AsyncMock,
        ) as mock_sync,
    ):
        await sync_subscription_schedule_from_stripe(stripe_schedule)

    mock_retrieve.assert_called_once_with("sub_pro_released")
    mock_sync.assert_awaited_once()


@pytest.mark.asyncio
async def test_sync_subscription_schedule_from_stripe_missing_sub_id_returns():
    """A schedule event with no 'subscription' field is logged and ignored."""
    with patch(
        "backend.data.credit.stripe.Subscription.retrieve_async",
        new_callable=AsyncMock,
    ) as mock_retrieve:
        await sync_subscription_schedule_from_stripe({"id": "sub_sched_1"})
    mock_retrieve.assert_not_called()


@pytest.mark.asyncio
async def test_sync_subscription_from_stripe_phase_transition_updates_tier():
    """When a schedule advances phases, Stripe fires customer.subscription.updated with
    the new price — the existing sync handler must update the DB tier accordingly."""
    mock_user = _make_user(tier=SubscriptionTier.BUSINESS)
    stripe_sub = {
        "id": "sub_pro",
        "customer": "cus_abc",
        "status": "active",
        # Phase advanced: price is now PRO (was BUSINESS before).
        "items": {"data": [{"price": {"id": "price_pro_monthly"}}]},
    }

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
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
async def test_release_schedule_idempotent_on_terminal_state():
    """SubscriptionSchedule.release raising InvalidRequestError on a terminal-state
    schedule is treated as success; we still continue to the cancel_at_period_end clear.
    """
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_biz",
            "schedule": "sub_sched_terminal",
            "cancel_at_period_end": True,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock()
    mock_user.stripe_customer_id = "cus_abc"

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.release_async",
            new_callable=AsyncMock,
            side_effect=stripe.InvalidRequestError(
                "Schedule has already been released",
                param="schedule",
            ),
        ) as mock_release,
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
    ):
        result = await release_pending_subscription_schedule("user-1")

    # Terminal-state release is treated as idempotent success; modify still runs.
    assert result is True
    mock_release.assert_called_once_with("sub_sched_terminal")
    mock_modify.assert_called_once_with("sub_biz", cancel_at_period_end=False)


@pytest.mark.asyncio
async def test_schedule_downgrade_releases_existing_schedule():
    """_schedule_downgrade_at_period_end releases any pre-existing schedule first."""
    import time as time_mod

    now = int(time_mod.time())
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_biz",
            "schedule": "sub_sched_old",
            "cancel_at_period_end": False,
            "items": {"data": [{"id": "si_biz", "price": {"id": "price_biz_monthly"}}]},
            "current_period_start": now - 3 * 24 * 3600,
            "current_period_end": now + 27 * 24 * 3600,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.BUSINESS

    mock_new_schedule = stripe.SubscriptionSchedule.construct_from(
        {"id": "sub_sched_new"}, "k"
    )

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
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.release_async",
            new_callable=AsyncMock,
        ) as mock_release,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.create_async",
            new_callable=AsyncMock,
            return_value=mock_new_schedule,
        ) as mock_create,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.modify_async",
            new_callable=AsyncMock,
        ),
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.PRO
        )

    assert result is True
    # Existing schedule released before creating the new one.
    mock_release.assert_called_once_with("sub_sched_old")
    mock_create.assert_called_once_with(from_subscription="sub_biz")
    # cancel_at_period_end was False, so Subscription.modify should not be called.
    mock_modify.assert_not_called()


@pytest.mark.asyncio
async def test_schedule_downgrade_clears_cancel_at_period_end():
    """_schedule_downgrade_at_period_end clears cancel_at_period_end before scheduling."""
    import time as time_mod

    now = int(time_mod.time())
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_biz",
            "schedule": None,
            "cancel_at_period_end": True,
            "items": {"data": [{"id": "si_biz", "price": {"id": "price_biz_monthly"}}]},
            "current_period_start": now - 3 * 24 * 3600,
            "current_period_end": now + 27 * 24 * 3600,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.BUSINESS

    mock_new_schedule = stripe.SubscriptionSchedule.construct_from(
        {"id": "sub_sched_new"}, "k"
    )

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
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.create_async",
            new_callable=AsyncMock,
            return_value=mock_new_schedule,
        ) as mock_create,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.modify_async",
            new_callable=AsyncMock,
        ),
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.PRO
        )

    assert result is True
    # cancel_at_period_end cleared before new schedule is created.
    mock_modify.assert_called_once_with("sub_biz", cancel_at_period_end=False)
    mock_create.assert_called_once_with(from_subscription="sub_biz")


@pytest.mark.asyncio
async def test_schedule_downgrade_rolls_back_orphan_on_modify_failure():
    """If SubscriptionSchedule.modify fails after a successful create, the
    orphaned schedule must be released so it doesn't stay attached and block
    future changes. The original StripeError re-raises to the caller.
    """
    import time as time_mod

    now = int(time_mod.time())
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_biz",
            "schedule": None,
            "cancel_at_period_end": False,
            "items": {"data": [{"id": "si_biz", "price": {"id": "price_biz_monthly"}}]},
            "current_period_start": now - 3 * 24 * 3600,
            "current_period_end": now + 27 * 24 * 3600,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.BUSINESS

    mock_new_schedule = stripe.SubscriptionSchedule.construct_from(
        {"id": "sub_sched_new"}, "k"
    )

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
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ),
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.create_async",
            new_callable=AsyncMock,
            return_value=mock_new_schedule,
        ) as mock_create,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.modify_async",
            new_callable=AsyncMock,
            side_effect=stripe.APIConnectionError("network down"),
        ) as mock_schedule_modify,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.release_async",
            new_callable=AsyncMock,
        ) as mock_release,
    ):
        with pytest.raises(stripe.APIConnectionError):
            await modify_stripe_subscription_for_tier("user-1", SubscriptionTier.PRO)

    mock_create.assert_called_once_with(from_subscription="sub_biz")
    mock_schedule_modify.assert_called_once()
    # Rollback must release the freshly-created (and now orphaned) schedule
    # id, not the pre-existing one (there was none here).
    mock_release.assert_called_once_with("sub_sched_new")


@pytest.mark.asyncio
async def test_release_ignoring_terminal_reraises_non_terminal_error():
    """_release_schedule_ignoring_terminal only swallows terminal-state errors.
    Typos / wrong ids / 404s surface so bugs aren't silently masked.
    """
    from backend.data.credit import _release_schedule_ignoring_terminal

    with patch(
        "backend.data.credit.stripe.SubscriptionSchedule.release_async",
        new_callable=AsyncMock,
        side_effect=stripe.InvalidRequestError(
            "No such subscription_schedule: 'sub_sched_typo'",
            param="schedule",
        ),
    ):
        with pytest.raises(stripe.InvalidRequestError):
            await _release_schedule_ignoring_terminal("sub_sched_typo", "test_context")


@pytest.mark.asyncio
async def test_release_ignoring_terminal_swallows_terminal_error():
    """Terminal-state messages are treated as idempotent success and return False."""
    from backend.data.credit import _release_schedule_ignoring_terminal

    with patch(
        "backend.data.credit.stripe.SubscriptionSchedule.release_async",
        new_callable=AsyncMock,
        side_effect=stripe.InvalidRequestError(
            "Schedule has already been released",
            param="schedule",
        ),
    ):
        result = await _release_schedule_ignoring_terminal(
            "sub_sched_done", "test_context"
        )

    assert result is False


@pytest.mark.asyncio
async def test_upgrade_releases_pending_schedule():
    """modify_stripe_subscription_for_tier upgrade path releases attached schedule first."""
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro",
            "schedule": "sub_sched_pending_downgrade",
            "cancel_at_period_end": False,
            "items": {"data": [{"id": "si_pro", "price": {"id": "price_pro_monthly"}}]},
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.PRO

    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value="price_biz_monthly",
        ),
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.release_async",
            new_callable=AsyncMock,
        ) as mock_release,
        patch(
            "backend.data.credit.set_subscription_tier",
            new_callable=AsyncMock,
        ),
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.BUSINESS
        )

    assert result is True
    # Pending schedule released before the upgrade modify call.
    mock_release.assert_called_once_with("sub_sched_pending_downgrade")
    mock_modify.assert_called_once_with(
        "sub_pro",
        items=[{"id": "si_pro", "price": "price_biz_monthly"}],
        proration_behavior="always_invoice",
        payment_behavior="error_if_incomplete",
        metadata={
            "user_id": "user-1",
            "tier": SubscriptionTier.BUSINESS.value,
            "billing_cycle": "monthly",
        },
    )


@pytest.mark.asyncio
async def test_modify_stripe_subscription_for_tier_upgrade_refreshes_metadata():
    """Pro→Max yearly upgrade must overwrite the original sub.metadata so the
    Stripe Dashboard and downstream observability reflect the new tier+cycle
    instead of the stale values left over from the original checkout."""
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro",
            "items": {"data": [{"id": "si_pro", "price": {"id": "price_pro_monthly"}}]},
            "schedule": None,
            "cancel_at_period_end": False,
            "metadata": {
                "user_id": "user-1",
                "tier": SubscriptionTier.PRO.value,
                "billing_cycle": "monthly",
            },
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.PRO

    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value="price_max_yearly",
        ),
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
        patch(
            "backend.data.credit.set_subscription_tier",
            new_callable=AsyncMock,
        ),
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.MAX, billing_cycle="yearly"
        )

    assert result is True
    mock_modify.assert_called_once_with(
        "sub_pro",
        items=[{"id": "si_pro", "price": "price_max_yearly"}],
        proration_behavior="always_invoice",
        payment_behavior="error_if_incomplete",
        metadata={
            "user_id": "user-1",
            "tier": SubscriptionTier.MAX.value,
            "billing_cycle": "yearly",
        },
    )


@pytest.mark.asyncio
async def test_next_phase_tier_and_start_logs_unknown_price(caplog):
    """_next_phase_tier_and_start emits a warning when the next-phase price is unmapped."""
    import logging
    import time as time_mod

    from backend.data.credit import _next_phase_tier_and_start

    now = int(time_mod.time())
    schedule = stripe.SubscriptionSchedule.construct_from(
        {
            "id": "sub_sched_unknown",
            "phases": [
                {
                    "start_date": now - 3 * 24 * 3600,
                    "end_date": now + 27 * 24 * 3600,
                    "items": [{"price": "price_current"}],
                },
                {
                    "start_date": now + 27 * 24 * 3600,
                    "items": [{"price": "price_unknown"}],
                },
            ],
        },
        "k",
    )
    price_to_tier = {"price_pro_monthly": SubscriptionTier.PRO}
    price_to_cycle = {"price_pro_monthly": "monthly"}

    with caplog.at_level(logging.WARNING, logger="backend.data.credit"):
        result = _next_phase_tier_and_start(
            schedule,
            price_to_tier,
            price_to_cycle,  # type: ignore[arg-type]
        )

    assert result is None
    assert any(
        "next_phase_tier_and_start: unknown price price_unknown" in record.message
        and "sub_sched_unknown" in record.message
        for record in caplog.records
    )


@pytest.mark.asyncio
async def test_get_pending_subscription_change_raises_when_price_lookups_fail():
    """When both LD price lookups return None, raise PendingChangeUnknown so the
    @cached wrapper doesn't store None and hide pending changes for 30s."""
    from backend.data.credit import PendingChangeUnknown

    _clear_cache(get_pending_subscription_change)

    mock_user = MagicMock()
    mock_user.stripe_customer_id = "cus_abc"

    async def mock_price_id(
        tier: SubscriptionTier, billing_cycle: str = "monthly"
    ) -> str | None:
        return None

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.get_subscription_price_id",
            side_effect=mock_price_id,
        ),
        pytest.raises(PendingChangeUnknown),
    ):
        await get_pending_subscription_change("user-price-fail")


@pytest.mark.asyncio
async def test_release_pending_subscription_schedule_invalidates_cache_on_partial_failure():
    """If schedule.release succeeds but cancel_at_period_end clear fails, the
    cache must still be invalidated — otherwise the UI shows a stale pending
    banner for up to 30s even though the schedule was actually released."""
    _clear_cache(get_pending_subscription_change)

    mock_user = MagicMock()
    mock_user.stripe_customer_id = "cus_abc"

    import time as time_mod

    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_mixed",
            "schedule": "sub_sched_to_release",
            "cancel_at_period_end": True,
            "current_period_end": int(time_mod.time()) + 10 * 24 * 3600,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    with (
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.release_async",
            new_callable=AsyncMock,
            return_value=MagicMock(),
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
            side_effect=stripe.APIConnectionError("transient Stripe error"),
        ),
        patch.object(
            get_pending_subscription_change, "cache_delete"
        ) as mock_cache_delete,
    ):
        with pytest.raises(stripe.APIConnectionError):
            await release_pending_subscription_schedule("user-partial")

        mock_cache_delete.assert_called_once_with("user-partial")


# ─── TOP_UP vs GRANT routing ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_grant_credits_writes_grant_not_top_up():
    """grant_credits writes a GRANT row and never touches Stripe — TOP_UP is
    reserved for actual user-initiated Stripe checkouts."""
    from prisma.enums import CreditTransactionType

    from backend.data.credit import UserCredit
    from backend.util.json import SafeJson

    credit_system = UserCredit()
    add_tx_mock = AsyncMock(return_value=(1500, "grant-txkey"))
    with patch.object(credit_system, "_add_transaction", add_tx_mock):
        balance = await credit_system.grant_credits(
            "user-1", 500, "Refund for failed CoPilot rate-limit reset"
        )

    assert balance == 1500
    add_tx_mock.assert_awaited_once()
    kwargs = add_tx_mock.await_args.kwargs
    assert kwargs["transaction_type"] == CreditTransactionType.GRANT
    metadata = kwargs["metadata"]
    assert isinstance(metadata, SafeJson)
    assert metadata.data["reason"] == "Refund for failed CoPilot rate-limit reset"


@pytest.mark.asyncio
async def test_grant_credits_rejects_negative_amount():
    """grant_credits only adds credits — negative amounts must raise."""
    from backend.data.credit import UserCredit

    credit_system = UserCredit()
    with pytest.raises(ValueError, match="must not be negative"):
        await credit_system.grant_credits("user-1", -100, "bug")


@pytest.mark.asyncio
async def test_admin_get_user_history_excludes_inactive_by_default():
    """Default call must filter out inactive ledger rows so phantom TOP_UP entries
    from abandoned Stripe checkouts don't pollute the admin dashboard."""
    from backend.data.credit import admin_get_user_history

    prisma_mock = MagicMock(
        find_many=AsyncMock(return_value=[]),
        count=AsyncMock(return_value=0),
    )
    with patch(
        "backend.data.credit.CreditTransaction.prisma", return_value=prisma_mock
    ):
        await admin_get_user_history()

    where = prisma_mock.find_many.await_args.kwargs["where"]
    assert where == {"isActive": True}
    count_where = prisma_mock.count.await_args.kwargs["where"]
    assert count_where == {"isActive": True}


@pytest.mark.asyncio
async def test_admin_get_user_history_include_inactive_omits_filter():
    """include_inactive=True surfaces phantom rows for debugging abandoned checkouts."""
    from prisma.enums import CreditTransactionType

    from backend.data.credit import admin_get_user_history

    prisma_mock = MagicMock(
        find_many=AsyncMock(return_value=[]),
        count=AsyncMock(return_value=0),
    )
    with patch(
        "backend.data.credit.CreditTransaction.prisma", return_value=prisma_mock
    ):
        await admin_get_user_history(
            transaction_filter=CreditTransactionType.TOP_UP,
            include_inactive=True,
        )

    where = prisma_mock.find_many.await_args.kwargs["where"]
    assert "isActive" not in where
    assert where["type"] == CreditTransactionType.TOP_UP


@pytest.mark.asyncio
async def test_modify_stripe_subscription_same_tier_yearly_to_monthly_schedules_downgrade():
    """Same-tier yearly→monthly is a cycle downgrade: defer to period end via
    Subscription Schedule, NOT immediate always_invoice modify.

    Without this branch, the dialog promise ("no charge today, switch at end of
    yearly period") is broken — Stripe would change the price + draft a $0
    prorated invoice immediately, mid-period."""
    import time as time_mod

    now = int(time_mod.time())
    period_end = now + 200 * 24 * 3600
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro_yearly",
            "items": {
                "data": [{"id": "si_pro_y", "price": {"id": "price_pro_yearly"}}]
            },
            "current_period_start": now - 165 * 24 * 3600,
            "current_period_end": period_end,
            "schedule": None,
            "cancel_at_period_end": False,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.PRO

    mock_schedule = stripe.SubscriptionSchedule.construct_from(
        {"id": "sub_sched_cycle"}, "k"
    )

    async def price_lookup(tier: SubscriptionTier, billing_cycle: str = "monthly"):
        if tier == SubscriptionTier.PRO and billing_cycle == "monthly":
            return "price_pro_monthly"
        if tier == SubscriptionTier.PRO and billing_cycle == "yearly":
            return "price_pro_yearly"
        if tier == SubscriptionTier.MAX and billing_cycle == "monthly":
            return "price_max_monthly"
        if tier == SubscriptionTier.MAX and billing_cycle == "yearly":
            return "price_max_yearly"
        return None

    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new=AsyncMock(side_effect=price_lookup),
        ),
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.create_async",
            new_callable=AsyncMock,
            return_value=mock_schedule,
        ) as mock_schedule_create,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.modify_async",
            new_callable=AsyncMock,
        ) as mock_schedule_modify,
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.PRO, billing_cycle="monthly"
        )

    assert result is True
    # Did NOT bill immediately — no Subscription.modify with always_invoice.
    mock_modify.assert_not_called()
    # Created a schedule that runs the current yearly price to period end, then
    # flips to the monthly price afterwards.
    mock_schedule_create.assert_called_once_with(from_subscription="sub_pro_yearly")
    assert mock_schedule_modify.call_count == 1
    _, kwargs = mock_schedule_modify.call_args
    phases = kwargs["phases"]
    assert phases[0]["items"][0]["price"] == "price_pro_yearly"
    assert phases[0]["end_date"] == period_end
    assert phases[1]["items"][0]["price"] == "price_pro_monthly"
    assert phases[0]["proration_behavior"] == "none"
    assert phases[1]["proration_behavior"] == "none"


@pytest.mark.asyncio
async def test_modify_stripe_subscription_same_tier_monthly_to_yearly_immediate():
    """Same-tier monthly→yearly is a cycle *upgrade* (more commitment): keep
    the immediate proration semantic so the user is billed for the longer
    cycle today rather than a free upgrade until period end."""
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro_monthly",
            "items": {
                "data": [{"id": "si_pro_m", "price": {"id": "price_pro_monthly"}}]
            },
            "schedule": None,
            "cancel_at_period_end": False,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.PRO

    async def price_lookup(tier: SubscriptionTier, billing_cycle: str = "monthly"):
        if tier == SubscriptionTier.PRO and billing_cycle == "monthly":
            return "price_pro_monthly"
        if tier == SubscriptionTier.PRO and billing_cycle == "yearly":
            return "price_pro_yearly"
        if tier == SubscriptionTier.MAX and billing_cycle == "monthly":
            return "price_max_monthly"
        if tier == SubscriptionTier.MAX and billing_cycle == "yearly":
            return "price_max_yearly"
        return None

    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new=AsyncMock(side_effect=price_lookup),
        ),
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.create_async",
            new_callable=AsyncMock,
        ) as mock_schedule_create,
        patch(
            "backend.data.credit.set_subscription_tier",
            new_callable=AsyncMock,
        ),
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.PRO, billing_cycle="yearly"
        )

    assert result is True
    mock_modify.assert_called_once_with(
        "sub_pro_monthly",
        items=[{"id": "si_pro_m", "price": "price_pro_yearly"}],
        proration_behavior="always_invoice",
        payment_behavior="error_if_incomplete",
        metadata={
            "user_id": "user-1",
            "tier": SubscriptionTier.PRO.value,
            "billing_cycle": "yearly",
        },
    )
    mock_schedule_create.assert_not_called()


@pytest.mark.asyncio
async def test_modify_stripe_subscription_tier_upgrade_yearly_still_immediate():
    """Regression: tier upgrade (PRO→MAX) on yearly stays on the immediate
    always_invoice path — same-tier-cycle-downgrade detection must not
    accidentally route tier upgrades to the period-end schedule."""
    mock_sub = stripe.Subscription.construct_from(
        {
            "id": "sub_pro_yearly",
            "items": {
                "data": [{"id": "si_pro_y", "price": {"id": "price_pro_yearly"}}]
            },
            "schedule": None,
            "cancel_at_period_end": False,
        },
        "k",
    )
    mock_list = MagicMock()
    mock_list.data = [mock_sub]

    mock_user = MagicMock(spec=User)
    mock_user.stripe_customer_id = "cus_abc"
    mock_user.subscription_tier = SubscriptionTier.PRO

    with (
        patch(
            "backend.data.credit.get_subscription_price_id",
            new_callable=AsyncMock,
            return_value="price_max_yearly",
        ),
        patch(
            "backend.data.credit.get_user_by_id",
            new_callable=AsyncMock,
            return_value=mock_user,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=mock_list,
        ),
        patch(
            "backend.data.credit.stripe.Subscription.modify_async",
            new_callable=AsyncMock,
        ) as mock_modify,
        patch(
            "backend.data.credit.stripe.SubscriptionSchedule.create_async",
            new_callable=AsyncMock,
        ) as mock_schedule_create,
        patch(
            "backend.data.credit.set_subscription_tier",
            new_callable=AsyncMock,
        ),
    ):
        result = await modify_stripe_subscription_for_tier(
            "user-1", SubscriptionTier.MAX, billing_cycle="yearly"
        )

    assert result is True
    mock_modify.assert_called_once()
    _, kwargs = mock_modify.call_args
    assert kwargs["proration_behavior"] == "always_invoice"
    mock_schedule_create.assert_not_called()
