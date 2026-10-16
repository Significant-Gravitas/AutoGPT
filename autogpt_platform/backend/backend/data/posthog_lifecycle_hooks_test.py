"""Every write that can change a user's lifecycle snapshot schedules a sync,
and a PostHog or Stripe failure in that sync never reaches the caller."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier

from backend.copilot.rate_limit import set_user_tier
from backend.data import posthog_lifecycle_sync as lifecycle
from backend.data import user as user_module
from backend.data.credit import set_subscription_tier, sync_subscription_from_stripe
from backend.data.stripe_reconciliation import _collect_status_page

TRIAL_SUB = {
    "id": "sub_trial",
    "customer": "cus_1",
    "status": "trialing",
    "metadata": {"user_id": "user-1", "trial_enrollment_id": "trial-1"},
}


async def test_subscription_sync_schedules_by_customer():
    with (
        patch(
            "backend.data.credit._sync_subscription_tier_from_stripe",
            new_callable=AsyncMock,
        ) as inner,
        patch("backend.data.credit.schedule_posthog_lifecycle_sync") as schedule,
    ):
        await sync_subscription_from_stripe(TRIAL_SUB)

    inner.assert_awaited_once_with(TRIAL_SUB)
    schedule.assert_called_once_with(stripe_customer_id="cus_1")


async def test_a_tier_change_honours_track_lifecycle_false():
    """The tier sweep's opt-out must reach set_subscription_tier too, or a
    tier change inside the sweep schedules a sync anyway."""
    update = AsyncMock()
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(update=update),
        ),
        patch("backend.data.credit.invalidate_subscription_caches"),
        patch("backend.data.credit.schedule_posthog_lifecycle_sync") as schedule,
    ):
        await set_subscription_tier(
            "user-1", SubscriptionTier.PRO, track_lifecycle=False
        )
        schedule.assert_not_called()
        await set_subscription_tier("user-1", SubscriptionTier.PRO)
        schedule.assert_called_once_with("user-1")
    assert update.await_count == 2


async def test_tier_sweep_does_not_schedule_lifecycle_syncs():
    """The 6-hourly tier sweep re-syncs every trial; scheduling from there
    would fan out one Stripe call per trial and could rate-limit the sweep's
    own listing. The daily lifecycle sweep covers those users."""
    page = MagicMock(
        data=[stripe.Subscription.construct_from(TRIAL_SUB, "k")], has_more=False
    )
    user = MagicMock(subscriptionTier=SubscriptionTier.TRIAL)
    with (
        patch(
            "backend.data.stripe_reconciliation.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            return_value=page,
        ),
        patch(
            "backend.data.credit._sync_subscription_tier_from_stripe",
            new_callable=AsyncMock,
        ) as inner,
        patch(
            "backend.data.stripe_reconciliation.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=user)),
        ),
        patch("backend.data.credit.schedule_posthog_lifecycle_sync") as schedule,
    ):
        tiers: dict[str, SubscriptionTier] = {}
        incomplete = await _collect_status_page("trialing", {}, tiers)

    assert incomplete is False
    inner.assert_awaited_once()
    schedule.assert_not_called()
    assert tiers == {"cus_1": SubscriptionTier.TRIAL}


async def test_failed_subscription_sync_raises_as_before_and_schedules_nothing():
    with (
        patch(
            "backend.data.credit._sync_subscription_tier_from_stripe",
            new_callable=AsyncMock,
            side_effect=ValueError("no enrollment"),
        ),
        patch("backend.data.credit.schedule_posthog_lifecycle_sync") as schedule,
    ):
        with pytest.raises(ValueError):
            await sync_subscription_from_stripe(TRIAL_SUB)
    schedule.assert_not_called()


async def test_trial_transitions_schedule_through_the_webhook_choke_point():
    """Trial start, cancel, conversion and expiry are all written by
    ``reconcile_trial_subscription``, which only runs inside
    ``sync_subscription_from_stripe``; its early return still schedules."""
    user = MagicMock(id="user-1", subscriptionTier=SubscriptionTier.NO_TIER)
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(find_first=AsyncMock(return_value=user)),
        ),
        patch(
            "backend.data.credit.reconcile_trial_subscription",
            new_callable=AsyncMock,
            return_value=(TRIAL_SUB, SubscriptionTier.TRIAL),
        ) as reconcile,
        patch("backend.data.credit.invalidate_subscription_caches"),
        patch("backend.data.credit.schedule_posthog_lifecycle_sync") as schedule,
    ):
        await sync_subscription_from_stripe(TRIAL_SUB)

    reconcile.assert_awaited_once_with("user-1", "sub_trial")
    schedule.assert_called_once_with(stripe_customer_id="cus_1")


async def test_webhook_survives_posthog_and_stripe_failures():
    """The real scheduler, with the sync's Stripe read and PostHog send both
    failing: the webhook returns normally and the failure stays in the
    background task."""
    client = MagicMock()
    client.capture.side_effect = RuntimeError("posthog down")
    user_row = MagicMock(
        id="user-1",
        createdAt=None,
        stripeCustomerId="cus_1",
        subscriptionTier=SubscriptionTier.PRO,
        subscriptionTrial=None,
    )
    lifecycle_prisma = MagicMock(
        find_first=AsyncMock(return_value=user_row),
        find_unique=AsyncMock(return_value=user_row),
    )
    lifecycle._syncs.clear()
    with (
        patch(
            "backend.data.credit._sync_subscription_tier_from_stripe",
            new_callable=AsyncMock,
        ),
        patch(
            "backend.data.posthog_lifecycle_sync.get_posthog_client",
            return_value=client,
        ),
        patch(
            "backend.data.posthog_lifecycle_sync.User.prisma",
            return_value=lifecycle_prisma,
        ),
        patch.object(
            stripe.Subscription,
            "list_async",
            AsyncMock(side_effect=stripe.APIConnectionError("stripe down")),
        ),
    ):
        await sync_subscription_from_stripe(TRIAL_SUB)
        while lifecycle._background_tasks:
            await asyncio.gather(*list(lifecycle._background_tasks))
            await asyncio.sleep(0)

    client.capture.assert_not_called()


async def test_tier_change_schedules_the_user():
    with (
        patch(
            "backend.data.credit.User.prisma",
            return_value=MagicMock(update=AsyncMock()),
        ),
        patch("backend.data.credit.invalidate_subscription_caches"),
        patch("backend.data.credit.schedule_posthog_lifecycle_sync") as schedule,
    ):
        await set_subscription_tier("user-1", SubscriptionTier.PRO)
    schedule.assert_called_once_with("user-1")


async def test_admin_tier_override_schedules_the_user():
    with (
        patch(
            "backend.copilot.rate_limit.PrismaUser.prisma",
            return_value=MagicMock(update=AsyncMock()),
        ),
        patch("backend.copilot.rate_limit._drift_check_background", new=AsyncMock()),
        patch("backend.copilot.rate_limit.get_user_tier", new=MagicMock()),
        patch("backend.copilot.rate_limit.get_user_by_id", new=MagicMock()),
        patch("backend.data.credit.get_pending_subscription_change", new=MagicMock()),
        patch("backend.copilot.rate_limit.schedule_posthog_lifecycle_sync") as schedule,
    ):
        await set_user_tier("user-1", SubscriptionTier.ENTERPRISE)
    schedule.assert_called_once_with("user-1")


@pytest.mark.parametrize("existing", [False, True])
async def test_signup_schedules_only_a_new_user(existing: bool):
    db_user = MagicMock(id="user-1", email="a@example.com", name=None)
    app_user = MagicMock(id="user-1")
    with (
        patch.object(user_module, "prisma") as prisma,
        patch.object(user_module, "_ensure_user_profile", new_callable=AsyncMock),
        patch.object(
            user_module,
            "ensure_personal_org",
            new_callable=AsyncMock,
            return_value=False,
        ),
        patch.object(user_module.User, "from_db", return_value=app_user),
        patch.object(user_module, "UserCreationResult") as result,
        patch.object(user_module, "schedule_posthog_lifecycle_sync") as schedule,
    ):
        prisma.user.find_unique = AsyncMock(return_value=db_user if existing else None)
        prisma.user.create = AsyncMock(return_value=db_user)
        await user_module.get_or_create_user_with_status(
            {"sub": "user-1", "email": "a@example.com"}
        )

    result.assert_called_once()
    if existing:
        schedule.assert_not_called()
    else:
        schedule.assert_called_once_with("user-1")
