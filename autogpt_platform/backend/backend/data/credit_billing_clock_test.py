"""Where the billing clock is copied from Stripe, and where it must not be:
the tier sync every subscription event passes through, and an upgrade's own
answer. Its SQL is tested against Postgres in usage_clock_integration_test.py."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import stripe
from prisma.enums import SubscriptionTier
from prisma.models import User

from backend.data import credit
from backend.data.credit import (
    modify_stripe_subscription_for_tier,
    sync_subscription_from_stripe,
)

START = 1_793_824_256
PRICES = {"price_pro": SubscriptionTier.PRO, "price_max": SubscriptionTier.MAX}


def _subscription(status: str = "active", **fields) -> dict:
    return {
        "id": "sub_1",
        "customer": "cus_123",
        "status": status,
        "current_period_start": START,
        "current_period_end": START + 30 * 86_400,
        "billing_cycle_anchor": START,
        "items": {"data": [{"price": {"id": "price_pro"}, "quantity": 1}]},
        **fields,
    }


def _user(tier: SubscriptionTier = SubscriptionTier.PRO) -> MagicMock:
    user = MagicMock(spec=User)
    user.id = "user-1"
    user.stripeCustomerId = "cus_123"
    user.subscriptionTier = tier
    return user


class Sync:
    """The tier sync with Stripe and the database stubbed out."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.user = _user()
        self.subscription = _subscription()
        self.save_clock = AsyncMock(return_value=True)
        self.set_tier = AsyncMock()
        self.settled = AsyncMock(return_value=True)
        self.tx = MagicMock(user=MagicMock(update=AsyncMock()))
        context = MagicMock()
        context.__aenter__ = AsyncMock(return_value=self.tx)
        context.__aexit__ = AsyncMock(return_value=False)
        no_subscriptions = MagicMock(data=[], has_more=False)
        patches = {
            "save_billing_clock": self.save_clock,
            "set_subscription_tier": self.set_tier,
            "get_subscription_trial": AsyncMock(return_value=None),
            "build_price_to_tier_map": AsyncMock(return_value=PRICES),
            "replaced_by_newer_plan": AsyncMock(return_value=False),
            "invalidate_billing_lookups": AsyncMock(),
            "schedule_posthog_lifecycle_sync": MagicMock(),
            "_cleanup_stale_subscriptions": AsyncMock(),
            "transaction": MagicMock(return_value=context),
        }
        for name, stub in patches.items():
            monkeypatch.setattr(credit, name, stub)
        monkeypatch.setattr(
            credit.User,
            "prisma",
            lambda: MagicMock(find_first=AsyncMock(side_effect=lambda **_: self.user)),
        )
        monkeypatch.setattr(
            credit.subscription_activation,
            "lock_activation_user",
            AsyncMock(side_effect=lambda *_: self.user),
        )
        monkeypatch.setattr(
            credit.subscription_activation,
            "publish_initial_pro_activation",
            self.settled,
        )
        monkeypatch.setattr(
            stripe.Subscription,
            "retrieve_async",
            AsyncMock(side_effect=lambda *_: self.subscription),
        )
        monkeypatch.setattr(
            stripe.Subscription, "list_async", AsyncMock(return_value=no_subscriptions)
        )

    async def run(self) -> None:
        await sync_subscription_from_stripe(self.subscription)


@pytest.fixture
def sync(monkeypatch: pytest.MonkeyPatch) -> Sync:
    return Sync(monkeypatch)


@pytest.mark.asyncio
async def test_a_live_subscriptions_period_is_copied(sync: Sync):
    await sync.run()

    sync.save_clock.assert_awaited_once_with("user-1", sync.subscription)


@pytest.mark.asyncio
async def test_a_renewal_is_copied_before_its_invoice_is_paid(sync: Sync):
    """Stripe charges a renewal about an hour after the period starts. The
    new month's allowance starts with the period: the unpaid draft invoice
    holds back the activation step, not the clock."""
    sync.settled.return_value = False

    await sync.run()

    sync.save_clock.assert_awaited_once()
    sync.tx.user.update.assert_not_awaited()  # and the tier is left as it was


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", ["past_due", "unpaid", "canceled", "incomplete", "paused"]
)
async def test_a_subscription_that_is_not_live_never_moves_the_clock(
    sync: Sync, status: str
):
    """A failed renewal's new period gives no new allowance: its plan is gone
    until the invoice is paid, and paying returns it in the same period."""
    sync.subscription = _subscription(status)

    await sync.run()

    sync.save_clock.assert_not_awaited()
    sync.set_tier.assert_awaited_once_with(
        "user-1", SubscriptionTier.NO_TIER, track_lifecycle=False
    )


@pytest.mark.asyncio
async def test_an_admin_managed_account_gets_no_stripe_clock(sync: Sync):
    sync.user = _user(SubscriptionTier.ENTERPRISE)

    await sync.run()

    sync.save_clock.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_old_plan_revived_beside_its_replacement_is_not_copied(
    sync: Sync, monkeypatch: pytest.MonkeyPatch
):
    credit.replaced_by_newer_plan.return_value = True
    monkeypatch.setattr(
        "backend.data.subscription_payment_failure.end_revived_plan", AsyncMock()
    )

    await sync.run()

    sync.save_clock.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_subscription_for_another_user_is_not_copied(sync: Sync):
    sync.subscription = _subscription(metadata={"user_id": "someone-else"})

    await sync.run()

    sync.save_clock.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_clock_that_cannot_be_saved_does_not_stop_the_tier(sync: Sync):
    sync.user = _user(SubscriptionTier.NO_TIER)
    sync.save_clock.side_effect = RuntimeError("database down")

    await sync.run()

    sync.tx.user.update.assert_awaited_once_with(
        where={"id": "user-1"},
        data={"subscriptionTier": SubscriptionTier.PRO, "subscriptionEndedAt": None},
    )


class Upgrade:
    """A Pro subscriber's upgrade to Max, with Stripe stubbed out."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[str] = []
        self.live = True
        self.subscription = {
            "id": "sub_pro",
            "schedule": None,
            "cancel_at_period_end": False,
            "items": {"data": [{"id": "si_pro", "price": {"id": "price_pro"}}]},
        }
        self.upgraded = _subscription(id="sub_pro", current_period_start=START + 500)
        self.save_clock = AsyncMock(return_value=True)
        self.set_tier = AsyncMock()
        self.modify = AsyncMock(side_effect=self._modify)
        self.release = AsyncMock(side_effect=self._release)
        user = MagicMock(stripe_customer_id="cus_123", subscription_tier="PRO")
        user.subscription_tier = SubscriptionTier.PRO
        patches = {
            "save_billing_clock": self.save_clock,
            "set_subscription_tier": self.set_tier,
            "get_subscription_price_id": AsyncMock(return_value="price_max"),
            "get_user_by_id": AsyncMock(return_value=user),
            "billing_clock_live": AsyncMock(side_effect=lambda: self.live),
            "invalidate_billing_lookups": AsyncMock(),
            "_track_billing_event": MagicMock(),
            "_get_active_subscription": AsyncMock(
                side_effect=lambda _: stripe.Subscription.construct_from(
                    self.subscription, "k"
                )
            ),
        }
        for name, stub in patches.items():
            monkeypatch.setattr(credit, name, stub)
        monkeypatch.setattr(stripe.Subscription, "modify_async", self.modify)
        monkeypatch.setattr(stripe.SubscriptionSchedule, "release_async", self.release)

    async def _modify(self, *_args, **_kwargs):
        self.calls.append("modify")
        return self.upgraded

    async def _release(self, *_args, **_kwargs):
        self.calls.append("release")

    async def run(self) -> bool:
        return await modify_stripe_subscription_for_tier("user-1", SubscriptionTier.MAX)


@pytest.fixture
def upgrade(monkeypatch: pytest.MonkeyPatch) -> Upgrade:
    return Upgrade(monkeypatch)


@pytest.mark.asyncio
async def test_from_launch_an_upgrade_starts_a_fresh_billing_month(upgrade: Upgrade):
    assert await upgrade.run() is True

    upgrade.modify.assert_awaited_once_with(
        "sub_pro",
        items=[{"id": "si_pro", "price": "price_max"}],
        proration_behavior="always_invoice",
        payment_behavior="error_if_incomplete",
        metadata={"user_id": "user-1", "tier": "MAX", "billing_cycle": "monthly"},
        billing_cycle_anchor="now",
    )
    # The new period is in Stripe's answer: the fresh windows don't wait for
    # the webhook.
    upgrade.save_clock.assert_awaited_once_with("user-1", upgrade.upgraded)
    upgrade.set_tier.assert_awaited_once_with("user-1", SubscriptionTier.MAX)
    event = credit._track_billing_event.call_args.args[2]
    assert event["billing_date_reset"] is True


@pytest.mark.asyncio
async def test_before_launch_an_upgrade_keeps_its_billing_period(upgrade: Upgrade):
    upgrade.live = False

    assert await upgrade.run() is True

    assert "billing_cycle_anchor" not in upgrade.modify.await_args.kwargs
    assert credit._track_billing_event.call_args.args[2]["billing_date_reset"] is False


@pytest.mark.asyncio
async def test_a_pending_downgrade_is_released_before_the_billing_date_is_reset(
    upgrade: Upgrade,
):
    """Stripe refuses to change a subscription a schedule still drives."""
    upgrade.subscription["schedule"] = "sub_sched_1"

    assert await upgrade.run() is True

    assert upgrade.calls == ["release", "modify"]
    upgrade.release.assert_awaited_once_with("sub_sched_1")
    assert upgrade.modify.await_args.kwargs["billing_cycle_anchor"] == "now"


@pytest.mark.asyncio
async def test_a_declined_card_changes_neither_the_plan_nor_the_clock(upgrade: Upgrade):
    upgrade.modify.side_effect = stripe.CardError("declined", "card", "card_declined")

    with pytest.raises(stripe.CardError):
        await upgrade.run()

    upgrade.save_clock.assert_not_awaited()
    upgrade.set_tier.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_clock_that_cannot_be_saved_does_not_fail_a_charged_upgrade(
    upgrade: Upgrade,
):
    upgrade.save_clock.side_effect = RuntimeError("database down")

    assert await upgrade.run() is True

    upgrade.set_tier.assert_awaited_once_with("user-1", SubscriptionTier.MAX)


@pytest.mark.asyncio
async def test_a_downgrade_still_waits_for_the_period_end(
    upgrade: Upgrade, monkeypatch: pytest.MonkeyPatch
):
    schedule = AsyncMock()
    monkeypatch.setattr(credit, "_schedule_downgrade_at_period_end", schedule)
    credit.get_user_by_id.return_value.subscription_tier = SubscriptionTier.MAX

    assert await modify_stripe_subscription_for_tier("user-1", SubscriptionTier.PRO)

    schedule.assert_awaited_once()
    upgrade.modify.assert_not_awaited()
    upgrade.save_clock.assert_not_awaited()
