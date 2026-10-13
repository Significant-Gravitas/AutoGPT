"""The billing page's pending subscription change stays fresh across API
workers.

Each "worker" is its own copy of ``backend.data.credit`` and of the pending
change cache, so it holds its own process-local state; both share the real
Redis of the test stack and one fake Stripe.
"""

import asyncio
import importlib.util
from datetime import UTC, datetime, timedelta
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
import stripe
from prisma.enums import SubscriptionTier

from backend.data import credit as worker_a
from backend.data import versioned_cache
from backend.data.redis_client import get_redis_async
from backend.data.subscription_pending_cache import PENDING_CHANGES
from backend.util.exceptions import InsufficientBalanceError

PERIOD_END = int((datetime.now(UTC) + timedelta(days=20)).timestamp())
WAIT = 5  # seconds; no wait in these tests may hang the suite


def _load_copy(name: str) -> ModuleType:
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_worker() -> ModuleType:
    """A second, independently loaded credit module with its own cache."""
    cache = _load_copy("backend.data.subscription_pending_cache")
    cache.PENDING_CHANGES.prefix = PENDING_CHANGES.prefix
    worker = _load_copy("backend.data.credit")
    worker.PENDING_CHANGES = cache.PENDING_CHANGES
    worker.invalidate_pending_subscription_change = (
        cache.invalidate_pending_subscription_change
    )
    return worker


class FakeStripe:
    """One customer's paid subscription, as Stripe would answer for it."""

    def __init__(self, customer_id: str) -> None:
        self.subscription = {
            "id": "sub_pro",
            "object": "subscription",
            "customer": customer_id,
            "status": "active",
            "cancel_at_period_end": False,
            "schedule": None,
            "metadata": {},
            "current_period_start": PERIOD_END - 30 * 24 * 60 * 60,
            "current_period_end": PERIOD_END,
            "items": {
                "object": "list",
                "data": [{"id": "si_1", "price": {"id": "price_PRO_monthly"}}],
                "has_more": False,
            },
        }
        self.lookups = 0
        self.reached = asyncio.Event()

    def current(self) -> stripe.Subscription:
        return stripe.Subscription.construct_from(self.subscription, "test-key")

    async def list_async(self, *, customer: str, status: str, limit: int):
        if status == "active":
            self.lookups += 1
        answer = self.current()
        matches = [answer] if status == answer["status"] else []
        return stripe.ListObject.construct_from(
            {"object": "list", "data": matches, "has_more": False}, "test-key"
        )

    async def modify_async(self, subscription_id: str, **changes):
        self.subscription = {**self.subscription, **changes}
        return self.current()

    async def cancel_async(self, subscription_id: str, **params):
        self.subscription = {**self.subscription, "status": "canceled"}
        return self.current()

    def hold_next_lookup(
        self, monkeypatch: pytest.MonkeyPatch, worker: ModuleType
    ) -> asyncio.Event:
        """Answer ``worker``'s next subscription lookup as it is now, once
        released: a slow answer that arrives after a change made meanwhile."""
        before = self.current()
        release = asyncio.Event()
        ask_stripe = worker._get_active_subscription
        held = False

        async def lookup(customer_id: str):
            nonlocal held
            if held:
                return await ask_stripe(customer_id)
            held = True
            self.reached.set()
            await asyncio.wait_for(release.wait(), WAIT)
            return before

        monkeypatch.setattr(worker, "_get_active_subscription", lookup)
        return release


async def _price_id(tier: SubscriptionTier, cycle: str = "monthly") -> str:
    return f"price_{tier.value}_{cycle}"


@pytest.fixture
def user_id() -> str:
    return f"user-{uuid4().hex}"


@pytest.fixture
def fake_stripe() -> FakeStripe:
    return FakeStripe(f"cus_{uuid4().hex}")


@pytest_asyncio.fixture
async def worker_b(user_id: str, fake_stripe: FakeStripe, monkeypatch):
    # These tests assert what is cached, not how fast: a busy CI runner can
    # take longer than the production deadline for one Redis call.
    monkeypatch.setattr(versioned_cache, "_REDIS_DEADLINE_SECONDS", WAIT)
    redis = await asyncio.wait_for(get_redis_async(), WAIT)
    assert await asyncio.wait_for(redis.ping(), WAIT), "these tests need Redis"
    user = MagicMock(
        id=user_id,
        stripe_customer_id=fake_stripe.subscription["customer"],
        subscription_tier=SubscriptionTier.PRO,
    )
    lookup_user = AsyncMock(return_value=user)
    worker = _load_worker()
    with (
        patch.object(stripe.Subscription, "list_async", fake_stripe.list_async),
        patch.object(stripe.Subscription, "modify_async", fake_stripe.modify_async),
        patch.object(stripe.Subscription, "cancel_async", fake_stripe.cancel_async),
        patch.object(worker_a, "get_user_by_id", lookup_user),
        patch.object(worker, "get_user_by_id", lookup_user),
        patch.object(worker_a, "get_subscription_price_id", _price_id),
        patch.object(worker, "get_subscription_price_id", _price_id),
        patch.object(worker_a, "_track_billing_event", MagicMock()),
        # The tier write and the analytics sync are not what's under test.
        patch.object(worker_a, "_sync_subscription_tier_from_stripe", AsyncMock()),
        patch.object(worker_a, "schedule_posthog_lifecycle_sync", MagicMock()),
    ):
        yield worker


@pytest.mark.asyncio
async def test_a_read_in_flight_during_a_cancellation_does_not_hide_it(
    user_id, fake_stripe, worker_b, monkeypatch
):
    """A page load on one worker reads the plan before the user cancels on
    another; its slow answer must not wipe out the "cancels on" badge."""
    release = fake_stripe.hold_next_lookup(monkeypatch, worker_b)
    in_flight = asyncio.create_task(worker_b.get_pending_subscription_change(user_id))
    await asyncio.wait_for(fake_stripe.reached.wait(), WAIT)

    assert await worker_a.cancel_stripe_subscription(user_id) is True
    cancelling = await worker_a.get_pending_subscription_change(user_id)
    assert cancelling is not None and cancelling[0] == SubscriptionTier.NO_TIER

    release.set()
    assert await asyncio.wait_for(in_flight, WAIT) is None

    for worker in (worker_a, worker_b):
        pending = await worker.get_pending_subscription_change(user_id)
        assert pending is not None
        assert pending[0] == SubscriptionTier.NO_TIER
        assert int(pending[1].timestamp()) == PERIOD_END


@pytest.mark.asyncio
async def test_a_cancellation_made_in_stripe_shows_on_every_worker(
    user_id, fake_stripe, worker_b
):
    """A cancellation from Stripe's billing portal reaches us only as a
    webhook, and leaves the tier as it is until the period ends."""
    assert await worker_b.get_pending_subscription_change(user_id) is None

    cancelled = await fake_stripe.modify_async("sub_pro", cancel_at_period_end=True)
    await worker_a.sync_subscription_from_stripe(dict(cancelled))

    pending = await worker_b.get_pending_subscription_change(user_id)
    assert pending is not None and pending[0] == SubscriptionTier.NO_TIER
    assert fake_stripe.lookups == 2


@pytest.mark.asyncio
async def test_a_renewal_the_balance_cant_cover_clears_the_pending_change(
    user_id, fake_stripe, worker_b
):
    """A cancellation is scheduled, then the renewal fails and the balance
    can't cover it, so we end the plan in Stripe now: the "cancels on" badge
    must go on every worker, not linger until it expires."""
    fake_stripe.subscription["cancel_at_period_end"] = True
    pending = await worker_b.get_pending_subscription_change(user_id)
    assert pending is not None and pending[0] == SubscriptionTier.NO_TIER

    users = MagicMock()
    users.prisma.return_value.find_first = AsyncMock(
        return_value=MagicMock(id=user_id, subscriptionTier=SubscriptionTier.PRO)
    )
    balance = MagicMock()
    balance._add_transaction = AsyncMock(
        side_effect=InsufficientBalanceError("no balance", user_id, 0, 5000)
    )
    renewal = {
        "id": "in_renewal",
        "customer": fake_stripe.subscription["customer"],
        "amount_due": 5000,
        "billing_reason": "subscription_cycle",
        "subscription": "sub_pro",
    }
    with (
        patch.object(worker_a, "User", users),
        patch.object(worker_a, "UserCredit", MagicMock(return_value=balance)),
        patch.object(worker_a, "set_subscription_tier", AsyncMock()) as set_tier,
    ):
        await worker_a.handle_subscription_payment_failure(renewal)

    assert fake_stripe.subscription["status"] == "canceled"
    set_tier.assert_awaited_once_with(user_id, SubscriptionTier.NO_TIER)
    assert await worker_b.get_pending_subscription_change(user_id) is None
