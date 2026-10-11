"""The billing page's subscription lookup stays fresh across API workers.

Each "worker" is its own copy of ``backend.data.credit``, with its own cache
state, the way separate uvicorn processes or pods hold theirs; both share the
real Redis of the test stack and one fake Stripe. A change made through one
worker must be what the other reads next.
"""

import asyncio
import importlib.util
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
import stripe
from prisma.enums import SubscriptionTier

from backend.data import credit as worker_a
from backend.data import subscription_trial_conversion as conversion
from backend.data.subscription_trial import TrialState

pytest_plugins = ("backend.data.subscription_trial_fixtures",)

TRIAL_END = int((datetime.now(UTC) + timedelta(days=7)).timestamp())
PAID_PERIOD_END = int((datetime.now(UTC) + timedelta(days=31)).timestamp())


def _load_worker() -> ModuleType:
    """A second, independently loaded copy of the credit module."""
    spec = importlib.util.find_spec("backend.data.credit")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeStripe:
    """One customer's subscription, as Stripe would answer for it."""

    def __init__(self, subscription: dict):
        self.subscription = subscription
        self.frozen: stripe.Subscription | None = None
        self.release: asyncio.Event | None = None
        self.reached = asyncio.Event()

    def hold_reads(self) -> asyncio.Event:
        """Answer lists from the subscription as it is now, once released:
        a slow answer that arrives after a change made meanwhile."""
        self.frozen = self.current()
        self.release = asyncio.Event()
        return self.release

    def stop_holding(self) -> None:
        self.frozen = None
        self.release = None

    def current(self) -> stripe.Subscription:
        return stripe.Subscription.construct_from(self.subscription, "test-key")

    async def list_async(self, *, customer: str, status: str, limit: int):
        answer = self.frozen or self.current()
        if self.release is not None:
            self.reached.set()
            await self.release.wait()
        matches = (
            [answer]
            if customer == answer["customer"] and status == answer["status"]
            else []
        )
        return stripe.ListObject.construct_from(
            {"data": matches, "has_more": False}, "test-key"
        )

    async def retrieve_async(self, subscription_id: str):
        return self.current()

    async def modify_async(self, subscription_id: str, **changes):
        if changes.get("trial_end") == "now":
            self.subscription = {
                **self.subscription,
                "status": "active",
                "cancel_at_period_end": False,
                "current_period_start": int(datetime.now(UTC).timestamp()),
                "current_period_end": PAID_PERIOD_END,
            }
        return self.current()


@pytest.fixture
def trial(trial: TrialState) -> TrialState:
    """A cancel-pending trial on a customer no other test touches."""
    now = datetime.now(UTC)
    return trial.model_copy(
        update={
            "customer_id": f"cus_{uuid4().hex}",
            "subscription_id": "sub_1",
            "status": "trialing",
            "card_verified_at": now - timedelta(days=2),
            "started_at": now - timedelta(days=2),
            "ends_at": datetime.fromtimestamp(TRIAL_END, UTC),
            "consumed_at": now - timedelta(days=2),
            "cancel_at_period_end": True,
        }
    )


@pytest.fixture
def fake_stripe(trial: TrialState, subscription: dict) -> FakeStripe:
    return FakeStripe(
        {
            **subscription,
            "customer": trial.customer_id,
            "cancel_at_period_end": True,
            "trial_end": TRIAL_END,
            "current_period_start": TRIAL_END - 7 * 24 * 60 * 60,
            "current_period_end": TRIAL_END,
        }
    )


@pytest_asyncio.fixture
async def worker_b(trial: TrialState, fake_stripe: FakeStripe):
    user = MagicMock(stripe_customer_id=trial.customer_id)
    lookup_user = AsyncMock(return_value=user)
    worker = _load_worker()

    @asynccontextmanager
    async def lock(user_id: str):
        yield

    with (
        patch.object(stripe.Subscription, "list_async", fake_stripe.list_async),
        patch.object(stripe.Subscription, "retrieve_async", fake_stripe.retrieve_async),
        patch.object(stripe.Subscription, "modify_async", fake_stripe.modify_async),
        patch.object(worker_a, "get_user_by_id", lookup_user),
        patch.object(worker, "get_user_by_id", lookup_user),
        # The tier write and the analytics sync are not what's under test.
        patch.object(worker_a, "_sync_subscription_tier_from_stripe", AsyncMock()),
        patch.object(worker_a, "schedule_posthog_lifecycle_sync", MagicMock()),
        patch.object(conversion, "subscription_checkout_lock", lock),
        patch.object(
            conversion,
            "subscription_card_can_be_charged",
            AsyncMock(return_value=True),
        ),
        patch.object(conversion, "expire_other_subscription_checkouts", AsyncMock()),
        patch.object(conversion, "other_plan_is_live", AsyncMock(return_value=False)),
    ):
        yield worker


@pytest.mark.asyncio
async def test_a_plan_bought_on_one_worker_shows_its_renewal_on_another(
    trial, worker_b
):
    """Trial → cancel → "Subscribe to Pro": the status read that follows the
    purchase lands on a worker that read the trial moments before."""
    assert await worker_b.get_active_subscription_period_end("user-1") == TRIAL_END

    await conversion.convert_cancel_pending_trial(trial, SubscriptionTier.PRO)

    assert (
        await worker_b.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )


@pytest.mark.asyncio
async def test_a_webhook_on_one_worker_refreshes_the_lookup_on_another(
    trial, fake_stripe, worker_b
):
    """A change made in Stripe (the dashboard, a renewal) reaches us only as a
    webhook, on whichever worker Stripe's request lands."""
    assert await worker_b.get_active_subscription_period_end("user-1") == TRIAL_END

    converted = await fake_stripe.modify_async("sub_1", trial_end="now")
    await worker_a.sync_subscription_from_stripe(dict(converted))

    assert (
        await worker_b.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )


@pytest.mark.asyncio
async def test_a_read_in_flight_during_the_purchase_does_not_cache_the_trial(
    trial, fake_stripe, worker_b
):
    """A page load whose Stripe answer predates the purchase, but arrives after
    it, must not hold the trial's renewal date for the next reads."""
    release = fake_stripe.hold_reads()
    in_flight = asyncio.create_task(
        worker_b.get_active_subscription_period_end("user-1")
    )
    await fake_stripe.reached.wait()

    await conversion.convert_cancel_pending_trial(trial, SubscriptionTier.PRO)
    release.set()
    assert await in_flight == TRIAL_END
    fake_stripe.stop_holding()

    assert (
        await worker_b.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )
