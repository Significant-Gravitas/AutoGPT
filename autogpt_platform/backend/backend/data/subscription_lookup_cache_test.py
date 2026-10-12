"""The billing page's subscription lookup stays fresh across API workers, and
Redis trouble never slows or blocks a billing change.

Each "worker" is its own copy of ``backend.data.credit`` and of the lookup
cache module, so it holds its own process-local state, the way separate
uvicorn processes or pods do; both share the real Redis of the test stack and
one fake Stripe.
"""

import asyncio
import importlib.util
import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
import stripe
from prisma.enums import SubscriptionTier
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisClusterException

from backend.data import credit as worker_a
from backend.data import subscription_lookup_cache as cache_a
from backend.data import subscription_trial_conversion as conversion
from backend.data import versioned_cache
from backend.data.redis_client import get_redis_async
from backend.data.subscription_lookup_unconfirmed import UnconfirmedInvalidations
from backend.data.subscription_trial import TrialState

pytest_plugins = ("backend.data.subscription_trial_fixtures",)

ACTIVE = cache_a.ACTIVE_SUBSCRIPTIONS

TRIAL_END = int((datetime.now(UTC) + timedelta(days=7)).timestamp())
PAID_PERIOD_END = int((datetime.now(UTC) + timedelta(days=31)).timestamp())
WAIT = 5  # seconds; no wait in these tests may hang the suite


def _load_copy(name: str) -> ModuleType:
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_worker() -> ModuleType:
    """A second, independently loaded credit module with its own cache module."""
    cache = _load_copy("backend.data.subscription_lookup_cache")
    cache.ACTIVE_SUBSCRIPTIONS.prefix = ACTIVE.prefix
    worker = _load_copy("backend.data.credit")
    worker.read_active_subscription = cache.read_active_subscription
    worker.invalidate_active_subscription_cache = (
        cache.invalidate_active_subscription_cache
    )
    return worker


class FakeStripe:
    """One customer's subscription, as Stripe would answer for it."""

    def __init__(self, subscription: dict):
        self.subscription = subscription
        self.lookups = 0
        self.reached = asyncio.Event()

    def current(self) -> stripe.Subscription:
        return stripe.Subscription.construct_from(self.subscription, "test-key")

    def hold_next_lookup(
        self, monkeypatch: pytest.MonkeyPatch, worker: ModuleType
    ) -> asyncio.Event:
        """Answer ``worker``'s next lookup with the subscription as it is now,
        once released: a slow answer that arrives after a change made
        meanwhile. Later lookups ask the fake Stripe as usual.

        Held per lookup, not per task: on Python 3.11 ``asyncio.wait_for``
        (inside ``stripe_call``) runs each SDK call in a task of its own."""
        before = self.current()
        release = asyncio.Event()
        ask_stripe = worker._get_active_subscription
        held = False

        async def lookup(customer_id: str):
            nonlocal held
            if held:
                return await ask_stripe(customer_id)
            held = True
            self.lookups += 1
            self.reached.set()
            await asyncio.wait_for(release.wait(), WAIT)
            return before

        monkeypatch.setattr(worker, "_get_active_subscription", lookup)
        return release

    async def list_async(self, *, customer: str, status: str, limit: int):
        if status == "active":
            # _get_active_subscription asks for "active" first on every lookup.
            self.lookups += 1
        # Stripe takes a moment to answer, so concurrent lookups overlap.
        await asyncio.sleep(0.02)
        answer = self.current()
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


@pytest.fixture
def save_tier() -> AsyncMock:
    """Where the paid plan is saved to the user; the DB write isn't under test."""
    return AsyncMock()


@pytest_asyncio.fixture
async def worker_b(trial: TrialState, fake_stripe: FakeStripe, save_tier: AsyncMock):
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
        patch.object(worker_a, "_sync_subscription_tier_from_stripe", save_tier),
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


@pytest_asyncio.fixture
async def redis_up(monkeypatch):
    """These tests prove what the cache does, so a Redis the cache can't use
    must fail them, not let every lookup fall through to Stripe and pass.

    They assert what is cached, not how fast: a busy CI runner can take
    longer than the production deadline for one Redis call, which would
    rightly send that read to Stripe and miscount the lookups."""
    monkeypatch.setattr(versioned_cache, "_REDIS_DEADLINE_SECONDS", WAIT)
    redis = await asyncio.wait_for(get_redis_async(), WAIT)
    assert await asyncio.wait_for(redis.ping(), WAIT)
    return redis


async def _cached_period_end(redis, customer_id: str) -> int | None:
    raw = await asyncio.wait_for(redis.get(ACTIVE._entry_key(customer_id)), WAIT)
    assert raw, "the lookup was not written to Redis"
    subscription = json.loads(raw)["value"]
    assert subscription is not None
    return subscription["current_period_end"]


@pytest.mark.asyncio
async def test_one_workers_lookup_serves_every_worker(
    trial, fake_stripe, worker_b, redis_up
):
    assert await worker_b.get_active_subscription_period_end("user-1") == TRIAL_END
    assert fake_stripe.lookups == 1
    assert await _cached_period_end(redis_up, trial.customer_id) == TRIAL_END

    assert await worker_b.get_active_subscription_period_end("user-1") == TRIAL_END
    assert await worker_a.get_active_subscription_period_end("user-1") == TRIAL_END
    assert fake_stripe.lookups == 1


@pytest.mark.asyncio
async def test_a_plan_bought_on_one_worker_shows_its_renewal_on_another(
    trial, fake_stripe, worker_b, save_tier, redis_up
):
    """Trial → cancel → "Subscribe to Pro": the status read that follows the
    purchase lands on a worker that read the trial moments before."""
    assert await worker_b.get_active_subscription_period_end("user-1") == TRIAL_END

    await conversion.convert_cancel_pending_trial(trial, SubscriptionTier.PRO)
    save_tier.assert_awaited_once()

    assert (
        await worker_b.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )
    assert fake_stripe.lookups == 2
    assert await _cached_period_end(redis_up, trial.customer_id) == PAID_PERIOD_END
    # The refreshed lookup is shared again.
    assert (
        await worker_a.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )
    assert fake_stripe.lookups == 2


@pytest.mark.asyncio
async def test_a_webhook_on_one_worker_refreshes_the_lookup_on_another(
    trial, fake_stripe, worker_b, redis_up
):
    """A change made in Stripe (the dashboard, a renewal) reaches us only as a
    webhook, on whichever worker Stripe's request lands."""
    assert await worker_b.get_active_subscription_period_end("user-1") == TRIAL_END

    converted = await fake_stripe.modify_async("sub_1", trial_end="now")
    await worker_a.sync_subscription_from_stripe(dict(converted))

    assert (
        await worker_b.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )
    assert fake_stripe.lookups == 2


@pytest.mark.asyncio
async def test_a_read_in_flight_during_the_purchase_does_not_cache_the_trial(
    trial, fake_stripe, worker_b, redis_up, monkeypatch
):
    """A page load whose Stripe answer predates the purchase, but arrives after
    it, must neither be served to later reads nor overwrite the fresh entry."""
    release = fake_stripe.hold_next_lookup(monkeypatch, worker_b)
    in_flight = asyncio.create_task(
        worker_b.get_active_subscription_period_end("user-1")
    )
    await asyncio.wait_for(fake_stripe.reached.wait(), WAIT)

    await conversion.convert_cancel_pending_trial(trial, SubscriptionTier.PRO)
    assert (
        await worker_a.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )
    release.set()
    assert await asyncio.wait_for(in_flight, WAIT) == TRIAL_END

    assert await _cached_period_end(redis_up, trial.customer_id) == PAID_PERIOD_END
    assert (
        await worker_b.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )
    assert fake_stripe.lookups == 2


@pytest.mark.asyncio
async def test_a_lookup_read_before_redis_lost_its_data_is_not_served(
    trial, fake_stripe, worker_b, redis_up, monkeypatch
):
    """Trial read → purchase → Redis restarts empty → the slow trial answer
    arrives. With no token left, nothing may vouch for that answer."""
    release = fake_stripe.hold_next_lookup(monkeypatch, worker_b)
    in_flight = asyncio.create_task(
        worker_b.get_active_subscription_period_end("user-1")
    )
    await asyncio.wait_for(fake_stripe.reached.wait(), WAIT)
    await conversion.convert_cancel_pending_trial(trial, SubscriptionTier.PRO)
    for key in (
        ACTIVE._version_key(trial.customer_id),
        ACTIVE._entry_key(trial.customer_id),
    ):
        await asyncio.wait_for(redis_up.delete(key), WAIT)

    release.set()
    assert await asyncio.wait_for(in_flight, WAIT) == TRIAL_END
    assert not await asyncio.wait_for(
        redis_up.get(ACTIVE._entry_key(trial.customer_id)), WAIT
    )

    assert (
        await worker_a.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )
    assert (
        await worker_b.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )
    assert fake_stripe.lookups == 2


@pytest.mark.asyncio
async def test_a_read_after_redis_lost_its_data_does_not_join_an_older_lookup(
    trial, fake_stripe, worker_b, redis_up, monkeypatch
):
    """A lookup started before a purchase on another worker, then Redis loses
    every key: a read on this worker after that asks Stripe itself instead of
    waiting for the old lookup's answer."""
    release = fake_stripe.hold_next_lookup(monkeypatch, worker_b)
    in_flight = asyncio.create_task(
        worker_b.get_active_subscription_period_end("user-1")
    )
    await asyncio.wait_for(fake_stripe.reached.wait(), WAIT)
    await conversion.convert_cancel_pending_trial(trial, SubscriptionTier.PRO)
    for key in (
        ACTIVE._version_key(trial.customer_id),
        ACTIVE._entry_key(trial.customer_id),
    ):
        await asyncio.wait_for(redis_up.delete(key), WAIT)

    try:
        after = await asyncio.wait_for(
            worker_b.get_active_subscription_period_end("user-1"), 2
        )
    finally:
        release.set()
    assert after == PAID_PERIOD_END
    assert await asyncio.wait_for(in_flight, WAIT) == TRIAL_END


@pytest.mark.asyncio
async def test_concurrent_misses_on_a_worker_share_one_stripe_call(
    fake_stripe, worker_b, redis_up
):
    ends = await asyncio.wait_for(
        asyncio.gather(
            *(worker_b.get_active_subscription_period_end("user-1") for _ in range(60))
        ),
        WAIT,
    )
    assert set(ends) == {TRIAL_END}
    assert fake_stripe.lookups == 1


@pytest.mark.asyncio
async def test_the_tier_is_saved_before_the_lookup_is_dropped(
    fake_stripe, worker_b, save_tier
):
    """Saving the tier can change Stripe too (a rejected trial is canceled), so
    the lookup is dropped only afterwards, and even when saving fails."""
    calls: list[str] = []

    async def failing_save(*args) -> None:
        calls.append("save")
        raise RuntimeError("the save failed")

    async def invalidate(customer_id: str) -> None:
        calls.append("invalidate")

    save_tier.side_effect = failing_save
    with patch.object(worker_a, "invalidate_active_subscription_cache", invalidate):
        with pytest.raises(RuntimeError):
            await worker_a.sync_subscription_from_stripe(fake_stripe.subscription)
    assert calls == ["save", "invalidate"]


class _Outage:
    """Stands in for ``get_redis_async``, which every Redis operation calls
    first, so ``attempts`` counts each time the cache tries Redis.

    ``serial_waits`` counts the deadline waits a caller pays one after the
    other: a hanging attempt that starts while no other is hanging opens a
    new wait; attempts made alongside it share that one."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.attempts = 0
        self.serial_waits = 0
        self._hanging = 0

    async def __call__(self) -> "_HangingRedis":
        self.attempts += 1
        if self.kind == "connect-hangs":
            await self.hang()
        if self.kind == "refused":
            raise RedisConnectionError("refused")
        if self.kind == "standalone-redis":
            raise RedisClusterException("Cluster mode is not enabled on this node")
        return _HangingRedis(self)

    async def hang(self) -> None:
        if self._hanging == 0:
            self.serial_waits += 1
        self._hanging += 1
        try:
            await asyncio.Event().wait()
        finally:
            self._hanging -= 1


class _HangingRedis:
    def __init__(self, outage: _Outage) -> None:
        self.outage = outage

    async def mget(self, *args, **kwargs):
        await self.outage.hang()

    async def set(self, *args, **kwargs):
        await self.outage.hang()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind", ["connect-hangs", "command-hangs", "refused", "standalone-redis"]
)
async def test_a_redis_outage_never_slows_or_blocks_billing(
    trial, fake_stripe, worker_b, save_tier, kind
):
    """With Redis hanging or unusable, lookups answer from Stripe after at
    most one deadline, then skip Redis entirely for the backoff, and a charge
    still saves the paid plan.

    Counts Redis attempts rather than timing them; the wall clock is only a
    backstop, far below what paying a deadline per read would take."""
    outage = _Outage(kind)
    hangs = kind in ("connect-hangs", "command-hangs")
    backstop = 10 * versioned_cache._REDIS_DEADLINE_SECONDS
    with patch.object(versioned_cache, "get_redis_async", outage):
        started = time.monotonic()
        ends = await asyncio.wait_for(
            asyncio.gather(
                *(
                    worker_a.get_active_subscription_period_end("user-1")
                    for _ in range(60)
                )
            ),
            WAIT,
        )
        first_wave = outage.attempts
        # Each concurrent read tries Redis at most once before the backoff,
        # and when Redis hangs every read pays that one deadline at the same
        # time, not one after another.
        assert 1 <= first_wave <= 60
        assert outage.serial_waits == (1 if hangs else 0)
        for _ in range(100):
            ends.append(await worker_a.get_active_subscription_period_end("user-1"))
        # The backoff holds: no read in its window tries Redis again.
        assert outage.attempts == first_wave
        assert set(ends) == {TRIAL_END}
        # Stripe traffic stays bounded while Redis can't be used.
        assert fake_stripe.lookups == 1
        assert time.monotonic() - started < backstop

        before = outage.attempts
        waits_before = outage.serial_waits
        started = time.monotonic()
        await asyncio.wait_for(
            conversion.convert_cancel_pending_trial(trial, SubscriptionTier.PRO),
            WAIT,
        )
        # Only invalidations, which never skip Redis, try it:
        # invalidate_billing_lookups drops both caches at once, after the tier
        # sync and again after saving the conversion, so the purchase pays at
        # most two deadlines in a row.
        assert outage.attempts - before == 4
        assert outage.serial_waits - waits_before == (2 if hangs else 0)
        assert time.monotonic() - started < backstop
        save_tier.assert_awaited_once()
        assert save_tier.await_args.args[0]["status"] == "active"

        # The change this process made is not hidden by what it kept.
        assert (
            await worker_a.get_active_subscription_period_end("user-1")
            == PAID_PERIOD_END
        )


@pytest.mark.asyncio
async def test_an_answer_read_before_a_change_is_not_kept_during_an_outage(
    trial, fake_stripe, worker_b, save_tier, monkeypatch
):
    """Redis is out: a lookup started before this worker's purchase answers
    after it. Its callers get the old answer, but this worker must not keep
    it and serve it for the next few seconds."""
    monkeypatch.setattr(
        versioned_cache,
        "get_redis_async",
        AsyncMock(side_effect=RedisConnectionError("refused")),
    )
    release = fake_stripe.hold_next_lookup(monkeypatch, worker_a)
    in_flight = asyncio.create_task(
        worker_a.get_active_subscription_period_end("user-1")
    )
    await asyncio.wait_for(fake_stripe.reached.wait(), WAIT)

    await conversion.convert_cancel_pending_trial(trial, SubscriptionTier.PRO)
    assert (
        await worker_a.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )
    release.set()
    assert await asyncio.wait_for(in_flight, WAIT) == TRIAL_END

    assert (
        await worker_a.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )
    assert fake_stripe.lookups == 2


class _BlippingRedis:
    """The real Redis, except that the next script call can be held on its way
    and SETs can be made to fail."""

    def __init__(self, redis):
        self.redis = redis
        self.fail_sets = False
        self.release_eval: asyncio.Event | None = None
        self.eval_reached = asyncio.Event()

    async def mget(self, *args, **kwargs):
        return await self.redis.mget(*args, **kwargs)

    async def get(self, *args, **kwargs):
        return await self.redis.get(*args, **kwargs)

    async def set(self, *args, **kwargs):
        if self.fail_sets:
            raise RedisConnectionError("blip")
        return await self.redis.set(*args, **kwargs)

    async def eval(self, *args):
        if self.release_eval is not None:
            release, self.release_eval = self.release_eval, None
            self.eval_reached.set()
            await asyncio.wait_for(release.wait(), WAIT)
        return await self.redis.eval(*args)


@pytest.mark.asyncio
async def test_a_write_racing_an_unrecorded_invalidation_is_not_served_later(
    trial, fake_stripe, worker_b, redis_up, monkeypatch
):
    """A lookup's write is on its way to Redis when this worker changes the
    subscription, and Redis blips so the new token isn't stored: the write
    lands under the old token. Once Redis is back, this worker must not
    serve it."""
    monkeypatch.setattr(versioned_cache, "_REDIS_DEADLINE_SECONDS", WAIT)
    redis = _BlippingRedis(redis_up)
    monkeypatch.setattr(
        versioned_cache, "get_redis_async", AsyncMock(return_value=redis)
    )
    redis.release_eval = release = asyncio.Event()
    first = asyncio.create_task(worker_a.get_active_subscription_period_end("user-1"))
    await asyncio.wait_for(redis.eval_reached.wait(), WAIT)

    await fake_stripe.modify_async("sub_1", trial_end="now")
    redis.fail_sets = True
    await cache_a.invalidate_active_subscription_cache(trial.customer_id)
    redis.fail_sets = False
    assert (
        await worker_a.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )

    release.set()
    assert await asyncio.wait_for(first, WAIT) == TRIAL_END
    # The old answer did land in Redis under the unchanged token.
    assert await _cached_period_end(redis_up, trial.customer_id) == TRIAL_END

    monkeypatch.setattr(versioned_cache._backoff, "until", 0.0)
    assert (
        await worker_a.get_active_subscription_period_end("user-1") == PAID_PERIOD_END
    )


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def redis_down(monkeypatch):
    monkeypatch.setattr(
        versioned_cache,
        "get_redis_async",
        AsyncMock(side_effect=RedisConnectionError("down")),
    )


@pytest.mark.asyncio
async def test_a_live_mark_survives_any_number_of_failed_invalidations(redis_down):
    """A mark is what stops this worker serving an entry written under the
    old token; other customers' failures must never push it out."""
    await cache_a.invalidate_active_subscription_cache("cus_changed")
    for i in range(2100):
        await cache_a.invalidate_active_subscription_cache(f"cus_other_{i}")
    assert ACTIVE._is_unconfirmed("cus_changed")


@pytest.mark.asyncio
async def test_a_renewed_mark_outlives_expired_ones(redis_down, monkeypatch):
    """Expired marks must not crowd out a live one, whatever their order."""
    clock = _Clock()
    monkeypatch.setattr(versioned_cache, "time", clock)
    monkeypatch.setattr(
        ACTIVE,
        "_unconfirmed",
        UnconfirmedInvalidations(
            lifetime=cache_a.SUBSCRIPTION_LOOKUP_TTL_SECONDS + 1, maxsize=4
        ),
    )
    for customer in ("cus_changed", "cus_1", "cus_2", "cus_3"):
        await cache_a.invalidate_active_subscription_cache(customer)
    clock.now += 60

    await cache_a.invalidate_active_subscription_cache("cus_changed")
    await cache_a.invalidate_active_subscription_cache("cus_4")
    assert ACTIVE._is_unconfirmed("cus_changed")
