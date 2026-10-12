"""Randomized interleavings of the versioned billing caches: the active
subscription lookup and the pending subscription change.

The invariant that matters to a customer: once this worker has changed a
subscription and invalidated the cache, no read that starts afterwards on
this worker returns the state as it was before the change. Each seed
interleaves reads, changes, Redis outages and blips (failed writes, replies
lost after the write applied, lost data), slow Stripe answers and slow
script writes, on a fake clock, with the in-process bounds shrunk so they
are under pressure. Nothing sleeps; the clock only moves when told to.
"""

import asyncio
import random
from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest
from prisma.enums import SubscriptionTier
from redis.exceptions import ConnectionError as RedisConnectionError

from backend.data import credit, versioned_cache
from backend.data.subscription_lookup_cache import (
    ACTIVE_SUBSCRIPTIONS,
    invalidate_active_subscription_cache,
    read_active_subscription,
)
from backend.data.subscription_lookup_unconfirmed import UnconfirmedInvalidations
from backend.data.subscription_pending_cache import (
    PENDING_CHANGES,
    PendingChange,
    invalidate_pending_subscription_change,
)

SEEDS = range(300)
STEPS = 80
CUSTOMERS = ("cus_a", "cus_b", "cus_c", "cus_d")
BASE = 1_800_000_000
# Real Redis finishes or drops a command within the client deadline.
DEADLINE = 0.3


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


class Holds:
    """Answers held until the run releases them; Redis ones are dropped once
    the clock passes the client deadline, as the real client would."""

    def __init__(self, clock: Clock, rng: random.Random, chance: float) -> None:
        self.clock = clock
        self.rng = rng
        self.chance = chance
        self.held: list[tuple[float, asyncio.Future[None]]] = []

    async def maybe_hold(self) -> None:
        if self.rng.random() >= self.chance:
            return
        future = asyncio.get_running_loop().create_future()
        self.held.append((self.clock.now, future))
        await future

    def release_one(self) -> None:
        waiting = [future for _, future in self.held if not future.done()]
        if waiting:
            self.rng.choice(waiting).set_result(None)

    def release_all(self) -> None:
        for _, future in self.held:
            if not future.done():
                future.set_result(None)

    def drop_older_than(self, limit: float) -> None:
        for started, future in self.held:
            if not future.done() and self.clock.now - started > limit:
                future.set_exception(TimeoutError("past the client deadline"))


class FakeRedis:
    """The commands the cache uses, with TTLs on the fake clock."""

    def __init__(self, clock: Clock, holds: Holds) -> None:
        self.clock = clock
        self.holds = holds
        self.data: dict[str, tuple[str, float]] = {}
        self.mode = "up"

    def _read(self, key: str) -> str | None:
        stored = self.data.get(key)
        if stored is None or stored[1] <= self.clock.now:
            return None
        return stored[0]

    def _reachable(self) -> None:
        if self.mode == "down":
            raise RedisConnectionError("down")

    async def mget(self, keys):
        self._reachable()
        return [self._read(key) for key in keys]

    async def get(self, key):
        self._reachable()
        return self._read(key)

    async def set(self, key, value, nx=False, ex=None):
        self._reachable()
        if self.mode == "set-fails":
            raise RedisConnectionError("blip")
        if nx and self._read(key) is not None:
            return None
        self.data[key] = (value, self.clock.now + ex)
        if self.mode == "reply-lost":
            raise RedisConnectionError("reply lost after the write applied")
        return True

    async def eval(self, script, numkeys, version_key, lookup_key, version, entry, ttl):
        self._reachable()
        await self.holds.maybe_hold()
        if self._read(version_key) != version:
            return None
        self.data[lookup_key] = (entry, self.clock.now + ttl)
        return True


class Run:
    def __init__(self, seed: int, target: str, monkeypatch: pytest.MonkeyPatch) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.clock = Clock()
        self.stripe_holds = Holds(self.clock, self.rng, 0.3)
        self.redis_holds = Holds(self.clock, self.rng, 0.3)
        self.redis = FakeRedis(self.clock, self.redis_holds)
        self.state = dict.fromkeys(CUSTOMERS, 0)
        self.completed = dict.fromkeys(CUSTOMERS, 0)
        self.violations: list[str] = []
        self.tasks: list[asyncio.Task[None]] = []
        self.cache, self.read_value, self.invalidate = TARGETS[target]

        async def get_redis():
            return self.redis

        async def fetch_pending(customer_id: str) -> PendingChange:
            value = await self.answer(customer_id)
            return PendingChange(
                tier=SubscriptionTier.NO_TIER,
                effective_at=datetime.fromtimestamp(BASE + value, UTC),
            )

        monkeypatch.setattr(versioned_cache, "time", self.clock)
        monkeypatch.setattr(versioned_cache, "get_redis_async", get_redis)
        monkeypatch.setattr(versioned_cache, "_REDIS_DEADLINE_SECONDS", 3600)
        monkeypatch.setattr(versioned_cache, "_FALLBACK_MAXSIZE", 1)
        monkeypatch.setattr(versioned_cache, "_backoff", versioned_cache._Backoff())

        async def user_by_id(user_id: str) -> MagicMock:
            # Each key is both the user id and their Stripe customer id.
            return MagicMock(stripe_customer_id=user_id)

        monkeypatch.setattr(credit, "_fetch_pending_subscription_change", fetch_pending)
        monkeypatch.setattr(credit, "get_user_by_id", user_by_id)
        monkeypatch.setattr(
            self.cache,
            "_unconfirmed",
            UnconfirmedInvalidations(lifetime=self.cache.ttl_seconds + 1, maxsize=1),
        )
        monkeypatch.setattr(self.cache, "_fallback", {})
        monkeypatch.setattr(self.cache, "_in_flight", {})

    async def answer(self, key: str) -> int:
        """What the source says now, delivered after a possible delay."""
        value = self.state[key]
        await self.stripe_holds.maybe_hold()
        return value

    async def read(self, key: str) -> None:
        required = self.completed[key]
        seen = await self.read_value(self, key)
        if seen < required:
            self.violations.append(
                f"seed {self.seed}: {key} read {seen} after change {required}"
            )

    async def change(self, key: str) -> None:
        self.state[key] += 1
        value = self.state[key]
        await self.invalidate(key)
        self.completed[key] = max(self.completed[key], value)

    def step(self) -> None:
        roll = self.rng.random()
        customer_id = self.rng.choice(CUSTOMERS)
        if roll < 0.35:
            self.tasks.append(asyncio.create_task(self.read(customer_id)))
        elif roll < 0.5:
            self.tasks.append(asyncio.create_task(self.change(customer_id)))
        elif roll < 0.6:
            self.redis.mode = self.rng.choice(
                ["up", "up", "down", "set-fails", "reply-lost"]
            )
        elif roll < 0.63:
            self.redis.data.clear()
        elif roll < 0.8:
            self.rng.choice([self.stripe_holds, self.redis_holds]).release_one()
        else:
            self.clock.now += self.rng.choice([0.1, 0.5, 2, 6, 16, 20])
            self.redis_holds.drop_older_than(DEADLINE)

    async def play(self) -> None:
        for _ in range(STEPS):
            self.step()
            for _ in range(self.rng.randint(1, 6)):
                await asyncio.sleep(0)
        self.redis.mode = "up"
        while not all(task.done() for task in self.tasks):
            self.stripe_holds.release_all()
            self.redis_holds.release_all()
            await asyncio.sleep(0)
        for task in self.tasks:
            task.result()


async def read_active(run: Run, customer_id: str) -> int:
    async def ask_stripe():
        value = await run.answer(customer_id)
        return {
            "status": "active",
            "current_period_start": BASE,
            "current_period_end": BASE + value,
        }

    subscription = await read_active_subscription(customer_id, ask_stripe)
    return (subscription.current_period_end or 0) - BASE if subscription else -1


async def read_pending(run: Run, user_id: str) -> int:
    pending = await credit.get_pending_subscription_change(user_id)
    return int(pending[1].timestamp()) - BASE if pending else -1


TARGETS = {
    "active-subscription": (
        ACTIVE_SUBSCRIPTIONS,
        read_active,
        invalidate_active_subscription_cache,
    ),
    "pending-change": (
        PENDING_CHANGES,
        read_pending,
        invalidate_pending_subscription_change,
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("target", list(TARGETS))
async def test_no_read_after_a_change_returns_the_old_state(monkeypatch, target):
    violations: list[str] = []
    for seed in SEEDS:
        with monkeypatch.context() as patched:
            run = Run(seed, target, patched)
            await asyncio.wait_for(run.play(), 30)
            violations.extend(run.violations)
    assert violations == []
