"""Randomized interleavings of the subscription lookup cache.

The invariant that matters to a customer: once this worker has changed a
subscription and invalidated the lookup, no read that starts afterwards on
this worker returns the subscription as it was before the change. Each seed
interleaves reads, changes, Redis outages and blips (failed writes, replies
lost after the write applied, lost data), slow Stripe answers and slow
script writes, on a fake clock, with the in-process bounds shrunk so they
are under pressure. Nothing sleeps; the clock only moves when told to.
"""

import asyncio
import random

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from backend.data import subscription_lookup_cache as cache

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
    def __init__(self, seed: int, monkeypatch: pytest.MonkeyPatch) -> None:
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

        async def get_redis():
            return self.redis

        monkeypatch.setattr(cache, "time", self.clock)
        monkeypatch.setattr(cache, "get_redis_async", get_redis)
        monkeypatch.setattr(cache, "_REDIS_DEADLINE_SECONDS", 3600)
        monkeypatch.setattr(cache, "_FALLBACK_MAXSIZE", 1)
        monkeypatch.setattr(cache, "_UNCONFIRMED_MAXSIZE", 1)
        monkeypatch.setattr(cache, "_backoff", cache._Backoff())
        monkeypatch.setattr(cache, "_unconfirmed", cache._UnconfirmedInvalidations())
        monkeypatch.setattr(cache, "_fallback", {})
        monkeypatch.setattr(cache, "_in_flight", {})

    def fetch(self, customer_id: str):
        async def ask_stripe():
            value = self.state[customer_id]
            await self.stripe_holds.maybe_hold()
            return {
                "status": "active",
                "current_period_start": BASE,
                "current_period_end": BASE + value,
            }

        return ask_stripe

    async def read(self, customer_id: str) -> None:
        required = self.completed[customer_id]
        subscription = await cache.read_active_subscription(
            customer_id, self.fetch(customer_id)
        )
        seen = (subscription.current_period_end or 0) - BASE if subscription else -1
        if seen < required:
            self.violations.append(
                f"seed {self.seed}: {customer_id} read {seen} after change {required}"
            )

    async def change(self, customer_id: str) -> None:
        self.state[customer_id] += 1
        value = self.state[customer_id]
        await cache.invalidate_active_subscription_cache(customer_id)
        self.completed[customer_id] = max(self.completed[customer_id], value)

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


@pytest.mark.asyncio
async def test_no_read_after_a_change_returns_the_old_subscription(monkeypatch):
    violations: list[str] = []
    for seed in SEEDS:
        with monkeypatch.context() as patched:
            run = Run(seed, patched)
            await asyncio.wait_for(run.play(), 30)
            violations.extend(run.violations)
    assert violations == []
