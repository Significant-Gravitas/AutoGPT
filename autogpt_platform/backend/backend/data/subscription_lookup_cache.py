"""The active-subscription lookup cache, shared by every API worker.

The billing status endpoint reads a customer's active subscription several
times per request, on almost every page load, so the lookup is cached briefly
in Redis. Each customer has a random version token; every subscription change
replaces it, so the next read on any worker misses and asks Stripe again. An
entry is written only while the token it was read under is still current, so
a read that was in flight when a change landed can't put the old answer back.
A missing token (expired, or Redis lost its data) is never treated as a
version: the reader establishes a fresh one before it asks Stripe.

Within a process, concurrent lookups for one customer and token share one
Stripe call. A lookup stores its answer only while it is still the current one
for its key, so an answer read before a change this process made is returned
to the callers that were already waiting for it, and never kept.

Redis is only an optimisation here. Every operation has a short hard deadline,
and after a failure this process skips Redis for a few seconds, so an outage
or a hung connection costs a lookup at most one deadline, and never blocks or
fails the billing change around it. Meanwhile answers are kept in process for
a few seconds, to keep Stripe traffic bounded. A Redis that isn't a cluster
(the platform always runs one) gets this fallback all the time.
"""

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Coroutine, TypeVar
from uuid import uuid4

import stripe
from pydantic import BaseModel, ValidationError

from backend.data.redis_client import AsyncRedisClient, get_redis_async

logger = logging.getLogger(__name__)

T = TypeVar("T")
Fetch = Callable[[], Awaitable[stripe.Subscription | None]]

SUBSCRIPTION_LOOKUP_TTL_SECONDS = 15
# Bounds connecting as well as the command: the shared client retries a
# failed connection for minutes on its own.
_REDIS_DEADLINE_SECONDS = 0.3
_REDIS_BACKOFF_SECONDS = 5.0
# How long this process reuses an answer while Redis can't be used; also how
# long another worker can show the old state after a change in that time.
_FALLBACK_TTL_SECONDS = 5.0
_FALLBACK_MAXSIZE = 2048
_UNCONFIRMED_MAXSIZE = 2048
_VERSION_TTL_SECONDS = 24 * 60 * 60
KEY_PREFIX = "active_subscription"
# The in-flight key for reads while Redis can't be used; tokens are hex.
_NO_REDIS = "no-redis"

# KEYS: version, lookup. ARGV: version read, entry, TTL.
_WRITE_IF_CURRENT = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
    return redis.call('SET', KEYS[2], ARGV[2], 'EX', ARGV[3])
end
return false
"""


class ActiveSubscription(BaseModel):
    """What the billing display reads from the active or trialing subscription.

    Cached instead of the Stripe object, which carries the API key it was
    fetched with.
    """

    status: str | None = None
    current_period_start: int | None = None
    current_period_end: int | None = None
    price_id: str | None = None


class _CachedLookup(BaseModel):
    version: str
    subscription: ActiveSubscription | None


class _RedisUnavailable(Exception):
    pass


class _Backoff:
    """Process-local: skip Redis for a while after it failed."""

    def __init__(self) -> None:
        self.until = 0.0

    def active(self) -> bool:
        return time.monotonic() < self.until

    def trip(self) -> None:
        if not self.active():
            logger.warning(
                "Subscription lookup cache unavailable; reading Stripe directly"
                f" for {_REDIS_BACKOFF_SECONDS:.0f}s"
            )
        self.until = time.monotonic() + _REDIS_BACKOFF_SECONDS


_backoff = _Backoff()
# Lookups running in this process, by (event loop, customer, version).
_in_flight: dict[
    tuple[asyncio.AbstractEventLoop, str, str],
    asyncio.Task[ActiveSubscription | None],
] = {}
# Answers kept while Redis can't be used: customer -> (expires at, answer).
_fallback: dict[str, tuple[float, ActiveSubscription | None]] = {}


class _UnconfirmedInvalidations:
    """Customers whose invalidation this process couldn't record in Redis.

    A write already on its way when the change landed is stored under the
    old, unchanged token, so until any such entry has expired this process
    reads the customer as if Redis were out. A mark is never dropped while it
    is live: with no room left, every customer is covered until the latest
    mark would have expired.
    """

    def __init__(self) -> None:
        self.until: dict[str, float] = {}
        self.everyone_until = 0.0

    def add(self, customer_id: str) -> None:
        now = time.monotonic()
        # Outlives any entry written now: a write in flight finishes within
        # the Redis deadline, and entries live for the lookup TTL.
        until = now + SUBSCRIPTION_LOOKUP_TTL_SECONDS + 1
        self._purge(now)
        # Re-inserted at the end, so insertion order stays expiry order.
        self.until.pop(customer_id, None)
        if len(self.until) >= _UNCONFIRMED_MAXSIZE:
            self.everyone_until = max(self.everyone_until, until)
            return
        self.until[customer_id] = until

    def clear(self, customer_id: str) -> None:
        self.until.pop(customer_id, None)

    def covers(self, customer_id: str) -> bool:
        now = time.monotonic()
        if now < self.everyone_until:
            return True
        until = self.until.get(customer_id)
        return until is not None and until > now

    def _purge(self, now: float) -> None:
        for customer_id, until in list(self.until.items()):
            if until > now:
                break
            del self.until[customer_id]


_unconfirmed = _UnconfirmedInvalidations()


def summarize_subscription(
    subscription: stripe.Subscription | dict | None,
) -> ActiveSubscription | None:
    if subscription is None:
        return None
    items = (subscription.get("items") or {}).get("data") or []
    price = items[0].get("price") if items else None
    return ActiveSubscription(
        status=subscription.get("status"),
        current_period_start=subscription.get("current_period_start"),
        current_period_end=subscription.get("current_period_end"),
        price_id=price if isinstance(price, str) else (price or {}).get("id"),
    )


async def read_active_subscription(
    customer_id: str, fetch: Fetch
) -> ActiveSubscription | None:
    """The cached lookup, or ``fetch``'s answer when it isn't cached."""
    if _is_unconfirmed(customer_id):
        return await _read_without_redis(customer_id, fetch)
    try:
        version, cached = await _read_entry(customer_id)
        if version is None:
            # Established before the in-flight key is chosen, so a read after
            # Redis lost the token never joins a lookup started before.
            version = await _establish_version(customer_id)
    except _RedisUnavailable:
        return await _read_without_redis(customer_id, fetch)
    if cached is not None and cached.version == version:
        return cached.subscription
    return await _shared(
        customer_id, version, lambda: _fetch_and_store(customer_id, version, fetch)
    )


async def invalidate_active_subscription_cache(customer_id: str) -> None:
    """Make every worker read the customer's subscription fresh from Stripe.

    Call it after the change has landed in Stripe, never before: a read
    started earlier could otherwise cache the old state under the new token.
    Never raises and returns within the Redis deadline, because the change it
    follows has already been made.
    """
    _forget_locally(customer_id)
    try:
        await _call_redis(
            lambda redis: redis.set(
                _version_key(customer_id), uuid4().hex, ex=_VERSION_TTL_SECONDS
            ),
            skip_during_backoff=False,
        )
    except _RedisUnavailable:
        # The old token still vouches for entries written under it.
        _unconfirmed.add(customer_id)
        logger.warning(
            f"Could not invalidate the subscription lookup for {customer_id};"
            f" other workers can show it stale for {SUBSCRIPTION_LOOKUP_TTL_SECONDS}s"
        )
    else:
        # From now on the script refuses anything read under the old token.
        _unconfirmed.clear(customer_id)


async def _read_entry(customer_id: str) -> tuple[str | None, _CachedLookup | None]:
    keys = (_version_key(customer_id), _lookup_key(customer_id))
    version, raw = await _call_redis(lambda redis: redis.mget(keys))
    return _text(version), _parse_lookup(_text(raw))


async def _establish_version(customer_id: str) -> str:
    """A fresh token for a customer who has none, or the one another reader
    created first."""
    token = uuid4().hex
    key = _version_key(customer_id)
    created = await _call_redis(
        lambda redis: redis.set(key, token, nx=True, ex=_VERSION_TTL_SECONDS)
    )
    if created:
        return token
    current = _text(await _call_redis(lambda redis: redis.get(key)))
    if current:
        return current
    # Expired again in between: a token of our own only costs a miss.
    await _call_redis(lambda redis: redis.set(key, token, ex=_VERSION_TTL_SECONDS))
    return token


async def _fetch_and_store(
    customer_id: str, version: str, fetch: Fetch
) -> ActiveSubscription | None:
    # Another lookup may have stored it while this one was being started.
    try:
        current, cached = await _read_entry(customer_id)
    except _RedisUnavailable:
        current, cached = None, None
    if current == version and cached is not None and cached.version == version:
        return cached.subscription
    subscription = summarize_subscription(await fetch())
    if not _still_current(customer_id, version) or _is_unconfirmed(customer_id):
        return subscription
    entry = _CachedLookup(version=version, subscription=subscription)
    keys_and_args = (
        _version_key(customer_id),
        _lookup_key(customer_id),
        version,
        entry.model_dump_json(),
        SUBSCRIPTION_LOOKUP_TTL_SECONDS,
    )
    try:
        await _call_redis(
            lambda redis: redis.eval(_WRITE_IF_CURRENT, 2, *keys_and_args)
        )
    except _RedisUnavailable:
        pass
    return subscription


async def _read_without_redis(
    customer_id: str, fetch: Fetch
) -> ActiveSubscription | None:
    kept = _fallback.get(customer_id)
    if kept is not None and kept[0] > time.monotonic():
        return kept[1]
    return await _shared(
        customer_id, _NO_REDIS, lambda: _fetch_and_keep(customer_id, fetch)
    )


async def _fetch_and_keep(customer_id: str, fetch: Fetch) -> ActiveSubscription | None:
    subscription = summarize_subscription(await fetch())
    if not _still_current(customer_id, _NO_REDIS):
        return subscription
    _fallback[customer_id] = (time.monotonic() + _FALLBACK_TTL_SECONDS, subscription)
    if len(_fallback) > _FALLBACK_MAXSIZE:
        # Dropping a kept answer only costs another Stripe call.
        del _fallback[next(iter(_fallback))]
    return subscription


def _is_unconfirmed(customer_id: str) -> bool:
    return _unconfirmed.covers(customer_id)


async def _shared(
    customer_id: str,
    version: str,
    lookup: Callable[[], Coroutine[Any, Any, ActiveSubscription | None]],
) -> ActiveSubscription | None:
    """Run ``lookup`` once for every concurrent caller with the same key. The
    version is part of the key, so a read after a change never joins one
    started before it."""
    key = (asyncio.get_running_loop(), customer_id, version)
    task = _in_flight.get(key)
    if task is None:
        task = asyncio.create_task(lookup())
        _in_flight[key] = task
        task.add_done_callback(lambda done: _drop_in_flight(key, done))
    # One caller giving up must not cancel the lookup the others wait on.
    return await asyncio.shield(task)


def _still_current(customer_id: str, version: str) -> bool:
    """Whether the running lookup still owns its in-flight key: invalidating
    in this process drops it, and then its answer must not be kept."""
    key = (asyncio.get_running_loop(), customer_id, version)
    return _in_flight.get(key) is asyncio.current_task()


def _drop_in_flight(
    key: tuple[asyncio.AbstractEventLoop, str, str],
    task: asyncio.Task[ActiveSubscription | None],
) -> None:
    if _in_flight.get(key) is task:
        del _in_flight[key]


def _forget_locally(customer_id: str) -> None:
    """Stop this process reusing what it read before a change: its fallback
    answer, and lookups still running (whose version may not have changed if
    Redis couldn't be told)."""
    _fallback.pop(customer_id, None)
    for key in [key for key in _in_flight if key[1] == customer_id]:
        del _in_flight[key]


def _version_key(customer_id: str) -> str:
    # The hash tag keeps both keys on one cluster slot, for MGET and the script.
    return f"{KEY_PREFIX}:{{{customer_id}}}:version"


def _lookup_key(customer_id: str) -> str:
    return f"{KEY_PREFIX}:{{{customer_id}}}:lookup"


def _text(value: str | bytes | None) -> str | None:
    # The shared client decodes responses; its type hints don't say so.
    return value.decode() if isinstance(value, bytes) else value


def _parse_lookup(raw: str | None) -> _CachedLookup | None:
    if not raw:
        return None
    try:
        return _CachedLookup.model_validate_json(raw)
    except ValidationError:
        return None


async def _call_redis(
    operation: Callable[[AsyncRedisClient], Awaitable[T]],
    *,
    skip_during_backoff: bool = True,
) -> T:
    if skip_during_backoff and _backoff.active():
        raise _RedisUnavailable()
    try:
        return await asyncio.wait_for(
            _with_client(operation), timeout=_REDIS_DEADLINE_SECONDS
        )
    except Exception as exc:
        _backoff.trip()
        raise _RedisUnavailable() from exc


async def _with_client(operation: Callable[[AsyncRedisClient], Awaitable[T]]) -> T:
    return await operation(await get_redis_async())
