"""The active-subscription lookup cache, shared by every API worker.

The billing status endpoint reads a customer's active subscription several
times per request, on almost every page load, so the lookup is cached briefly
in Redis. Each entry records the customer's version token at the time it was
read from Stripe; every subscription change replaces the token, so the next
read on any worker misses and asks Stripe again. A read that was already in
flight when the change landed writes its answer under the old token, where no
one reads it.

Redis is only an optimisation here. Every operation has a short hard deadline,
and after a failure this process skips Redis for a few seconds, so an outage
or a hung connection costs a lookup at most one deadline before it goes to
Stripe, and never blocks or fails the billing change around it.
"""

import asyncio
import logging
import time
from typing import Awaitable, Callable, TypeVar
from uuid import uuid4

import stripe
from pydantic import BaseModel, ValidationError

from backend.data.redis_client import AsyncRedisClient, get_redis_async

logger = logging.getLogger(__name__)

T = TypeVar("T")

SUBSCRIPTION_LOOKUP_TTL_SECONDS = 15
# Bounds connecting as well as the command: the shared client retries a
# failed connection for minutes on its own.
_REDIS_DEADLINE_SECONDS = 0.3
_REDIS_BACKOFF_SECONDS = 5.0
_VERSION_TTL_SECONDS = 24 * 60 * 60
_DEFAULT_VERSION = "0"
KEY_PREFIX = "active_subscription"


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
    customer_id: str,
    fetch: Callable[[], Awaitable[stripe.Subscription | None]],
) -> ActiveSubscription | None:
    """The cached lookup, or ``fetch``'s answer when it isn't cached or Redis
    can't be used."""
    keys = (_version_key(customer_id), _lookup_key(customer_id))
    try:
        stored_version, cached = await _call_redis(lambda redis: redis.mget(keys))
    except _RedisUnavailable:
        return summarize_subscription(await fetch())
    version = _text(stored_version) or _DEFAULT_VERSION
    hit = _parse_lookup(_text(cached), version)
    if hit is not None:
        return hit.subscription
    subscription = summarize_subscription(await fetch())
    entry = _CachedLookup(version=version, subscription=subscription)
    try:
        await _call_redis(
            lambda redis: redis.set(
                _lookup_key(customer_id),
                entry.model_dump_json(),
                ex=SUBSCRIPTION_LOOKUP_TTL_SECONDS,
            )
        )
    except _RedisUnavailable:
        pass
    return subscription


async def invalidate_active_subscription_cache(customer_id: str) -> None:
    """Make every worker read the customer's subscription fresh from Stripe.

    Call it after the change has landed in Stripe, never before: a read
    started earlier could otherwise cache the old state under the new token.
    Never raises and returns within the Redis deadline, because the change it
    follows has already been made.
    """
    try:
        await _call_redis(
            lambda redis: redis.set(
                _version_key(customer_id), uuid4().hex, ex=_VERSION_TTL_SECONDS
            ),
            skip_during_backoff=False,
        )
    except _RedisUnavailable:
        logger.warning(
            f"Could not invalidate the subscription lookup for {customer_id};"
            f" it can stay stale for {SUBSCRIPTION_LOOKUP_TTL_SECONDS}s"
        )


def _version_key(customer_id: str) -> str:
    # The hash tag keeps both keys on one cluster slot, so one MGET reads them.
    return f"{KEY_PREFIX}:{{{customer_id}}}:version"


def _lookup_key(customer_id: str) -> str:
    return f"{KEY_PREFIX}:{{{customer_id}}}:lookup"


def _text(value: str | bytes | None) -> str | None:
    # The shared client decodes responses; its type hints don't say so.
    return value.decode() if isinstance(value, bytes) else value


def _parse_lookup(raw: str | None, version: str) -> _CachedLookup | None:
    if not raw:
        return None
    try:
        entry = _CachedLookup.model_validate_json(raw)
    except ValidationError:
        return None
    return entry if entry.version == version else None


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
