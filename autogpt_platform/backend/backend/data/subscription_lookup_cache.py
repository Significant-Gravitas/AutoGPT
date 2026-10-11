"""The active-subscription lookup cache, shared by every API worker.

The billing status endpoint reads a customer's active subscription several
times per request, on almost every page load, so the lookup is cached briefly
in Redis. Each customer's entries are keyed by a version token that every
subscription change replaces, so the next read on any worker misses and asks
Stripe again. Deleting the entry instead would let a read already in flight
when the change landed write the old subscription back for another window.
"""

import logging
from uuid import uuid4

import stripe
from pydantic import BaseModel

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

SUBSCRIPTION_LOOKUP_TTL_SECONDS = 15
# Far longer than an entry lives, so once a token expires nothing is still
# cached under the default one it falls back to.
_VERSION_TTL_SECONDS = 24 * 60 * 60
_DEFAULT_VERSION = "0"


class ActiveSubscription(BaseModel):
    """What the billing display reads from the active or trialing subscription.

    Cached instead of the Stripe object, which carries the API key it was
    fetched with.
    """

    status: str | None = None
    current_period_start: int | None = None
    current_period_end: int | None = None
    price_id: str | None = None


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


def _version_key(customer_id: str) -> str:
    return f"active_subscription_version:{customer_id}"


async def get_subscription_lookup_version(customer_id: str) -> str | None:
    """The customer's current version token, or None when Redis can't be
    read, in which case the caller asks Stripe directly."""
    try:
        redis = await get_redis_async()
        version = await redis.get(_version_key(customer_id))
    except Exception:
        logger.warning(
            f"Could not read the subscription lookup version for {customer_id}",
            exc_info=True,
        )
        return None
    if isinstance(version, bytes):
        version = version.decode()
    return version or _DEFAULT_VERSION


async def invalidate_active_subscription_cache(customer_id: str) -> None:
    """Make every worker read the customer's subscription fresh from Stripe.

    Call it after the change has landed in Stripe, never before: a read
    started earlier could otherwise cache the old state under the new token.
    Never raises, because the change it follows has already been made.
    """
    try:
        redis = await get_redis_async()
        await redis.set(_version_key(customer_id), uuid4().hex, ex=_VERSION_TTL_SECONDS)
    except Exception:
        logger.exception(
            f"Could not invalidate the subscription lookup for {customer_id};"
            f" it can stay stale for {SUBSCRIPTION_LOOKUP_TTL_SECONDS}s"
        )
