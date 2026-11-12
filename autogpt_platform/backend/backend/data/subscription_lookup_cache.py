"""The active-subscription lookup the billing page reads, cached for every
API worker.

The billing status endpoint reads a customer's active subscription several
times per request, on almost every page load, so the lookup is cached briefly
in Redis (see :mod:`backend.data.versioned_cache` for how a change reaches
every worker on its next read).
"""

from typing import Awaitable, Callable

import stripe
from pydantic import BaseModel

from backend.data.versioned_cache import VersionedCache

SUBSCRIPTION_LOOKUP_TTL_SECONDS = 15


class ActiveSubscription(BaseModel):
    """What the billing display reads from the active or trialing subscription.

    Cached instead of the Stripe object, which carries the API key it was
    fetched with.
    """

    status: str | None = None
    current_period_start: int | None = None
    current_period_end: int | None = None
    price_id: str | None = None


# Keyed by Stripe customer id.
ACTIVE_SUBSCRIPTIONS: VersionedCache[ActiveSubscription | None] = VersionedCache(
    "active_subscription",
    ActiveSubscription | None,
    ttl_seconds=SUBSCRIPTION_LOOKUP_TTL_SECONDS,
)


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
    fetch: Callable[[], Awaitable[stripe.Subscription | dict | None]],
) -> ActiveSubscription | None:
    """The cached lookup, or ``fetch``'s answer when it isn't cached."""

    async def lookup() -> ActiveSubscription | None:
        return summarize_subscription(await fetch())

    return await ACTIVE_SUBSCRIPTIONS.read(customer_id, lookup)


async def invalidate_active_subscription_cache(customer_id: str) -> None:
    """Make every worker read the customer's subscription fresh from Stripe.
    Call it after the change has landed in Stripe; it never raises."""
    await ACTIVE_SUBSCRIPTIONS.invalidate(customer_id)
