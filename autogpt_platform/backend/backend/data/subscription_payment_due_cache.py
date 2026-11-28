"""A failed renewal's open bill, as the billing status shows it, cached for
every API worker.

The status endpoint reads it for every NO_TIER customer on every page load,
and the paywall polls that endpoint while a bill is open, so the two to four
Subscription lists behind it are cached briefly (see
:mod:`backend.data.versioned_cache` for how a change reaches every worker on
its next read). Paying the bill changes the subscription, and every such
change reaches ``invalidate_billing_lookups``, which drops this too.
"""

from pydantic import BaseModel

from backend.data.versioned_cache import VersionedCache

PAYMENT_DUE_TTL_SECONDS = 30


class SubscriptionPaymentDue(BaseModel):
    amount_due: int
    currency: str
    payment_url: str
    plan: str | None = None
    due_since: int | None = None


# Keyed by Stripe customer id.
PAYMENTS_DUE: VersionedCache[SubscriptionPaymentDue | None] = VersionedCache(
    "subscription_payment_due",
    SubscriptionPaymentDue | None,
    ttl_seconds=PAYMENT_DUE_TTL_SECONDS,
)


async def invalidate_subscription_payment_due(customer_id: str) -> None:
    """Make every worker read the customer's open bill fresh from Stripe.
    Call it after the change has landed in Stripe; it never raises."""
    await PAYMENTS_DUE.invalidate(customer_id)
