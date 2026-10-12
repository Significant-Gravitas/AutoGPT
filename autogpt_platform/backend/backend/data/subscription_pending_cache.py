"""The pending subscription change the billing page shows, cached for every
API worker.

A queued downgrade (a Subscription Schedule phase) or cancellation
(``cancel_at_period_end``) is read on every dashboard and billing page load
and takes two Subscription lists plus a Schedule retrieve, so it is cached
briefly; see :mod:`backend.data.versioned_cache` for how a change reaches
every worker on its next read.
"""

from datetime import datetime
from typing import Literal

from prisma.enums import SubscriptionTier
from pydantic import BaseModel

from backend.data.versioned_cache import VersionedCache

PENDING_CHANGE_TTL_SECONDS = 30


class PendingChange(BaseModel):
    tier: SubscriptionTier
    effective_at: datetime
    billing_cycle: Literal["monthly", "yearly"] | None = None


# Keyed by Stripe customer id: the pending change is read from the customer's
# Stripe subscription alone, so every change to it comes from Stripe and
# reaches sync_subscription_from_stripe, which knows the customer.
PENDING_CHANGES: VersionedCache[PendingChange | None] = VersionedCache(
    "pending_subscription_change",
    PendingChange | None,
    ttl_seconds=PENDING_CHANGE_TTL_SECONDS,
)


async def invalidate_pending_subscription_change(customer_id: str) -> None:
    """Make every worker read the customer's pending change fresh from Stripe.
    Call it after the change has landed in Stripe; it never raises."""
    await PENDING_CHANGES.invalidate(customer_id)
