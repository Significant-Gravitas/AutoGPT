"""One-shot claims for billing emails.

Stripe replays webhooks, and several of these events can legitimately arrive
more than once (a retried delivery, an out-of-order subscription update). Each
billing email therefore claims a key derived from the resource it is about —
the invoice, or the subscription plus its period — so a replay is a no-op
rather than a second email in the customer's inbox.

Falls open on a Redis failure: a rare duplicate is a better outcome than
dropping a payment-failed notice entirely.
"""

import logging

from backend.data.redis_client import get_redis_async

logger = logging.getLogger(__name__)

# Comfortably longer than Stripe's retry window, and long enough that an
# out-of-order redelivery days later still finds the claim.
CLAIM_TTL_SECONDS = 60 * 60 * 24 * 30
_PREFIX = "lifecycle_email"


async def claim_once(key: str) -> bool:
    """Returns True for the first caller with this key, False afterwards."""
    try:
        redis_client = await get_redis_async()
        claimed = await redis_client.set(
            f"{_PREFIX}:{key}", "1", nx=True, ex=CLAIM_TTL_SECONDS
        )
        return bool(claimed)
    except Exception:
        logger.warning(
            "Lifecycle email dedupe failed for %s; sending anyway", key, exc_info=True
        )
        return True
