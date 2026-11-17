"""Settling a usage reset's card payment: grant it once, and count it once.

Asked by whoever learns of the payment first: Stripe's webhook, for a
completed Checkout session that says it is a reset's, or the customer
returning from Checkout. Granting is ``usage_reset_card.card_reset_outcome``;
this adds the analytics event, sent by the one call that granted.
"""

from pydantic import BaseModel

from backend.copilot.usage_mode import now_seconds
from backend.data.usage_reset import ResetPurchase
from backend.data.usage_reset_card import Ungranted, card_reset_outcome
from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent


class CardSettlement(BaseModel):
    # The granted purchase, now or earlier; None when nothing was granted.
    purchase: ResetPurchase | None = None
    # Why nothing was granted, when nothing was.
    ungranted: Ungranted | None = None


async def settle_card_reset(
    session_id: str, *, user_id: str | None = None
) -> CardSettlement:
    """Grant the reset a Checkout session paid for, once. With ``user_id``
    only that user's own session counts."""
    grant, ungranted = await card_reset_outcome(
        session_id, now_seconds(), user_id=user_id
    )
    if grant is None:
        return CardSettlement(ungranted=ungranted)
    if grant.granted_now:
        purchase = grant.purchase
        posthog_client.capture(
            purchase.user_id,
            PostHogEvent.USAGE_RESET_PURCHASED,
            {
                "tier": purchase.tier,
                "price_credits": purchase.price_credits,
                "paid_with": "card",
            },
        )
    return CardSettlement(purchase=grant.purchase)
