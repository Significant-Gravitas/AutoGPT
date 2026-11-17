"""Settling a usage reset's card payment, for whoever learns of it first:
Stripe's webhook, for a completed Checkout session that says it is a
reset's, or the customer returning from Checkout."""

from pydantic import BaseModel

from backend.copilot.usage_mode import now_seconds
from backend.data.usage_reset import ResetPurchase
from backend.data.usage_reset_card import Ungranted, card_reset_outcome


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
    return CardSettlement(purchase=grant.purchase)
