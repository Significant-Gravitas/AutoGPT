"""Paid usage resets: a fresh weekly allowance with the daily limit lifted.

A reset applies to one weekly window of the buyer's billing clock: the one
they are in when it is granted. Until that window's own reset, which does
not move, they have one full weekly limit from zero and no daily limit. It is
use-it-or-lose-it, like the window it belongs to.

Two ways to pay, never mixed:

- **Credits**, when the balance covers the whole price: one database
  transaction takes the credits and grants the reset, so neither can exist
  without the other.
- **Card**, otherwise: a Stripe Checkout session for the whole price. The
  purchase is stored as *pending* and grants nothing. It is granted exactly
  once, when Stripe says the session is paid (its webhook, or the buyer
  returning from Checkout, whichever comes first): the grant is one
  conditional update that only a pending row passes. An abandoned, expired
  or declined payment leaves it pending forever, which is nothing.

A card payment is granted for the window the buyer is in when Stripe confirms
it, since that is when they get it. A session can only be paid while the
window it was opened in lasts (it expires first), so that is the same window
unless Stripe's confirmation itself arrives late.

The buyer's key makes a purchase idempotent: a retry or a double click gets
the one purchase back. Each granted reset is numbered within its window and
the number is unique, so two purchases racing each other can't both be "the
next reset": with credits the second is refused and charged nothing.
"""

import logging
from typing import Literal

from prisma.enums import CreditTransactionType, SubscriptionTier
from prisma.errors import UniqueViolationError
from prisma.models import UsageReset
from pydantic import BaseModel
from redis.exceptions import RedisClusterException, RedisError

from backend.copilot.usage_clock import clock_from_state
from backend.copilot.usage_counters import mark_reset
from backend.data.credit import UsageTransactionMetadata, UserCredit
from backend.data.db import transaction
from backend.data.pro_activation import get_usage_activation_state
from backend.util.exceptions import InsufficientBalanceError
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)

PaidWith = Literal["credits", "card"]
ResetStatus = Literal["pending", "granted"]


class ResetPurchase(BaseModel):
    """One purchase of a usage reset. Times are Unix seconds."""

    id: str
    user_id: str
    # The plan it was bought on, as the tier's name.
    tier: str
    # The price in cents, whichever way it was paid.
    price_credits: int
    paid_with: PaidWith
    status: ResetStatus
    week_start: int | None = None
    week_end: int | None = None
    # Which reset of its window it is (1 for the first), once granted.
    sequence: int | None = None
    granted_at: int | None = None
    checkout_session_id: str | None = None
    payment_intent_id: str | None = None

    @classmethod
    def from_db(cls, row: UsageReset) -> "ResetPurchase":
        return cls(
            id=row.id,
            user_id=row.userId,
            tier=SubscriptionTier(row.tier).value,
            price_credits=row.priceCredits,
            paid_with="card" if row.paidWith == "card" else "credits",
            status="granted" if row.status == "granted" else "pending",
            week_start=row.weekStart,
            week_end=row.weekEnd,
            sequence=row.sequence,
            granted_at=row.grantedAt,
            checkout_session_id=row.stripeCheckoutSessionId,
            payment_intent_id=row.stripePaymentIntentId,
        )


class ResetWindowChanged(Exception):
    """The buyer's weekly window or plan is no longer the one the purchase
    was priced and agreed for (an upgrade or a renewal landed meanwhile)."""


class CreditReset(BaseModel):
    purchase: ResetPurchase
    # The credit balance after the purchase.
    balance: int
    # False when this key had already bought it: nothing was charged now.
    charged: bool


def credit_transaction_key(purchase_key: str) -> str:
    return f"USAGE-RESET-{purchase_key}"


async def find_reset_purchase(user_id: str, purchase_key: str) -> ResetPurchase | None:
    """The purchase this key already made, if it did."""
    row = await UsageReset.prisma().find_first(
        where={"userId": user_id, "purchaseKey": purchase_key}
    )
    return ResetPurchase.from_db(row) if row else None


async def list_reset_purchases(user_id: str, limit: int = 20) -> list[ResetPurchase]:
    """The user's latest purchases, pending ones included, newest first."""
    rows = await UsageReset.prisma().find_many(
        where={"userId": user_id}, order={"createdAt": "desc"}, take=limit
    )
    return [ResetPurchase.from_db(row) for row in rows]


async def buy_reset_with_credits(
    user_id: str,
    purchase_key: str,
    *,
    tier: SubscriptionTier,
    price_credits: int,
    week_start: int,
    week_end: int,
    sequence: int,
    now: int,
) -> CreditReset:
    """Charge ``price_credits`` and grant the window's reset number
    ``sequence``, both or neither.

    Raises ``InsufficientBalanceError`` when the balance doesn't cover it,
    ``UniqueViolationError`` when another purchase already is that reset
    (the window was reset meanwhile), and ``ResetWindowChanged`` when the
    buyer's window or plan is no longer the one given; nothing is written or
    charged then.
    """
    if price_credits <= 0 or week_end <= week_start or sequence <= 0:
        raise ValueError("A usage reset needs a price and a window")
    credits = UserCredit()
    try:
        async with transaction() as tx:
            await _still_in_window(user_id, tier, week_start, week_end, now)
            balance, _ = await credits._add_transaction(
                user_id=user_id,
                amount=-price_credits,
                transaction_type=CreditTransactionType.USAGE,
                transaction_key=credit_transaction_key(purchase_key),
                metadata=SafeJson(
                    UsageTransactionMetadata(reason="Usage reset").model_dump()
                ),
                client=tx,
            )
            row = await tx.usagereset.create(
                data={
                    "userId": user_id,
                    "purchaseKey": purchase_key,
                    "tier": tier,
                    "priceCredits": price_credits,
                    "paidWith": "credits",
                    "status": "granted",
                    "weekStart": week_start,
                    "weekEnd": week_end,
                    "sequence": sequence,
                    "grantedAt": now,
                }
            )
    except (UniqueViolationError, InsufficientBalanceError, ResetWindowChanged):
        # The same key may have committed first: then this is that purchase,
        # whether this attempt met its unique key or, the first having spent
        # the balance, found too few credits left.
        bought = await find_reset_purchase(user_id, purchase_key)
        if bought is None or bought.status != "granted":
            raise
        return CreditReset(
            purchase=bought, balance=await credits.get_credits(user_id), charged=False
        )
    logger.info("A usage reset was bought with credits")
    await announce_reset(user_id)
    return CreditReset(
        purchase=ResetPurchase.from_db(row), balance=balance, charged=True
    )


async def _still_in_window(
    user_id: str, tier: SubscriptionTier, week_start: int, week_end: int, now: int
) -> None:
    """Raise unless the stored clock still puts the buyer in this window on
    this plan. Read last, as the charge begins: an upgrade or renewal that
    landed after the eligibility check would otherwise be charged for a
    reset of a window the limiter has already left."""
    state = await get_usage_activation_state(user_id)
    clock = clock_from_state(state, now)
    same_window = clock is not None and (
        clock.windows.week_start,
        clock.windows.week_end,
    ) == (week_start, week_end)
    if not same_window or state.tier.value != tier.value:
        raise ResetWindowChanged()


async def announce_reset(user_id: str) -> None:
    """Tell every process recording this user's costs that their state just
    changed. Best effort: without it a process may count on the replaced
    counters for up to the minute it remembers a state."""
    try:
        await mark_reset(user_id)
    except (RedisError, RedisClusterException, ConnectionError, OSError):
        logger.warning("Could not announce a granted usage reset")
