"""Paid extra weeks of usage: bought with credits, used after the weekly limit.

An extra week is its own allowance. It holds the plan's weekly limit as it was
when bought, lasts 30 days from the purchase whatever resets, renewals or plan
changes happen meanwhile, and is only drawn on once the plan's own weekly
limit has run out.

The purchase is one database transaction that takes the credits and creates
the allowance together, so neither can exist without the other. The buyer's
key makes it idempotent: a retry, a double click or two requests racing get
the one purchase back and are charged once.
"""

import logging

from prisma.enums import CreditTransactionType, SubscriptionTier
from prisma.errors import UniqueViolationError
from prisma.models import UsageExtraAllowance
from pydantic import BaseModel

from backend.data.credit import UsageTransactionMetadata, UserCredit
from backend.data.db import query_raw_with_schema, transaction
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)

EXTRA_WEEK_SECONDS = 30 * 86_400


class ExtraWeek(BaseModel):
    """One extra week. Amounts are microdollars, times Unix seconds."""

    id: str
    # The plan it was bought on, as the tier's name.
    tier: str
    amount: int
    used: int
    price_credits: int
    purchased_at: int
    expires_at: int

    @classmethod
    def from_db(cls, row: UsageExtraAllowance) -> "ExtraWeek":
        return cls(
            id=row.id,
            tier=SubscriptionTier(row.tier).value,
            amount=row.amountMicrodollars,
            used=row.usedMicrodollars,
            price_credits=row.priceCredits,
            purchased_at=row.purchasedAt,
            expires_at=row.expiresAt,
        )


class ExtraWeekPurchase(BaseModel):
    extra_week: ExtraWeek
    # The credit balance after the purchase.
    balance: int
    # False when this key had already bought it: nothing was charged now.
    charged: bool


def credit_transaction_key(purchase_key: str) -> str:
    return f"EXTRA-WEEK-{purchase_key}"


async def buy_extra_week(
    user_id: str,
    purchase_key: str,
    *,
    tier: SubscriptionTier,
    amount: int,
    price_credits: int,
    now: int,
) -> ExtraWeekPurchase:
    """Charge ``price_credits`` and create the allowance, both or neither.

    Raises ``InsufficientBalanceError`` when the balance doesn't cover it;
    nothing is written then.
    """
    if amount <= 0 or price_credits <= 0:
        raise ValueError("An extra week needs an amount and a price")
    credits = UserCredit()
    bought = await _find_purchase(user_id, purchase_key)
    if bought is not None:
        return ExtraWeekPurchase(
            extra_week=bought, balance=await credits.get_credits(user_id), charged=False
        )
    try:
        async with transaction() as tx:
            balance, _ = await credits._add_transaction(
                user_id=user_id,
                amount=-price_credits,
                transaction_type=CreditTransactionType.USAGE,
                transaction_key=credit_transaction_key(purchase_key),
                metadata=SafeJson(
                    UsageTransactionMetadata(reason="Extra week of usage").model_dump()
                ),
                client=tx,
            )
            row = await tx.usageextraallowance.create(
                data={
                    "userId": user_id,
                    "purchaseKey": purchase_key,
                    "tier": tier,
                    "amountMicrodollars": amount,
                    "priceCredits": price_credits,
                    "purchasedAt": now,
                    "expiresAt": now + EXTRA_WEEK_SECONDS,
                }
            )
    except UniqueViolationError:
        # The same key committed between the lookup and here: that purchase.
        bought = await _find_purchase(user_id, purchase_key)
        if bought is None:
            raise
        return ExtraWeekPurchase(
            extra_week=bought, balance=await credits.get_credits(user_id), charged=False
        )
    # No user id: the code scanner counts it as private here. The purchase
    # is on record as its credit transaction and its PostHog event.
    logger.info("An extra week was bought")
    return ExtraWeekPurchase(
        extra_week=ExtraWeek.from_db(row), balance=balance, charged=True
    )


async def consume_extra_allowance(user_id: str, amount: int, now: int) -> int:
    """Take up to ``amount`` microdollars from the user's unexpired extra
    weeks, the one expiring first before the others. Returns what was taken.

    One statement: the rows are locked before their balances are read, so two
    costs recorded at once can't both spend the same remainder.
    """
    if amount <= 0:
        return 0
    rows = await query_raw_with_schema(
        """WITH locked AS (
            SELECT "id", "amountMicrodollars" - "usedMicrodollars" AS remaining,
                "expiresAt", "purchasedAt"
            FROM {schema_prefix}"UsageExtraAllowance"
            WHERE "userId" = $1 AND "expiresAt" > $3::bigint
                AND "usedMicrodollars" < "amountMicrodollars"
            ORDER BY "expiresAt", "purchasedAt", "id"
            FOR UPDATE
        ),
        queued AS (
            SELECT "id", remaining,
                COALESCE(SUM(remaining) OVER (
                    ORDER BY "expiresAt", "purchasedAt", "id"
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS ahead
            FROM locked
        ),
        taken AS (
            UPDATE {schema_prefix}"UsageExtraAllowance" e
            SET "usedMicrodollars" = e."usedMicrodollars"
                + LEAST(queued.remaining, $2::bigint - queued.ahead)
            FROM queued
            WHERE e."id" = queued."id" AND queued.ahead < $2::bigint
            RETURNING LEAST(queued.remaining, $2::bigint - queued.ahead) AS amount
        )
        SELECT COALESCE(SUM(amount), 0)::bigint AS absorbed FROM taken""",
        user_id,
        amount,
        now,
    )
    return int(rows[0]["absorbed"]) if rows else 0


async def list_extra_weeks(user_id: str, now: int) -> list[ExtraWeek]:
    """The user's unexpired extra weeks, the one expiring first first."""
    rows = await UsageExtraAllowance.prisma().find_many(
        where={"userId": user_id, "expiresAt": {"gt": now}},
        order=[{"expiresAt": "asc"}, {"purchasedAt": "asc"}],
    )
    return [ExtraWeek.from_db(row) for row in rows]


async def _find_purchase(user_id: str, purchase_key: str) -> ExtraWeek | None:
    row = await UsageExtraAllowance.prisma().find_first(
        where={"userId": user_id, "purchaseKey": purchase_key}
    )
    return ExtraWeek.from_db(row) if row else None
