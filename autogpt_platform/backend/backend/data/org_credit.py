"""Org-level credit operations.

Mirrors the UserCreditBase pattern but operates on OrgBalance and
OrgCreditTransaction tables instead of UserBalance and CreditTransaction.

All balance mutations use atomic SQL to prevent race conditions.
"""

import logging
from datetime import datetime, timezone

import stripe
from prisma.enums import CreditTransactionType, OnboardingStep
from pydantic import BaseModel

from backend.data.credit import (
    InvoiceListItem,
    UsageTransactionMetadata,
    UserCreditBase,
)
from backend.data.db import prisma
from backend.data.model import RefundRequest, TransactionHistory
from backend.util.exceptions import InsufficientBalanceError
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)


class _BalanceResult(BaseModel):
    balance: int


async def get_org_credits(org_id: str) -> int:
    """Get the current credit balance for an organization."""
    balance = await prisma.orgbalance.find_unique(where={"orgId": org_id})
    return balance.balance if balance else 0


async def spend_org_credits(
    org_id: str,
    user_id: str,
    amount: int,
    team_id: str | None = None,
    metadata: dict | None = None,
) -> int:
    """Atomically spend credits from the org balance.

    Uses a single UPDATE ... WHERE balance >= $amount to prevent race
    conditions. If the UPDATE affects 0 rows, the balance is insufficient.

    Returns:
        The remaining balance.

    Raises:
        InsufficientBalanceError: If the org doesn't have enough credits.
    """
    if amount <= 0:
        raise ValueError("Spend amount must be positive")

    # Atomic deduct — only succeeds if balance >= amount.
    # Uses RETURNING to get the new balance in the same statement,
    # avoiding a stale read from a separate query.
    result = await prisma.query_raw(
        """
        UPDATE "OrgBalance"
        SET "balance" = "balance" - $1, "updatedAt" = NOW()
        WHERE "orgId" = $2 AND "balance" >= $1
        RETURNING "balance"
        """,
        amount,
        org_id,
    )

    if not result:
        # No row matched — insufficient balance
        current = await get_org_credits(org_id)
        raise InsufficientBalanceError(
            f"Organization has {current} credits but needs {amount}",
            user_id=user_id,
            balance=current,
            amount=amount,
        )

    new_balance = result[0]["balance"]

    # Record the transaction
    tx_data: dict = {
        "orgId": org_id,
        "initiatedByUserId": user_id,
        "amount": -amount,
        "type": CreditTransactionType.USAGE,
        "runningBalance": new_balance,
    }
    if team_id:
        tx_data["teamId"] = team_id
    if metadata:
        tx_data["metadata"] = SafeJson(metadata)

    await prisma.orgcredittransaction.create(data=tx_data)

    return new_balance


async def top_up_org_credits(
    org_id: str,
    amount: int,
    user_id: str | None = None,
    metadata: dict | None = None,
) -> int:
    """Atomically add credits to the org balance.

    Creates the OrgBalance row if it doesn't exist (upsert pattern via raw SQL).

    Returns the new balance.
    """
    if amount <= 0:
        raise ValueError("Top-up amount must be positive")

    # Atomic upsert — INSERT or UPDATE in one statement.
    # Uses RETURNING to get the new balance in the same statement,
    # avoiding a stale read from a separate query.
    result = await prisma.query_raw(
        """
        INSERT INTO "OrgBalance" ("orgId", "balance", "updatedAt")
        VALUES ($1, $2, NOW())
        ON CONFLICT ("orgId")
        DO UPDATE SET "balance" = "OrgBalance"."balance" + $2, "updatedAt" = NOW()
        RETURNING "balance"
        """,
        org_id,
        amount,
    )

    new_balance = result[0]["balance"]

    # Record the transaction
    tx_data: dict = {
        "orgId": org_id,
        "amount": amount,
        "type": CreditTransactionType.TOP_UP,
        "runningBalance": new_balance,
    }
    if user_id:
        tx_data["initiatedByUserId"] = user_id
    if metadata:
        tx_data["metadata"] = SafeJson(metadata)

    await prisma.orgcredittransaction.create(data=tx_data)

    return new_balance


async def get_org_transaction_history(
    org_id: str,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Get credit transaction history for an organization."""
    transactions = await prisma.orgcredittransaction.find_many(
        where={"orgId": org_id, "isActive": True},
        order={"createdAt": "desc"},
        take=limit,
        skip=offset,
    )
    return [
        {
            "transactionKey": t.transactionKey,
            "createdAt": t.createdAt,
            "amount": t.amount,
            "type": t.type,
            "runningBalance": t.runningBalance,
            "initiatedByUserId": t.initiatedByUserId,
            "teamId": t.teamId,
            "metadata": t.metadata,
        }
        for t in transactions
    ]


async def get_seat_info(org_id: str) -> dict:
    """Get seat utilization for an organization."""
    seats = await prisma.organizationseatassignment.find_many(
        where={"organizationId": org_id}
    )
    total = len(seats)
    active = sum(1 for s in seats if s.status == "ACTIVE")
    return {
        "total": total,
        "active": active,
        "inactive": total - active,
        "seats": [
            {
                "userId": s.userId,
                "seatType": s.seatType,
                "status": s.status,
                "createdAt": s.createdAt,
            }
            for s in seats
        ],
    }


async def assign_seat(
    org_id: str,
    user_id: str,
    seat_type: str = "FREE",
    assigned_by: str | None = None,
) -> dict:
    """Assign a seat to a user in the organization."""
    seat = await prisma.organizationseatassignment.upsert(
        where={
            "organizationId_userId": {
                "organizationId": org_id,
                "userId": user_id,
            }
        },
        data={
            "create": {
                "organizationId": org_id,
                "userId": user_id,
                "seatType": seat_type,
                "status": "ACTIVE",
                "assignedByUserId": assigned_by,
            },
            "update": {
                "seatType": seat_type,
                "status": "ACTIVE",
                "assignedByUserId": assigned_by,
            },
        },
    )
    return {
        "userId": seat.userId,
        "seatType": seat.seatType,
        "status": seat.status,
    }


async def unassign_seat(org_id: str, user_id: str) -> None:
    """Deactivate a user's seat assignment."""
    await prisma.organizationseatassignment.update(
        where={
            "organizationId_userId": {
                "organizationId": org_id,
                "userId": user_id,
            }
        },
        data={"status": "INACTIVE"},
    )


class OrgCreditModel(UserCreditBase):
    """Credit model that routes billing operations to the org-level tables.

    Wraps the standalone org credit functions so that billing routes can
    transparently operate on org credits when an ``organization_id`` is
    present in the request context.
    """

    def __init__(self, org_id: str):
        self._org_id = org_id

    async def get_credits(
        self, user_id: str, organization_id: str | None = None
    ) -> int:
        return await get_org_credits(self._org_id)

    async def get_transaction_history(
        self,
        user_id: str,
        transaction_count_limit: int,
        transaction_time_ceiling: datetime | None = None,
        transaction_type: str | None = None,
    ) -> TransactionHistory:
        raw = await get_org_transaction_history(
            self._org_id,
            limit=transaction_count_limit,
            offset=0,
        )
        from backend.data.model import CreditTransactionItem

        # TransactionHistory expects CreditTransactionItem; running_balance
        # lives on UserTransaction but isn't part of this DTO and is dropped.
        transactions = [
            CreditTransactionItem(
                user_id=user_id,
                amount=t["amount"],
                transaction_type=t.get("type", CreditTransactionType.USAGE),
                transaction_key=t.get("transactionKey", ""),
                description=f"{t.get('type', 'UNKNOWN')} Transaction",
            )
            for t in raw
        ]
        return TransactionHistory(
            transactions=transactions,
            next_transaction_time=None,
        )

    async def get_refund_requests(self, user_id: str) -> list[RefundRequest]:
        return []

    async def spend_credits(
        self,
        user_id: str,
        cost: int,
        metadata: UsageTransactionMetadata,
        fail_insufficient_credits: bool = True,
    ) -> int:
        return await spend_org_credits(
            self._org_id, user_id, cost, metadata=metadata.model_dump()
        )

    async def top_up_credits(
        self,
        user_id: str,
        amount: int,
        organization_id: str | None = None,
        **kwargs,
    ):
        await top_up_org_credits(self._org_id, amount, user_id=user_id)

    async def grant_credits(
        self,
        user_id: str,
        amount: int,
        reason: str,
        transaction_key: str | None = None,
    ) -> int:
        return await top_up_org_credits(
            self._org_id, amount, user_id=user_id, metadata={"reason": reason}
        )

    async def onboarding_reward(
        self, user_id: str, credits: int, step: OnboardingStep
    ) -> bool:
        return False

    async def top_up_intent(self, user_id: str, amount: int) -> str:
        raise NotImplementedError("Org-level Stripe top-up intent not yet implemented")

    async def top_up_refund(
        self, user_id: str, transaction_key: str, metadata: dict[str, str]
    ) -> int:
        raise NotImplementedError("Org-level top-up refund not yet implemented")

    async def deduct_credits(self, request: stripe.Refund | stripe.Dispute):
        raise NotImplementedError("Org-level credit deduction not yet implemented")

    async def handle_dispute(self, dispute: stripe.Dispute):
        raise NotImplementedError("Org-level dispute handling not yet implemented")

    async def fulfill_checkout(
        self, *, session_id: str | None = None, user_id: str | None = None
    ):
        raise NotImplementedError("Org-level checkout fulfillment not yet implemented")

    async def list_invoices(
        self, user_id: str, limit: int = 24
    ) -> list[InvoiceListItem]:
        org = await prisma.organization.find_unique(where={"id": self._org_id})
        if not org or not org.stripeCustomerId:
            return []

        from fastapi.concurrency import run_in_threadpool

        limit = max(1, min(limit, 100))
        try:
            invoices = await run_in_threadpool(
                stripe.Invoice.list,
                customer=org.stripeCustomerId,
                limit=limit,
            )
        except stripe.StripeError:
            logger.exception("Stripe invoice list failed for org %s", self._org_id)
            return []

        return [
            InvoiceListItem(
                id=invoice.id or "",
                number=invoice.number,
                created_at=datetime.fromtimestamp(
                    invoice.created or 0,
                    tz=timezone.utc,
                ),
                total_cents=invoice.total or 0,
                amount_paid_cents=invoice.amount_paid or 0,
                currency=(invoice.currency or "usd").lower(),
                status=invoice.status or "open",
                description=invoice.description,
                hosted_invoice_url=invoice.hosted_invoice_url,
                invoice_pdf_url=invoice.invoice_pdf,
            )
            for invoice in invoices.data
        ]
