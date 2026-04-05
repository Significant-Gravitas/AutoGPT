"""Org-level credit operations.

Mirrors the UserCreditBase pattern but operates on OrgBalance and
OrgCreditTransaction tables instead of UserBalance and CreditTransaction.

All balance mutations use atomic SQL to prevent race conditions.
"""

import logging

from prisma.enums import CreditTransactionType
from pydantic import BaseModel

from backend.data.db import prisma
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

    # Atomic deduct — only succeeds if balance >= amount
    rows_affected = await prisma.execute_raw(
        """
        UPDATE "OrgBalance"
        SET "balance" = "balance" - $1, "updatedAt" = NOW()
        WHERE "orgId" = $2 AND "balance" >= $1
        """,
        amount,
        org_id,
    )

    if rows_affected == 0:
        current = await get_org_credits(org_id)
        raise InsufficientBalanceError(
            f"Organization has {current} credits but needs {amount}",
            user_id=user_id,
            balance=current,
            amount=amount,
        )

    # Read the new balance for the transaction record
    new_balance = await get_org_credits(org_id)

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

    # Atomic upsert — INSERT or UPDATE in one statement
    await prisma.execute_raw(
        """
        INSERT INTO "OrgBalance" ("orgId", "balance", "updatedAt")
        VALUES ($1, $2, NOW())
        ON CONFLICT ("orgId")
        DO UPDATE SET "balance" = "OrgBalance"."balance" + $2, "updatedAt" = NOW()
        """,
        org_id,
        amount,
    )

    new_balance = await get_org_credits(org_id)

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
