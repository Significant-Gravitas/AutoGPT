"""Org-level credit operations.

Mirrors the UserCreditBase pattern but operates on OrgBalance and
OrgCreditTransaction tables instead of UserBalance and CreditTransaction.
"""

import logging

from prisma.enums import CreditTransactionType

from backend.data.db import prisma
from backend.util.exceptions import InsufficientBalanceError

logger = logging.getLogger(__name__)


async def get_org_credits(org_id: str) -> int:
    """Get the current credit balance for an organization."""
    balance = await prisma.orgbalance.find_unique(where={"orgId": org_id})
    return balance.balance if balance else 0


async def spend_org_credits(
    org_id: str,
    user_id: str,
    amount: int,
    workspace_id: str | None = None,
    metadata: dict | None = None,
) -> int:
    """Spend credits from the org balance.

    Args:
        org_id: The organization ID.
        user_id: The user initiating the spend.
        amount: The amount to deduct (positive integer).
        workspace_id: Optional workspace context for attribution.
        metadata: Optional metadata for the transaction.

    Returns:
        The remaining balance.

    Raises:
        InsufficientBalanceError: If the org doesn't have enough credits.
    """
    balance = await prisma.orgbalance.find_unique(where={"orgId": org_id})
    current = balance.balance if balance else 0

    if current < amount:
        raise InsufficientBalanceError(
            f"Organization has {current} credits but needs {amount}",
            user_id=user_id,
            balance=current,
            amount=amount,
        )

    new_balance = current - amount

    await prisma.orgbalance.update(
        where={"orgId": org_id},
        data={"balance": new_balance},
    )

    await prisma.orgcredittransaction.create(
        data={
            "orgId": org_id,
            "initiatedByUserId": user_id,
            "workspaceId": workspace_id,
            "amount": -amount,
            "type": CreditTransactionType.USAGE,
            "runningBalance": new_balance,
            "metadata": metadata,
        }
    )

    return new_balance


async def top_up_org_credits(
    org_id: str,
    amount: int,
    user_id: str | None = None,
    metadata: dict | None = None,
) -> int:
    """Add credits to the org balance.

    Returns the new balance.
    """
    balance = await prisma.orgbalance.find_unique(where={"orgId": org_id})
    current = balance.balance if balance else 0
    new_balance = current + amount

    if balance:
        await prisma.orgbalance.update(
            where={"orgId": org_id},
            data={"balance": new_balance},
        )
    else:
        await prisma.orgbalance.create(
            data={"orgId": org_id, "balance": new_balance},
        )

    await prisma.orgcredittransaction.create(
        data={
            "orgId": org_id,
            "initiatedByUserId": user_id,
            "amount": amount,
            "type": CreditTransactionType.TOP_UP,
            "runningBalance": new_balance,
            "metadata": metadata,
        }
    )

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
            "workspaceId": t.workspaceId,
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
