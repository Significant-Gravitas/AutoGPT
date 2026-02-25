"""
V2 External API - Credits Endpoints

Provides access to credit balance and transaction history.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query, Security
from prisma.enums import APIKeyPermission

from backend.api.external.middleware import require_permission
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.credit import get_user_credit_model

from .common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from .models import CreditBalance, CreditTransaction, CreditTransactionsResponse

logger = logging.getLogger(__name__)

credits_router = APIRouter()


# ============================================================================
# Endpoints
# ============================================================================


@credits_router.get(
    path="",
    summary="Get credit balance",
    response_model=CreditBalance,
)
async def get_balance(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_CREDITS)
    ),
) -> CreditBalance:
    """
    Get the current credit balance for the authenticated user.
    """
    user_credit_model = await get_user_credit_model(auth.user_id)
    balance = await user_credit_model.get_credits(auth.user_id)

    return CreditBalance(balance=balance)


@credits_router.get(
    path="/transactions",
    summary="Get transaction history",
    response_model=CreditTransactionsResponse,
)
async def get_transactions(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_CREDITS)
    ),
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
    transaction_type: Optional[str] = Query(
        default=None,
        description="Filter by transaction type (TOP_UP, USAGE, GRANT, REFUND)",
    ),
) -> CreditTransactionsResponse:
    """
    Get credit transaction history for the authenticated user.

    Returns transactions sorted by most recent first.
    """
    user_credit_model = await get_user_credit_model(auth.user_id)

    history = await user_credit_model.get_transaction_history(
        user_id=auth.user_id,
        transaction_count_limit=page_size,
        transaction_type=transaction_type,
    )

    transactions = [
        CreditTransaction(
            transaction_key=t.transaction_key,
            amount=t.amount,
            type=t.transaction_type.value,
            transaction_time=t.transaction_time,
            running_balance=t.running_balance,
            description=t.description,
        )
        for t in history.transactions
    ]

    # Note: The current credit module doesn't support true pagination,
    # so we're returning what we have
    total_count = len(transactions)
    total_pages = 1  # Without true pagination support

    return CreditTransactionsResponse(
        transactions=transactions,
        total_count=total_count,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )
