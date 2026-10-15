import logging
import typing
from datetime import datetime, timezone

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, Body, HTTPException, Query, Security
from prisma.enums import CreditTransactionType
from pydantic import BaseModel

from backend.data.credit import (
    CREDIT_EXPORT_MAX_DAYS,
    admin_export_user_history,
    admin_get_user_history,
    get_user_credit_model,
)
from backend.data.model import UserTransaction
from backend.data.platform_cost import (
    COPILOT_USAGE_EXPORT_MAX_DAYS,
    CopilotWeeklyUsageRow,
    get_copilot_weekly_usage_for_export,
)
from backend.util.json import SafeJson

from .model import AddUserCreditsResponse, UserHistoryResponse

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/admin",
    tags=["credits", "admin"],
    dependencies=[Security(requires_admin_user)],
)


@router.post(
    "/add_credits", response_model=AddUserCreditsResponse, summary="Add Credits to User"
)
async def add_user_credits(
    user_id: typing.Annotated[str, Body()],
    amount: typing.Annotated[int, Body()],
    comments: typing.Annotated[str, Body()],
    admin_user_id: str = Security(get_user_id),
):
    logger.info(
        f"Admin user {admin_user_id} is adding {amount} credits to user {user_id}"
    )
    user_credit_model = await get_user_credit_model(user_id)
    new_balance, transaction_key = await user_credit_model._add_transaction(
        user_id,
        amount,
        transaction_type=CreditTransactionType.GRANT,
        metadata=SafeJson({"admin_id": admin_user_id, "reason": comments}),
    )
    return {
        "new_balance": new_balance,
        "transaction_key": transaction_key,
    }


@router.get(
    "/users_history",
    response_model=UserHistoryResponse,
    summary="Get All Users History",
)
async def admin_get_all_user_history(
    admin_user_id: str = Security(get_user_id),
    search: typing.Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
    transaction_filter: typing.Optional[CreditTransactionType] = None,
    include_inactive: bool = False,
):
    """ """
    logger.info(f"Admin user {admin_user_id} is getting grant history")

    try:
        resp = await admin_get_user_history(
            page=page,
            page_size=page_size,
            search=search,
            transaction_filter=transaction_filter,
            include_inactive=include_inactive,
        )
        logger.info(f"Admin user {admin_user_id} got {len(resp.history)} grant history")
        return resp
    except Exception as e:
        logger.exception(f"Error getting grant history: {e}")
        raise e


class CreditTransactionsExportResponse(BaseModel):
    transactions: list[UserTransaction]
    total_rows: int
    window_days: int
    max_window_days: int


@router.get(
    "/transactions/export",
    response_model=CreditTransactionsExportResponse,
    summary="Export Credit Transactions",
)
async def export_credit_transactions(
    # Typed Optional so orval generates `Date | null` and the URL builder
    # handles the union correctly.  Both are validated as required below.
    start: typing.Optional[datetime] = Query(
        None, description="ISO timestamp (inclusive)"
    ),
    end: typing.Optional[datetime] = Query(
        None, description="ISO timestamp (inclusive)"
    ),
    transaction_type: typing.Optional[CreditTransactionType] = Query(None),
    user_id: typing.Optional[str] = Query(None),
    include_inactive: bool = Query(
        False,
        description=(
            "Include inactive rows (e.g. abandoned Stripe checkouts). "
            "Off by default so phantom rows aren't surfaced in normal exports."
        ),
    ),
    admin_user_id: str = Security(get_user_id),
) -> CreditTransactionsExportResponse:
    """Export CreditTransaction rows in [start, end].

    Capped at CREDIT_EXPORT_MAX_DAYS days and CREDIT_EXPORT_MAX_ROWS rows;
    over-cap requests fail fast with 400 so callers narrow the window
    instead of receiving silently truncated data.
    """
    if start is None or end is None:
        raise HTTPException(
            status_code=400, detail="start and end query params are required"
        )
    # Coerce naive datetimes to UTC at the boundary so neither the data layer
    # nor the response builder hits a TypeError on (end - start) when callers
    # send mixed naive/aware shapes.
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    logger.info(
        "Admin %s exporting credit transactions [%s..%s] type=%s user=%s incl_inactive=%s",
        admin_user_id,
        start.isoformat(),
        end.isoformat(),
        transaction_type.value if transaction_type else None,
        user_id,
        include_inactive,
    )
    try:
        history = await admin_export_user_history(
            start=start,
            end=end,
            transaction_type=transaction_type,
            user_id=user_id,
            include_inactive=include_inactive,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CreditTransactionsExportResponse(
        transactions=history,
        total_rows=len(history),
        window_days=(end - start).days,
        max_window_days=CREDIT_EXPORT_MAX_DAYS,
    )


class CopilotUsageExportResponse(BaseModel):
    rows: list[CopilotWeeklyUsageRow]
    total_rows: int
    window_days: int
    max_window_days: int


@router.get(
    "/copilot-usage/export",
    response_model=CopilotUsageExportResponse,
    summary="Export Copilot Weekly Usage vs Rate Limit",
)
async def export_copilot_weekly_usage(
    # Typed Optional so orval generates `Date | null` and the URL builder
    # handles the union correctly.  Both are validated as required below.
    start: typing.Optional[datetime] = Query(
        None, description="ISO timestamp (inclusive)"
    ),
    end: typing.Optional[datetime] = Query(
        None, description="ISO timestamp (inclusive)"
    ),
    admin_user_id: str = Security(get_user_id),
) -> CopilotUsageExportResponse:
    """Export per-(user, ISO week) copilot spend with the user's tier limit.

    Uses the same 90-day window cap and 100k row cap as the credit export.
    """
    if start is None or end is None:
        raise HTTPException(
            status_code=400, detail="start and end query params are required"
        )
    # Coerce naive datetimes to UTC at the boundary so neither the data layer
    # nor the response builder hits a TypeError on (end - start) when callers
    # send mixed naive/aware shapes.
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    logger.info(
        "Admin %s exporting copilot weekly usage [%s..%s]",
        admin_user_id,
        start.isoformat(),
        end.isoformat(),
    )
    try:
        rows = await get_copilot_weekly_usage_for_export(start=start, end=end)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return CopilotUsageExportResponse(
        rows=rows,
        total_rows=len(rows),
        window_days=(end - start).days,
        max_window_days=COPILOT_USAGE_EXPORT_MAX_DAYS,
    )
