from datetime import datetime
from typing import Annotated

from autogpt_libs.auth import get_request_context, get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from fastapi import APIRouter, Header, HTTPException, Query, Response, Security

from backend.api.model import RequestTopUp
from backend.data.credit import (
    AutoTopUpConfig,
    InvoiceListItem,
    RefundRequest,
    TransactionHistory,
    get_auto_top_up,
    get_credit_model,
    set_auto_top_up,
)

# All nine routes carry tags=["credits"] and only Security(requires_user), so
# both live at the mount and the router rather than on each route.
router = APIRouter(dependencies=[Security(requires_user)])


@router.get(
    path="/credits",
    summary="Get user credits",
)
async def get_user_credits(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> dict[str, int]:
    credit_model = await get_credit_model(user_id, ctx.org_id)
    return {"credits": await credit_model.get_credits(user_id)}


@router.post(
    path="/credits",
    summary="Request credit top up",
)
async def request_top_up(
    request: RequestTopUp,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    x_datafast_visitor_id: Annotated[
        str | None, Header(include_in_schema=False)
    ] = None,
    x_datafast_session_id: Annotated[
        str | None, Header(include_in_schema=False)
    ] = None,
):
    credit_model = await get_credit_model(user_id, ctx.org_id)
    checkout_url = await credit_model.top_up_intent(
        user_id,
        request.credit_amount,
        datafast_visitor_id=x_datafast_visitor_id,
        datafast_session_id=x_datafast_session_id,
    )
    return {"checkout_url": checkout_url}


@router.post(
    path="/credits/{transaction_key}/refund",
    summary="Refund credit transaction",
)
async def refund_top_up(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    transaction_key: str,
    metadata: dict[str, str],
) -> int:
    credit_model = await get_credit_model(user_id, ctx.org_id)
    return await credit_model.top_up_refund(user_id, transaction_key, metadata)


@router.patch(
    path="/credits",
    summary="Fulfill checkout session",
)
async def fulfill_checkout(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
):
    credit_model = await get_credit_model(user_id, ctx.org_id)
    await credit_model.fulfill_checkout(user_id=user_id)
    return Response(status_code=200)


@router.post(
    path="/credits/auto-top-up",
    summary="Configure auto top up",
)
async def configure_user_auto_top_up(
    request: AutoTopUpConfig,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> str:
    """Configure auto top-up settings and perform an immediate top-up if needed.

    Raises HTTPException(422) if the request parameters are invalid or if
    the credit top-up fails.
    """
    if request.threshold < 0:
        raise HTTPException(status_code=422, detail="Threshold must be greater than 0")
    if request.amount < 500 and request.amount != 0:
        raise HTTPException(
            status_code=422, detail="Amount must be greater than or equal to 500"
        )
    if request.amount != 0 and request.amount < request.threshold:
        raise HTTPException(
            status_code=422, detail="Amount must be greater than or equal to threshold"
        )

    credit_model = await get_credit_model(user_id, ctx.org_id)
    current_balance = await credit_model.get_credits(user_id)

    try:
        if current_balance < request.threshold:
            await credit_model.top_up_credits(user_id, request.amount)
        else:
            await credit_model.top_up_credits(user_id, 0)
    except NotImplementedError as e:
        raise HTTPException(
            status_code=501, detail="Auto top-up is not available in this context"
        ) from e
    except ValueError as e:
        known_messages = (
            "must not be negative",
            "already exists for user",
            "No payment method found",
        )
        if any(msg in str(e) for msg in known_messages):
            raise HTTPException(status_code=422, detail=str(e))
        raise

    try:
        await set_auto_top_up(
            user_id, AutoTopUpConfig(threshold=request.threshold, amount=request.amount)
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return "Auto top-up settings updated"


@router.get(
    path="/credits/auto-top-up",
    summary="Get auto top up",
)
async def get_user_auto_top_up(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> AutoTopUpConfig:
    return await get_auto_top_up(user_id)


@router.get(
    path="/credits/transactions",
    summary="Get credit history",
)
async def get_credit_history(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    transaction_time: datetime | None = None,
    transaction_type: str | None = None,
    transaction_count_limit: int = 100,
    cursor: str | None = None,
) -> TransactionHistory:
    if transaction_count_limit < 1 or transaction_count_limit > 1000:
        raise ValueError("Transaction count limit must be between 1 and 1000")

    credit_model = await get_credit_model(user_id, ctx.org_id)
    return await credit_model.get_transaction_history(
        user_id=user_id,
        transaction_time_ceiling=transaction_time,
        transaction_count_limit=transaction_count_limit,
        transaction_type=transaction_type,
        cursor=cursor,
        viewer_organization_id=ctx.org_id,
    )


@router.get(
    path="/credits/refunds",
    summary="Get refund requests",
)
async def get_refund_requests(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> list[RefundRequest]:
    credit_model = await get_credit_model(user_id, ctx.org_id)
    return await credit_model.get_refund_requests(user_id)


@router.get(
    path="/credits/invoices",
    summary="List Stripe invoices",
)
async def list_invoices(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    limit: int = Query(24, ge=1, le=100),
) -> list[InvoiceListItem]:
    """Recent Stripe invoices for the current user.

    Each item includes ``hosted_invoice_url`` (Stripe-hosted view) and
    ``invoice_pdf_url`` (direct PDF download). Returns an empty list when
    the credit system is disabled or the user has no Stripe customer yet.
    """
    credit_model = await get_credit_model(user_id, ctx.org_id)
    return await credit_model.list_invoices(user_id, limit=limit)
