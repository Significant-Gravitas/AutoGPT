"""
V2 External API - Credits Endpoints

Provides read-only access to credit balance, transaction history,
subscription status, invoices, and execution cost summary.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

import stripe
from fastapi import APIRouter, HTTPException, Query, Security
from prisma.enums import APIKeyPermission, SubscriptionTier
from starlette.concurrency import run_in_threadpool

from backend.api.external.middleware import require_permission
from backend.copilot.rate_limit import get_tier_multipliers
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.credit import (
    PendingChangeUnknown,
    get_active_subscription_period_end,
    get_pending_subscription_change,
    get_proration_credit_cents,
    get_subscription_price_id,
    get_user_billing_cycle,
    get_user_credit_model,
)
from backend.data.execution_cost_summary import get_user_cost_summary
from backend.data.user import get_user_by_id
from backend.util.cache import cached

from .common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from .models import (
    AutomationCostSummary,
    CreditBalance,
    CreditTransaction,
    CreditTransactionsResponse,
    InvoiceItem,
    InvoiceListResponse,
    SubscriptionStatus,
)

logger = logging.getLogger(__name__)

credits_router = APIRouter(tags=["credits"])


# ============================================================================
# Endpoints
# ============================================================================


@credits_router.get(
    path="",
    summary="Get credit balance",
    operation_id="getCreditBalance",
)
async def get_balance(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_CREDITS)
    ),
) -> CreditBalance:
    """Get the current credit balance for the authenticated user."""
    user_credit_model = await get_user_credit_model(auth.user_id)
    balance = await user_credit_model.get_credits(auth.user_id)

    return CreditBalance(balance=balance)


@credits_router.get(
    path="/transactions",
    summary="Get credit transaction history",
    operation_id="listCreditTransactions",
)
async def get_transactions(
    limit: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Items per page (max {MAX_PAGE_SIZE})",
    ),
    cursor: Optional[str] = Query(
        default=None,
        description=(
            "Pagination cursor (ISO datetime from previous response's next_cursor)"
        ),
    ),
    transaction_type: Optional[str] = Query(
        default=None,
        description="Filter by transaction type (TOP_UP, USAGE, GRANT, REFUND)",
    ),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_CREDITS)
    ),
) -> CreditTransactionsResponse:
    """Get credit transaction history for the authenticated user."""
    user_credit_model = await get_user_credit_model(auth.user_id)

    transaction_time_ceiling: datetime | None = None
    if cursor:
        try:
            transaction_time_ceiling = datetime.fromisoformat(cursor)
        except ValueError:
            raise HTTPException(
                status_code=422, detail="Invalid cursor format (expected ISO datetime)"
            )

    history = await user_credit_model.get_transaction_history(
        user_id=auth.user_id,
        transaction_count_limit=limit,
        transaction_time_ceiling=transaction_time_ceiling,
        transaction_type=transaction_type,
    )

    transactions = [CreditTransaction.from_internal(t) for t in history.transactions]
    next_cursor = (
        history.next_transaction_time.isoformat()
        if history.next_transaction_time
        else None
    )

    return CreditTransactionsResponse(
        transactions=transactions,
        next_cursor=next_cursor,
    )


@credits_router.get(
    path="/subscription",
    summary="Get subscription status",
    operation_id="getSubscriptionStatus",
)
async def get_subscription_status(
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_CREDITS)
    ),
) -> SubscriptionStatus:
    """Get the current subscription tier, pricing, and pending changes."""
    user = await get_user_by_id(auth.user_id)
    tier = user.subscription_tier or SubscriptionTier.NO_TIER

    priceable_tiers = [
        SubscriptionTier.BASIC,
        SubscriptionTier.PRO,
        SubscriptionTier.MAX,
        SubscriptionTier.BUSINESS,
    ]
    monthly_price_ids, yearly_price_ids = await asyncio.gather(
        asyncio.gather(
            *[get_subscription_price_id(t, "monthly") for t in priceable_tiers]
        ),
        asyncio.gather(
            *[get_subscription_price_id(t, "yearly") for t in priceable_tiers]
        ),
    )

    monthly_costs, yearly_costs = await asyncio.gather(
        asyncio.gather(*[_get_stripe_price_amount(pid) for pid in monthly_price_ids]),
        asyncio.gather(*[_get_stripe_price_amount(pid) for pid in yearly_price_ids]),
    )

    tier_costs: dict[str, int] = {}
    tier_costs_yearly: dict[str, int] = {}
    for t, m_pid, y_pid, m_cost, y_cost in zip(
        priceable_tiers,
        monthly_price_ids,
        yearly_price_ids,
        monthly_costs,
        yearly_costs,
    ):
        if m_pid or y_pid:
            tier_costs[t.value] = m_cost if m_pid else 0
        if y_pid:
            tier_costs_yearly[t.value] = y_cost

    multipliers = await get_tier_multipliers()
    tier_multipliers: dict[str, float] = {
        t.value: multipliers.get(t.value, 1.0)
        for t in priceable_tiers
        if t.value in tier_costs
    }

    user_cycle = await get_user_billing_cycle(auth.user_id) or "monthly"
    if user_cycle == "yearly":
        current_monthly_cost = tier_costs_yearly.get(tier.value, 0)
    else:
        current_monthly_cost = tier_costs.get(tier.value, 0)

    proration_credit, current_period_end = await asyncio.gather(
        get_proration_credit_cents(auth.user_id, current_monthly_cost),
        get_active_subscription_period_end(auth.user_id),
    )

    try:
        pending = await get_pending_subscription_change(auth.user_id)
    except (stripe.StripeError, PendingChangeUnknown):
        logger.exception(
            "get_subscription_status: failed to resolve pending change for user %s",
            auth.user_id,
        )
        pending = None

    response = SubscriptionStatus(
        tier=tier.value,
        monthly_cost=current_monthly_cost,
        tier_costs=tier_costs,
        tier_costs_yearly=tier_costs_yearly,
        billing_cycle=user_cycle,
        tier_multipliers=tier_multipliers,
        proration_credit_cents=proration_credit,
        has_active_stripe_subscription=current_period_end is not None,
        current_period_end=current_period_end,
    )
    if pending is not None:
        pending_tier_enum, pending_effective_at, pending_cycle = pending
        if pending_tier_enum in (
            SubscriptionTier.NO_TIER,
            SubscriptionTier.BASIC,
            SubscriptionTier.PRO,
            SubscriptionTier.MAX,
            SubscriptionTier.BUSINESS,
        ):
            response.pending_tier = pending_tier_enum.value
            response.pending_tier_effective_at = pending_effective_at
            response.pending_billing_cycle = pending_cycle

    return response


@credits_router.get(
    path="/invoices",
    summary="List Stripe invoices",
    operation_id="listCreditInvoices",
)
async def list_invoices(
    limit: int = Query(24, ge=1, le=100, description="Max invoices to return"),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_CREDITS)
    ),
) -> InvoiceListResponse:
    """Recent Stripe invoices for the current user."""
    user_credit_model = await get_user_credit_model(auth.user_id)
    invoices = await user_credit_model.list_invoices(auth.user_id, limit=limit)

    return InvoiceListResponse(
        invoices=[InvoiceItem.from_internal(inv) for inv in invoices],
    )


@credits_router.get(
    path="/cost-summary",
    summary="Get execution cost summary",
    operation_id="getAutomationCostSummary",
)
async def get_cost_summary(
    since: Optional[datetime] = Query(
        default=None,
        description="Window start (UTC). Defaults to start of current calendar month.",
    ),
    until: Optional[datetime] = Query(
        default=None,
        description="Window end (UTC). Defaults to now.",
    ),
    top_runs_limit: int = Query(
        10,
        ge=1,
        le=50,
        description="Maximum number of top-cost runs to return.",
    ),
    auth: APIAuthorizationInfo = Security(
        require_permission(APIKeyPermission.READ_CREDITS)
    ),
) -> AutomationCostSummary:
    """Aggregated cost breakdown for the user's graph executions."""
    if since is not None and until is not None and since > until:
        raise HTTPException(
            status_code=422,
            detail="`since` must be earlier than or equal to `until`.",
        )
    summary = await get_user_cost_summary(
        user_id=auth.user_id,
        since=since,
        until=until,
        top_runs_limit=top_runs_limit,
    )
    return AutomationCostSummary.from_internal(summary)


# ============================================================================
# Helpers
# ============================================================================


@cached(ttl_seconds=300, maxsize=32, cache_none=False)
async def _get_stripe_price_amount(price_id: str | None) -> int:
    """Return the unit_amount (cents) for a Stripe Price ID, cached 5 minutes."""
    if not price_id:
        return 0
    try:
        price = await run_in_threadpool(stripe.Price.retrieve, price_id)
        return price.unit_amount or 0
    except stripe.StripeError:
        logger.warning(
            "Failed to retrieve Stripe price %s — returning 0 (not cached)",
            price_id,
        )
        return 0
