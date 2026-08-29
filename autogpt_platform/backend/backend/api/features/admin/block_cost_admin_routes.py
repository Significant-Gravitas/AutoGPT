import logging
import math
import typing
from datetime import datetime, timezone

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, HTTPException, Query, Security
from pydantic import BaseModel

from backend.data.block_cost_analytics import (
    ANALYTICS_MAX_DAYS,
    BlockCostEstimateRow,
    compute_block_cost_estimates,
)

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/admin",
    tags=["admin", "blocks"],
    dependencies=[Security(requires_admin_user)],
)


class BlockCostEstimatesResponse(BaseModel):
    estimates: list[BlockCostEstimateRow]
    total_rows: int
    window_days: int
    max_window_days: int
    min_samples: int
    generated_at: datetime


@router.get(
    "/blocks/cost-estimates",
    response_model=BlockCostEstimatesResponse,
    summary="Export Block Cost Estimates",
)
async def get_block_cost_estimates(
    start: typing.Optional[datetime] = Query(
        None, description="ISO timestamp (inclusive)"
    ),
    end: typing.Optional[datetime] = Query(
        None, description="ISO timestamp (inclusive)"
    ),
    min_samples: int = Query(
        10, ge=1, description="Minimum executions per block to include"
    ),
    admin_user_id: str = Security(get_user_id),
) -> BlockCostEstimatesResponse:
    """Aggregate per-block average credits-per-execution over [start, end].

    Capped at ANALYTICS_MAX_DAYS days. Returns only blocks whose current cost
    type is dynamic (SECOND/ITEMS/COST_USD) — static-cost blocks already
    charge correctly pre-flight and don't need an estimate override. TOKENS
    is excluded because `compute_token_credits` already supplies a per-model
    floor at pre-flight; a per-block historical mean would lose that
    granularity.
    """
    if start is None or end is None:
        raise HTTPException(
            status_code=400, detail="start and end query params are required"
        )
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    logger.info(
        "Admin %s aggregating block cost estimates [%s..%s] min_samples=%s",
        admin_user_id,
        start.isoformat(),
        end.isoformat(),
        min_samples,
    )

    try:
        rows = await compute_block_cost_estimates(
            start=start, end=end, min_samples=min_samples
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Ceil so a [00:00:00Z, 23:59:59.999Z] window — what the frontend sends for
    # an inclusive 7-day pick — reports 7, not 6 (`.days` would truncate).
    window_days = math.ceil((end - start).total_seconds() / 86400)

    return BlockCostEstimatesResponse(
        estimates=rows,
        total_rows=len(rows),
        window_days=window_days,
        max_window_days=ANALYTICS_MAX_DAYS,
        min_samples=min_samples,
        generated_at=datetime.now(timezone.utc),
    )
