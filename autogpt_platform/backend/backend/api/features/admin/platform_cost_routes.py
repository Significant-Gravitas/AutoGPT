import logging
from datetime import datetime

from autogpt_libs.auth import get_user_id, requires_admin_user
from fastapi import APIRouter, Query, Security
from pydantic import BaseModel

from backend.data.platform_cost import (
    CostLogRow,
    PlatformCostDashboard,
    get_platform_cost_dashboard,
    get_platform_cost_logs,
    get_platform_cost_logs_for_export,
)
from backend.util.models import Pagination

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/platform-costs",
    tags=["platform-cost", "admin"],
    dependencies=[Security(requires_admin_user)],
)


class PlatformCostLogsResponse(BaseModel):
    logs: list[CostLogRow]
    pagination: Pagination


@router.get(
    "/dashboard",
    response_model=PlatformCostDashboard,
    summary="Get Platform Cost Dashboard",
)
async def get_cost_dashboard(
    admin_user_id: str = Security(get_user_id),
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    provider: str | None = Query(None),
    user_id: str | None = Query(None),
    model: str | None = Query(None),
    block_name: str | None = Query(None),
    tracking_type: str | None = Query(None),
    graph_exec_id: str | None = Query(None),
):
    logger.info("Admin %s fetching platform cost dashboard", admin_user_id)
    return await get_platform_cost_dashboard(
        start=start,
        end=end,
        provider=provider,
        user_id=user_id,
        model=model,
        block_name=block_name,
        tracking_type=tracking_type,
        graph_exec_id=graph_exec_id,
    )


@router.get(
    "/logs",
    response_model=PlatformCostLogsResponse,
    summary="Get Platform Cost Logs",
)
async def get_cost_logs(
    admin_user_id: str = Security(get_user_id),
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    provider: str | None = Query(None),
    user_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    model: str | None = Query(None),
    block_name: str | None = Query(None),
    tracking_type: str | None = Query(None),
    graph_exec_id: str | None = Query(None),
):
    logger.info("Admin %s fetching platform cost logs", admin_user_id)
    logs, total = await get_platform_cost_logs(
        start=start,
        end=end,
        provider=provider,
        user_id=user_id,
        page=page,
        page_size=page_size,
        model=model,
        block_name=block_name,
        tracking_type=tracking_type,
        graph_exec_id=graph_exec_id,
    )
    total_pages = (total + page_size - 1) // page_size
    return PlatformCostLogsResponse(
        logs=logs,
        pagination=Pagination(
            total_items=total,
            total_pages=total_pages,
            current_page=page,
            page_size=page_size,
        ),
    )


class PlatformCostExportResponse(BaseModel):
    logs: list[CostLogRow]
    total_rows: int
    truncated: bool


@router.get(
    "/logs/export",
    response_model=PlatformCostExportResponse,
    summary="Export Platform Cost Logs",
)
async def export_cost_logs(
    admin_user_id: str = Security(get_user_id),
    start: datetime | None = Query(None),
    end: datetime | None = Query(None),
    provider: str | None = Query(None),
    user_id: str | None = Query(None),
    model: str | None = Query(None),
    block_name: str | None = Query(None),
    tracking_type: str | None = Query(None),
    graph_exec_id: str | None = Query(None),
):
    logger.info("Admin %s exporting platform cost logs", admin_user_id)
    logs, truncated = await get_platform_cost_logs_for_export(
        start=start,
        end=end,
        provider=provider,
        user_id=user_id,
        model=model,
        block_name=block_name,
        tracking_type=tracking_type,
        graph_exec_id=graph_exec_id,
    )
    return PlatformCostExportResponse(
        logs=logs,
        total_rows=len(logs),
        truncated=truncated,
    )
