"""
Admin routes for system diagnostics and monitoring.
"""

import logging

from autogpt_libs.auth import requires_admin_user
from fastapi import APIRouter, HTTPException, Query, Security

from backend.data.diagnostics import (
    ExecutionDiagnostics,
    RunningExecutionDetails,
    get_execution_diagnostics,
    get_running_executions_details,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin/diagnostics",
    tags=["diagnostics", "admin"],
    dependencies=[Security(requires_admin_user)],
)


@router.get(
    "/executions/running",
    response_model=list[RunningExecutionDetails],
    summary="List Running Executions",
)
async def list_running_executions(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """
    Get a list of currently running or queued executions with detailed information.

    Args:
        limit: Maximum number of executions to return (1-100)
        offset: Number of executions to skip for pagination

    Returns:
        List of running executions with details
    """
    try:
        logger.info(f"Listing running executions (limit={limit}, offset={offset})")

        executions = await get_running_executions_details(limit=limit, offset=offset)

        # Get total count for pagination
        from backend.data.diagnostics import get_execution_diagnostics as get_diag

        diagnostics = await get_diag()
        total_count = diagnostics.total_running + diagnostics.total_queued

        logger.info(
            f"Found {len(executions)} running executions (total: {total_count})"
        )

        return executions

    except Exception as e:
        logger.error(f"Error listing running executions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing running executions: {str(e)}",
        )


@router.get(
    "/executions/stats",
    response_model=ExecutionDiagnostics,
    summary="Get Execution Statistics",
)
async def get_execution_stats():
    """
    Get overall statistics about execution statuses.

    Returns:
        Execution diagnostics with counts by status
    """
    try:
        logger.info("Getting execution statistics")
        diagnostics = await get_execution_diagnostics()
        logger.info(
            f"Execution stats - Running: {diagnostics.total_running}, "
            f"Queued: {diagnostics.total_queued}, "
            f"Incomplete: {diagnostics.total_incomplete}"
        )
        return diagnostics

    except Exception as e:
        logger.error(f"Error getting execution statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting execution statistics: {str(e)}",
        )