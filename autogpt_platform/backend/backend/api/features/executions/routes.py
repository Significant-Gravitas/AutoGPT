import asyncio
import logging
from datetime import datetime, timezone
from typing import Annotated, Optional

from autogpt_libs.auth import get_request_context, get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from fastapi import APIRouter, Body, HTTPException, Path, Query, Response, Security
from starlette.status import HTTP_204_NO_CONTENT, HTTP_404_NOT_FOUND

from backend.api.features.executions.activity_gate import (
    hide_activity_summaries_if_disabled,
    hide_activity_summary_if_disabled,
)
from backend.api.features.executions.model import (
    ExecutionShareRequest,
    ExecutionShareResponse,
)
from backend.api.features.workspace.routes import create_file_download_response
from backend.data import execution as execution_db
from backend.data import graph as graph_db
from backend.data.execution_cost_summary import (
    UserExecutionCostSummary,
    get_user_cost_summary,
)
from backend.data.onboarding import (
    OnboardingStep,
    complete_onboarding_step,
    get_user_onboarding,
)
from backend.data.sharing.tokens import SHARE_TOKEN_PATTERN, generate_share_token
from backend.data.workspace import get_workspace_file_by_id
from backend.executor import utils as execution_utils
from backend.util.exceptions import NotFoundError
from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# No router-level auth dependency: the two /public/shared routes are
# deliberately unauthenticated, so each route keeps its own.
router = APIRouter()


@router.post(
    path="/graphs/{graph_id}/executions/{graph_exec_id}/stop",
    summary="Stop graph execution",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def stop_graph_run(
    graph_id: str, graph_exec_id: str, user_id: Annotated[str, Security(get_user_id)]
) -> execution_db.GraphExecutionMeta | None:
    res = await _stop_graph_run(
        user_id=user_id,
        graph_id=graph_id,
        graph_exec_id=graph_exec_id,
    )
    if not res:
        return None
    return res[0]


async def _stop_graph_run(
    user_id: str,
    graph_id: Optional[str] = None,
    graph_exec_id: Optional[str] = None,
) -> list[execution_db.GraphExecutionMeta]:
    graph_execs = await execution_db.get_graph_executions(
        user_id=user_id,
        graph_id=graph_id,
        graph_exec_id=graph_exec_id,
        statuses=[
            execution_db.ExecutionStatus.INCOMPLETE,
            execution_db.ExecutionStatus.QUEUED,
            execution_db.ExecutionStatus.RUNNING,
        ],
    )
    stopped_execs = [
        execution_utils.stop_graph_execution(graph_exec_id=exec.id, user_id=user_id)
        for exec in graph_execs
    ]
    await asyncio.gather(*stopped_execs)
    return graph_execs


@router.get(
    path="/executions",
    summary="List all executions",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def list_graphs_executions(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> list[execution_db.GraphExecutionMeta]:
    paginated_result = await execution_db.get_graph_executions_paginated(
        user_id=user_id,
        page=1,
        page_size=250,
        organization_id=ctx.org_id,
    )

    # Apply feature flags to filter out disabled features
    filtered_executions = await hide_activity_summaries_if_disabled(
        paginated_result.executions, user_id
    )
    return filtered_executions


@router.get(
    path="/executions/cost-summary",
    summary="User cost summary",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def get_executions_cost_summary(
    user_id: Annotated[str, Security(get_user_id)],
    since: datetime | None = Query(
        None,
        description="Window start (UTC). Defaults to start of current calendar month.",
    ),
    until: datetime | None = Query(
        None,
        description="Window end (UTC). Defaults to now.",
    ),
    top_runs_limit: int = Query(
        10,
        ge=1,
        le=50,
        description="Maximum number of top-cost runs to return.",
    ),
) -> UserExecutionCostSummary:
    """Aggregated cost breakdown for the calling user's graph executions."""
    if since is not None and until is not None and since > until:
        raise HTTPException(
            status_code=422,
            detail="`since` must be earlier than or equal to `until`.",
        )
    return await get_user_cost_summary(
        user_id=user_id,
        since=since,
        until=until,
        top_runs_limit=top_runs_limit,
    )


@router.get(
    path="/graphs/{graph_id}/executions",
    summary="List graph executions",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def list_graph_executions(
    graph_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(
        25, ge=1, le=100, description="Number of executions per page"
    ),
) -> execution_db.GraphExecutionsPaginated:
    paginated_result = await execution_db.get_graph_executions_paginated(
        graph_id=graph_id,
        user_id=user_id,
        page=page,
        page_size=page_size,
        organization_id=ctx.org_id,
    )

    # Apply feature flags to filter out disabled features
    filtered_executions = await hide_activity_summaries_if_disabled(
        paginated_result.executions, user_id
    )
    onboarding = await get_user_onboarding(user_id)
    if (
        onboarding.onboardingAgentExecutionId
        and onboarding.onboardingAgentExecutionId
        in [exec.id for exec in filtered_executions]
        and OnboardingStep.GET_RESULTS not in onboarding.completedSteps
    ):
        await complete_onboarding_step(user_id, OnboardingStep.GET_RESULTS)

    return execution_db.GraphExecutionsPaginated(
        executions=filtered_executions, pagination=paginated_result.pagination
    )


@router.get(
    path="/graphs/{graph_id}/executions/{graph_exec_id}",
    summary="Get execution details",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def get_graph_execution(
    graph_id: str,
    graph_exec_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> execution_db.GraphExecution | execution_db.GraphExecutionWithNodes:
    result = await execution_db.get_graph_execution(
        user_id=user_id,
        execution_id=graph_exec_id,
        include_node_executions=True,
        organization_id=ctx.org_id,
    )
    if not result or result.graph_id != graph_id:
        raise HTTPException(
            status_code=404, detail=f"Graph execution #{graph_exec_id} not found."
        )

    if not await graph_db.get_graph(
        graph_id=result.graph_id,
        version=result.graph_version,
        user_id=user_id,
        organization_id=ctx.org_id,
    ):
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail=f"Graph #{graph_id} not found"
        )

    # Apply feature flags to filter out disabled features
    result = await hide_activity_summary_if_disabled(result, user_id)
    onboarding = await get_user_onboarding(user_id)
    if (
        onboarding.onboardingAgentExecutionId == graph_exec_id
        and OnboardingStep.GET_RESULTS not in onboarding.completedSteps
    ):
        await complete_onboarding_step(user_id, OnboardingStep.GET_RESULTS)

    return result


@router.delete(
    path="/executions/{graph_exec_id}",
    summary="Delete graph execution",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
    status_code=HTTP_204_NO_CONTENT,
)
async def delete_graph_execution(
    graph_exec_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> None:
    await execution_db.delete_graph_execution(
        graph_exec_id=graph_exec_id, user_id=user_id
    )


@router.post(
    "/graphs/{graph_id}/executions/{graph_exec_id}/share",
    dependencies=[Security(requires_user)],
)
async def enable_execution_sharing(
    graph_id: Annotated[str, Path],
    graph_exec_id: Annotated[str, Path],
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    _body: ExecutionShareRequest = Body(default=ExecutionShareRequest()),
) -> ExecutionShareResponse:
    """Enable sharing for a graph execution."""
    # Verify the execution belongs to the user
    execution = await execution_db.get_graph_execution(
        user_id=user_id, execution_id=graph_exec_id
    )
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    # Generate a unique share token
    share_token = generate_share_token()

    # Remove stale allowlist records before updating the token — prevents a
    # window where old records + new token could coexist.
    await execution_db.delete_shared_execution_files(execution_id=graph_exec_id)

    # Update the execution with share info — the underlying update_many
    # also enforces (id, user_id) at the DB layer, so a TOCTOU delete
    # between the pre-check above and this write surfaces as 404 rather
    # than a silent no-op.
    try:
        await execution_db.update_graph_execution_share_status(
            execution_id=graph_exec_id,
            user_id=user_id,
            is_shared=True,
            share_token=share_token,
            shared_at=datetime.now(timezone.utc),
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Create allowlist of workspace files referenced in outputs
    await execution_db.create_shared_execution_files(
        execution_id=graph_exec_id,
        share_token=share_token,
        user_id=user_id,
        outputs=execution.outputs,
    )

    # Return the share URL
    frontend_url = settings.config.frontend_base_url or "http://localhost:3000"
    share_url = f"{frontend_url}/share/{share_token}"

    return ExecutionShareResponse(share_url=share_url, share_token=share_token)


@router.delete(
    "/graphs/{graph_id}/executions/{graph_exec_id}/share",
    status_code=HTTP_204_NO_CONTENT,
    dependencies=[Security(requires_user)],
)
async def disable_execution_sharing(
    graph_id: Annotated[str, Path],
    graph_exec_id: Annotated[str, Path],
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> None:
    """Disable sharing for a graph execution."""
    # Verify the execution belongs to the user
    execution = await execution_db.get_graph_execution(
        user_id=user_id, execution_id=graph_exec_id
    )
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found")

    # Remove shared file allowlist records
    await execution_db.delete_shared_execution_files(execution_id=graph_exec_id)

    # Remove share info — owner-gated at the DB layer; TOCTOU delete
    # after the pre-check surfaces as 404.
    try:
        await execution_db.update_graph_execution_share_status(
            execution_id=graph_exec_id,
            user_id=user_id,
            is_shared=False,
            share_token=None,
            shared_at=None,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/public/shared/{share_token}")
async def get_shared_execution(
    share_token: Annotated[
        str,
        Path(pattern=SHARE_TOKEN_PATTERN),
    ],
) -> execution_db.SharedExecutionResponse:
    """Get a shared graph execution by share token (no auth required)."""
    execution = await execution_db.get_graph_execution_by_share_token(share_token)
    if not execution:
        raise HTTPException(status_code=404, detail="Shared execution not found")

    return execution


@router.get(
    "/public/shared/{share_token}/files/{file_id}/download",
    summary="Download a file from a shared execution",
    operation_id="download_shared_file",
    tags=["graphs"],
)
async def download_shared_file(
    share_token: Annotated[
        str,
        Path(pattern=SHARE_TOKEN_PATTERN),
    ],
    file_id: Annotated[
        str,
        Path(pattern=SHARE_TOKEN_PATTERN),
    ],
) -> Response:
    """Download a workspace file from a shared execution (no auth required).

    Validates that the file was explicitly exposed when sharing was enabled.
    Returns a uniform 404 for all failure modes to prevent enumeration attacks.
    """
    # Single-query validation against the allowlist
    execution_id = await execution_db.get_shared_execution_file(
        share_token=share_token, file_id=file_id
    )
    if not execution_id:
        raise HTTPException(status_code=404, detail="Not found")

    # Look up the actual file (no workspace scoping needed — the allowlist
    # already validated that this file belongs to the shared execution)
    file = await get_workspace_file_by_id(file_id)
    if not file:
        raise HTTPException(status_code=404, detail="Not found")

    return await create_file_download_response(file)
