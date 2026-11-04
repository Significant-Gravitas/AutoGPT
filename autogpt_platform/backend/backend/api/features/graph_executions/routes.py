import asyncio
from datetime import datetime, timezone
from typing import Annotated

from autogpt_libs.auth import get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction, TeamAction
from fastapi import APIRouter, Body, HTTPException, Path, Query, Response, Security
from starlette.status import HTTP_204_NO_CONTENT, HTTP_404_NOT_FOUND

from backend.api.features.graph_executions import sharing
from backend.api.features.graph_executions.model import (
    ExecutionShareRequest,
    ExecutionShareResponse,
)
from backend.api.features.workspace.routes import create_file_download_response
from backend.api.live_auth import (
    requires_live_resource_permission as requires_resource_permission,
)
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
from backend.data.tenancy import live_resource_permission_barrier
from backend.executor import utils as execution_utils
from backend.util.exceptions import NotFoundError
from backend.util.models import MAX_PAGE

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
    graph_id: str,
    graph_exec_id: str,
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.EXECUTE_RESOURCES, TeamAction.EXECUTE_AGENTS
        ),
    ],
) -> execution_db.GraphExecutionMeta | None:
    res = await _stop_graph_run(
        user_id=user_id,
        graph_id=graph_id,
        graph_exec_id=graph_exec_id,
        organization_id=ctx.org_id,
        team_id_restriction=ctx.team_id,
    )
    if not res:
        return None
    return res[0]


async def _stop_graph_run(
    user_id: str,
    graph_id: str,
    graph_exec_id: str,
    organization_id: str | None = None,
    team_id_restriction: str | None = None,
) -> list[execution_db.GraphExecutionMeta]:
    candidate = await execution_db.get_graph_execution(
        user_id=user_id,
        execution_id=graph_exec_id,
        organization_id=organization_id,
        team_id_restriction=team_id_restriction,
    )
    active_statuses = {
        execution_db.ExecutionStatus.INCOMPLETE,
        execution_db.ExecutionStatus.QUEUED,
        execution_db.ExecutionStatus.RUNNING,
    }
    if (
        candidate is None
        or candidate.user_id != user_id
        or candidate.graph_id != graph_id
        or candidate.status not in active_statuses
    ):
        return []

    async with live_resource_permission_barrier(
        user_id,
        organization_id,
        candidate.team_id,
        OrgAction.EXECUTE_RESOURCES,
        TeamAction.EXECUTE_AGENTS,
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
        execution = await execution_db.get_graph_execution_exact_scope(
            user_id,
            graph_exec_id,
            organization_id,
            candidate.team_id,
        )
        if (
            execution is None
            or execution.user_id != user_id
            or execution.graph_id != graph_id
            or execution.status not in active_statuses
        ):
            return []
        await execution_utils.stop_graph_execution(
            graph_exec_id=execution.id, user_id=user_id
        )
        return [execution]


@router.get(
    path="/executions",
    summary="List all executions",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def list_graphs_executions(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
    ],
) -> list[execution_db.GraphExecutionMeta]:
    paginated_result = await execution_db.get_graph_executions_paginated(
        user_id=user_id,
        page=1,
        page_size=250,
        organization_id=ctx.org_id,
        team_id_restriction=ctx.team_id,
    )

    return paginated_result.executions


@router.get(
    path="/executions/cost-summary",
    summary="User cost summary",
    tags=["graphs"],
    dependencies=[Security(requires_user)],
)
async def get_executions_cost_summary(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
    ],
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
        organization_id=ctx.org_id,
        team_id_restriction=ctx.team_id,
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
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
    ],
    page: int = Query(1, ge=1, le=MAX_PAGE, description="Page number (1-indexed)"),
    page_size: int = Query(
        25, ge=1, le=100, description="Number of executions per page"
    ),
) -> execution_db.GraphExecutionsPaginated:
    paginated_result, onboarding = await asyncio.gather(
        execution_db.get_graph_executions_paginated(
            graph_id=graph_id,
            user_id=user_id,
            page=page,
            page_size=page_size,
            organization_id=ctx.org_id,
            team_id_restriction=ctx.team_id,
        ),
        get_user_onboarding(user_id),
    )
    if (
        onboarding.onboardingAgentExecutionId
        and onboarding.onboardingAgentExecutionId
        in [exec.id for exec in paginated_result.executions]
        and OnboardingStep.GET_RESULTS not in onboarding.completedSteps
    ):
        await complete_onboarding_step(user_id, OnboardingStep.GET_RESULTS)

    return execution_db.GraphExecutionsPaginated(
        executions=paginated_result.executions, pagination=paginated_result.pagination
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
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
    ],
) -> execution_db.GraphExecution | execution_db.GraphExecutionWithNodes:
    result = await execution_db.get_graph_execution(
        user_id=user_id,
        execution_id=graph_exec_id,
        include_node_executions=True,
        organization_id=ctx.org_id,
        team_id_restriction=ctx.team_id,
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
        team_id=ctx.team_id,
    ):
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, detail=f"Graph #{graph_id} not found"
        )

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
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.DELETE_AGENTS
        ),
    ],
) -> None:
    candidate = await execution_db.get_graph_execution(
        user_id=user_id,
        execution_id=graph_exec_id,
        organization_id=ctx.org_id,
        team_id_restriction=ctx.team_id,
    )
    if candidate is None:
        raise HTTPException(status_code=404, detail="Execution not found")
    async with live_resource_permission_barrier(
        user_id,
        ctx.org_id,
        candidate.team_id,
        OrgAction.CREATE_RESOURCES,
        TeamAction.DELETE_AGENTS,
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
        execution = await execution_db.get_graph_execution_exact_scope(
            user_id,
            graph_exec_id,
            ctx.org_id,
            candidate.team_id,
        )
        if execution is None:
            raise HTTPException(status_code=404, detail="Execution not found")
        await execution_db.delete_graph_execution(
            graph_exec_id=graph_exec_id,
            user_id=user_id,
            organization_id=ctx.org_id,
            team_id_restriction=candidate.team_id,
            exact_scope=True,
        )
        # The public file download checks only the allowlist, which would
        # otherwise outlive the run (the same revocation as the v2 API's).
        await execution_db.delete_shared_execution_files(execution_id=graph_exec_id)


@router.post(
    "/graphs/{graph_id}/executions/{graph_exec_id}/share",
    dependencies=[Security(requires_user)],
)
async def enable_execution_sharing(
    graph_id: Annotated[str, Path],
    graph_exec_id: Annotated[str, Path],
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS
        ),
    ],
    _body: ExecutionShareRequest = Body(default=ExecutionShareRequest()),
) -> ExecutionShareResponse:
    """Enable sharing for a graph execution."""
    candidate = await execution_db.get_graph_execution(
        user_id=user_id,
        execution_id=graph_exec_id,
        organization_id=ctx.org_id,
        team_id_restriction=ctx.team_id,
    )
    if not candidate or candidate.graph_id != graph_id:
        raise HTTPException(status_code=404, detail="Execution not found")
    async with live_resource_permission_barrier(
        user_id,
        ctx.org_id,
        candidate.team_id,
        OrgAction.CREATE_RESOURCES,
        TeamAction.CREATE_AGENTS,
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
        execution = await execution_db.get_graph_execution_exact_scope(
            user_id,
            graph_exec_id,
            ctx.org_id,
            candidate.team_id,
        )
        if not execution or execution.graph_id != graph_id:
            raise HTTPException(status_code=404, detail="Execution not found")

        share_token = generate_share_token()
        try:
            await execution_db.replace_execution_share(
                execution_id=graph_exec_id,
                user_id=user_id,
                organization_id=ctx.org_id,
                team_id=candidate.team_id,
                is_shared=True,
                share_token=share_token,
                shared_at=datetime.now(timezone.utc),
                outputs=execution.outputs,
            )
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    return ExecutionShareResponse(
        share_url=sharing.share_url(share_token), share_token=share_token
    )


@router.delete(
    "/graphs/{graph_id}/executions/{graph_exec_id}/share",
    status_code=HTTP_204_NO_CONTENT,
    dependencies=[Security(requires_user)],
)
async def disable_execution_sharing(
    graph_id: Annotated[str, Path],
    graph_exec_id: Annotated[str, Path],
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS
        ),
    ],
) -> None:
    """Disable sharing for a graph execution."""
    candidate = await execution_db.get_graph_execution(
        user_id=user_id,
        execution_id=graph_exec_id,
        organization_id=ctx.org_id,
        team_id_restriction=ctx.team_id,
    )
    if not candidate or candidate.graph_id != graph_id:
        raise HTTPException(status_code=404, detail="Execution not found")
    async with live_resource_permission_barrier(
        user_id,
        ctx.org_id,
        candidate.team_id,
        OrgAction.CREATE_RESOURCES,
        TeamAction.CREATE_AGENTS,
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
        execution = await execution_db.get_graph_execution_exact_scope(
            user_id,
            graph_exec_id,
            ctx.org_id,
            candidate.team_id,
        )
        if not execution or execution.graph_id != graph_id:
            raise HTTPException(status_code=404, detail="Execution not found")

        try:
            await execution_db.replace_execution_share(
                execution_id=graph_exec_id,
                user_id=user_id,
                organization_id=ctx.org_id,
                team_id=candidate.team_id,
                is_shared=False,
                share_token=None,
                shared_at=None,
                outputs=execution.outputs,
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
    async with execution_db.shared_execution_file_access(share_token, file_id) as file:
        if file is None:
            raise HTTPException(status_code=404, detail="Not found")
        return await create_file_download_response(file)
