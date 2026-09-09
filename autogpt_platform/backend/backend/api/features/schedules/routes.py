from typing import Annotated, Any

from autogpt_libs.auth import get_request_context, get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from fastapi import APIRouter, Body, HTTPException, Path, Security
from starlette.status import HTTP_404_NOT_FOUND

from backend.api.features.experts import experts_db
from backend.api.features.schedules.model import ScheduleCreationRequest
from backend.data import graph as graph_db
from backend.data.onboarding import OnboardingStep, complete_onboarding_step
from backend.data.tenancy import get_user_team_ids
from backend.data.user import get_user_by_id
from backend.executor import scheduler
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import NotFoundError
from backend.util.timezone_utils import (
    convert_utc_time_to_user_timezone,
    get_user_timezone_or_utc,
)

# This router mounts at /api and keeps the section's full paths: its five routes
# span two prefixes (/schedules and /graphs/{graph_id}/schedules), so a prefixed
# mount would have to move one of them.
router = APIRouter(dependencies=[Security(requires_user)])


@router.post(
    path="/graphs/{graph_id}/schedules",
    summary="Create execution schedule",
)
async def create_graph_execution_schedule(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    graph_id: str = Path(..., description="ID of the graph to schedule"),
    schedule_params: ScheduleCreationRequest = Body(),
) -> scheduler.GraphExecutionJobInfo:
    graph = await graph_db.get_graph(
        graph_id=graph_id,
        version=schedule_params.graph_version,
        user_id=user_id,
    )
    if not graph:
        raise HTTPException(
            status_code=404,
            detail=f"Graph #{graph_id} v{schedule_params.graph_version} not found.",
        )

    # Use timezone from request if provided, otherwise fetch from user profile
    if schedule_params.timezone:
        user_timezone = schedule_params.timezone
    else:
        user = await get_user_by_id(user_id)
        user_timezone = get_user_timezone_or_utc(user.timezone if user else None)

    # Expert attribution: explicit expert_id must be an active expert owned
    # by the caller; when omitted, a unique (user, graph) → expert match
    # keeps attribution for schedules created through the generic UI.
    expert_id = schedule_params.expert_id
    if expert_id is not None:
        expert = await experts_db.get_expert(
            user_id, expert_id, include_workflows=False
        )
        if expert is None or expert.is_archived:
            raise HTTPException(
                status_code=404, detail=f"Expert #{expert_id} not found."
            )
    else:
        expert_id = await experts_db.resolve_expert_for_graph(user_id, graph_id)

    result = await get_scheduler_client().add_execution_schedule(
        user_id=user_id,
        graph_id=graph_id,
        graph_version=graph.version,
        name=schedule_params.name,
        cron=schedule_params.cron,
        input_data=schedule_params.inputs,
        input_credentials=schedule_params.credentials,
        user_timezone=user_timezone,
        organization_id=ctx.org_id,
        team_id=ctx.team_id,
        expert_id=expert_id,
    )

    # Convert the next_run_time back to user timezone for display
    if result.next_run_time:
        result.next_run_time = convert_utc_time_to_user_timezone(
            result.next_run_time, user_timezone
        )

    await complete_onboarding_step(user_id, OnboardingStep.SCHEDULE_AGENT)

    return result


@router.get(
    path="/graphs/{graph_id}/schedules",
    summary="List execution schedules for a graph",
)
async def list_graph_execution_schedules(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    graph_id: str = Path(),
) -> list[scheduler.GraphExecutionJobInfo]:
    team_ids = await get_user_team_ids(user_id, ctx.org_id) if ctx.org_id else []
    return await get_scheduler_client().get_graph_execution_schedules(
        user_id=user_id,
        graph_id=graph_id,
        organization_id=ctx.org_id,
        team_ids=team_ids,
    )


@router.get(
    path="/schedules",
    summary="List execution schedules for a user",
)
async def list_all_graphs_execution_schedules(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> list[scheduler.GraphExecutionJobInfo]:
    team_ids = await get_user_team_ids(user_id, ctx.org_id) if ctx.org_id else []
    return await get_scheduler_client().get_graph_execution_schedules(
        user_id=user_id,
        organization_id=ctx.org_id,
        team_ids=team_ids,
    )


# Keep above /schedules/{schedule_id}: a GET added there would otherwise match
# "followups" as an id, since FastAPI answers with the first route that fits.
@router.get(
    path="/schedules/followups",
    summary="List copilot follow-up schedules for a user",
    operation_id="listCopilotFollowupSchedules",
)
async def list_copilot_turn_schedules(
    user_id: Annotated[str, Security(get_user_id)],
) -> list[scheduler.CopilotTurnJobInfo]:
    """Return only copilot-turn schedules for the current user.

    Sibling of :func:`list_all_graphs_execution_schedules`; one route per kind
    keeps the generated frontend client typed to a single concrete return type
    instead of a discriminated union.
    """
    schedules = await get_scheduler_client().get_execution_schedules(
        user_id=user_id, kind="copilot_turn"
    )
    # Defensive isinstance filter mirrors ``get_graph_execution_schedules``
    # (executor.scheduler.Scheduler) — the scheduler is the source of truth
    # for the ``kind`` filter, but we narrow the polymorphic
    # ``list[GraphExecutionJobInfo | CopilotTurnJobInfo]`` to the typed
    # subset before returning so the generated frontend client gets a single
    # concrete schema. If a row ever slips through the discriminator (e.g.
    # legacy untyped row, scheduler-side bug), we drop it rather than fail
    # the response with a Pydantic validation error.
    return [s for s in schedules if isinstance(s, scheduler.CopilotTurnJobInfo)]


@router.delete(
    path="/schedules/{schedule_id}",
    summary="Delete execution schedule",
)
async def delete_graph_execution_schedule(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[RequestContext, Security(get_request_context)],
    schedule_id: str = Path(..., description="ID of the schedule to delete"),
) -> dict[str, Any]:
    try:
        await get_scheduler_client().delete_schedule(schedule_id, user_id=user_id)
    except NotFoundError:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Schedule #{schedule_id} not found",
        )
    return {"id": schedule_id}
