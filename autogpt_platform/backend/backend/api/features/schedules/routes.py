import logging
from typing import Annotated, Any

from autogpt_libs.auth import get_user_id, requires_user
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction, TeamAction
from fastapi import APIRouter, Body, Depends, HTTPException, Path, Security
from prisma.enums import GrantCapability
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_503_SERVICE_UNAVAILABLE,
)

from backend.api.features.experts import experts_db
from backend.api.features.library import db as library_db
from backend.api.features.schedule_visibility import visible_graph_schedules
from backend.api.features.schedules.model import (
    ResumeScheduledChatResponse,
    ScheduleCreationRequest,
    ScheduleUpdateRequest,
    SetScheduledChatCadenceRequest,
)
from backend.api.live_auth import (
    requires_live_resource_permission as requires_resource_permission,
)
from backend.api.live_auth import resolve_write_team_id as _resolve_write_team_id
from backend.copilot.scheduled_chats import cadence_service, forget
from backend.copilot.scheduled_chats import service as scheduled_chats
from backend.data import graph as graph_db
from backend.data.entitlement import enforce_payment_paywall
from backend.data.grants import resolve_graph_grant
from backend.data.onboarding import OnboardingStep, complete_onboarding_step
from backend.data.tenancy import (
    get_user_team_ids,
    live_resource_access_barrier,
    live_resource_permission_barrier,
)
from backend.data.user import get_user_by_id
from backend.executor import scheduler
from backend.util.clients import get_scheduler_client
from backend.util.exceptions import NotAuthorizedError, NotFoundError
from backend.util.timezone_utils import (
    convert_utc_time_to_user_timezone,
    get_user_timezone_or_utc,
)

logger = logging.getLogger(__name__)

# This router mounts at /api and keeps the section's full paths: its routes
# span two prefixes (/schedules and /graphs/{graph_id}/schedules), so a prefixed
# mount would have to move one of them.
router = APIRouter(dependencies=[Security(requires_user)])


async def _schedule_team_ids(user_id: str, ctx: RequestContext) -> list[str]:
    if ctx.team_id is not None:
        return [ctx.team_id]
    return await get_user_team_ids(user_id, ctx.org_id) if ctx.org_id else []


async def _scheduled_chat_scopes(
    user_id: str, ctx: RequestContext
) -> dict[str, tuple[str | None, str | None]]:
    """The caller's scheduled chats this workspace can see, by the key their
    failure streaks use (a follow-up's schedule id, or a routine's id), with
    the organization and team each one runs in. Paused jobs count: a streak's
    pause is what stopped them."""
    jobs = await get_scheduler_client().get_execution_schedules(
        user_id=user_id,
        kind="copilot_turn",
        organization_id=ctx.org_id,
        team_ids=await _schedule_team_ids(user_id, ctx),
        include_paused=True,
    )
    return {
        job.routine_id or job.id: (job.organization_id or None, job.team_id)
        for job in jobs
        if isinstance(job, scheduler.CopilotTurnJobInfo)
    }


@router.post(
    dependencies=[Depends(enforce_payment_paywall)],
    path="/graphs/{graph_id}/schedules",
    summary="Create execution schedule",
)
async def create_graph_execution_schedule(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS
        ),
    ],
    graph_id: str = Path(..., description="ID of the graph to schedule"),
    schedule_params: ScheduleCreationRequest = Body(),
) -> scheduler.GraphExecutionJobInfo:
    if (
        ctx.team_id is not None
        and schedule_params.team_id is not None
        and schedule_params.team_id != ctx.team_id
    ):
        raise HTTPException(
            status_code=400,
            detail="The requested team must match the active workspace context.",
        )
    graph = await graph_db.get_graph(
        graph_id=graph_id,
        version=schedule_params.graph_version,
        user_id=user_id,
        organization_id=ctx.org_id,
        team_id=ctx.team_id,
    )
    if not graph:
        raise HTTPException(
            status_code=404,
            detail=f"Graph #{graph_id} v{schedule_params.graph_version} not found.",
        )

    # A schedule is bound to a graph, so it inherits the graph's tenant unless
    # the picker explicitly targets a team the caller may write into, or the
    # request carries an active team. As in update_graph, the graph's team is
    # only inherited when the graph lives in the org this request acts in.
    if schedule_params.team_id is not None:
        resolved_team_id = await _resolve_write_team_id(
            user_id, ctx.org_id, schedule_params.team_id
        )
    elif ctx.team_id is not None:
        resolved_team_id = ctx.team_id
    else:
        resolved_team_id = (
            graph.team_id if graph.organization_id == ctx.org_id else None
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
        if (expert.organization_id, expert.team_id) != (
            ctx.org_id,
            resolved_team_id,
        ):
            raise HTTPException(
                status_code=404, detail=f"Expert #{expert_id} not found."
            )
    else:
        expert_id = await experts_db.resolve_expert_for_graph(
            user_id,
            graph_id,
            organization_id=ctx.org_id,
            team_id=resolved_team_id,
            enforce_scope=True,
        )

    library_agent = await library_db.get_library_agent_by_graph_id(
        user_id=user_id,
        graph_id=graph_id,
        graph_version=graph.version,
        organization_id=ctx.org_id,
        team_id_restriction=resolved_team_id,
        exact_scope=True,
    )
    authorized_scope = library_agent is not None
    if library_agent is None and resolved_team_id is not None:
        grant = await resolve_graph_grant(
            user_id,
            graph_id,
            capability=GrantCapability.EXECUTE,
            graph_version=graph.version,
            organization_id=ctx.org_id,
            team_id_restriction=resolved_team_id,
        )
        authorized_scope = grant is not None
    if not authorized_scope:
        raise HTTPException(
            status_code=404,
            detail=f"Graph #{graph_id} is not available in this workspace",
        )

    async with live_resource_permission_barrier(
        user_id,
        ctx.org_id,
        resolved_team_id,
        OrgAction.CREATE_RESOURCES,
        TeamAction.CREATE_AGENTS,
    ) as allowed:
        if not allowed:
            raise HTTPException(status_code=403, detail="Resource scope is inactive")
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
            team_id=resolved_team_id,
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
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
    ],
    graph_id: str = Path(),
) -> list[scheduler.GraphExecutionJobInfo]:
    team_ids = await _schedule_team_ids(user_id, ctx)
    schedules = await get_scheduler_client().get_graph_execution_schedules(
        user_id=user_id,
        graph_id=graph_id,
        organization_id=ctx.org_id,
        team_ids=team_ids,
        include_paused=True,
    )
    return await visible_graph_schedules(schedules, user_id)


@router.get(
    path="/schedules",
    summary="List execution schedules for a user",
)
async def list_all_graphs_execution_schedules(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
    ],
) -> list[scheduler.GraphExecutionJobInfo]:
    team_ids = await _schedule_team_ids(user_id, ctx)
    schedules = await get_scheduler_client().get_graph_execution_schedules(
        user_id=user_id,
        organization_id=ctx.org_id,
        team_ids=team_ids,
        include_paused=True,
    )
    return await visible_graph_schedules(schedules, user_id)


# Keep above /schedules/{schedule_id}: a GET added there would otherwise match
# "followups" as an id, since FastAPI answers with the first route that fits.
@router.get(
    path="/schedules/followups",
    summary="List copilot follow-up schedules for a user",
    operation_id="listCopilotFollowupSchedules",
)
async def list_copilot_turn_schedules(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
    ],
) -> list[scheduler.CopilotTurnJobInfo]:
    """Return only copilot-turn schedules for the current user.

    Sibling of :func:`list_all_graphs_execution_schedules`; one route per kind
    keeps the generated frontend client typed to a single concrete return type
    instead of a discriminated union.
    """
    team_ids = await _schedule_team_ids(user_id, ctx)
    schedules = await get_scheduler_client().get_execution_schedules(
        user_id=user_id,
        kind="copilot_turn",
        organization_id=ctx.org_id,
        team_ids=team_ids,
    )
    # Defensive isinstance filter mirrors ``get_graph_execution_schedules``
    # (executor.scheduler.Scheduler) — the scheduler is the source of truth
    # for the ``kind`` filter, but we narrow the polymorphic
    # ``list[GraphExecutionJobInfo | CopilotTurnJobInfo]`` to the typed
    # subset before returning so the generated frontend client gets a single
    # concrete schema. If a row ever slips through the discriminator (e.g.
    # legacy untyped row, scheduler-side bug), we drop it rather than fail
    # the response with a Pydantic validation error.
    return [
        s
        for s in schedules
        if isinstance(s, scheduler.CopilotTurnJobInfo) and s.user_id == user_id
    ]


@router.get(
    path="/schedules/followups/paused",
    summary="List scheduled chats paused after failing the same way",
    operation_id="listPausedScheduledChats",
)
async def list_paused_scheduled_chats(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
    ],
) -> list[scheduled_chats.PausedScheduledChat]:
    """A paused job drops out of the follow-ups list above, so the chats a
    failure streak paused are listed here, each once however many fire times
    it has, with why and what Resume takes. Only the chats that run in this
    workspace are listed."""
    in_scope = await _scheduled_chat_scopes(user_id, ctx)
    return [
        chat
        for chat in await scheduled_chats.list_paused_scheduled_chats(user_id)
        if chat.schedule_key in in_scope
    ]


@router.post(
    dependencies=[Depends(enforce_payment_paywall)],
    path="/schedules/followups/{schedule_key}/resume",
    summary="Resume a scheduled chat paused after failing the same way",
    operation_id="resumeScheduledChat",
)
async def resume_scheduled_chat(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS
        ),
    ],
    schedule_key: str = Path(
        ..., description="The routine's id, or the follow-up's schedule id"
    ),
) -> ResumeScheduledChatResponse:
    scope = (await _scheduled_chat_scopes(user_id, ctx)).get(schedule_key)
    if scope is None:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail="This scheduled chat is not paused.",
        )
    organization_id, team_id = scope
    async with live_resource_access_barrier(
        user_id, organization_id, team_id, "create"
    ) as allowed:
        if not allowed:
            raise HTTPException(403, detail="Resource scope is inactive")
        try:
            resumed = await scheduled_chats.resume_scheduled_chat(user_id, schedule_key)
        except Exception:
            # The scheduler refused or the database blinked: the chat stays
            # paused, so Resume can simply be tried again.
            logger.exception("Could not resume scheduled chat %s", schedule_key)
            raise HTTPException(
                status_code=HTTP_503_SERVICE_UNAVAILABLE,
                detail="This scheduled chat could not be resumed just now. Try again.",
            )
    if not resumed:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail="This scheduled chat is not paused.",
        )
    return ResumeScheduledChatResponse(resumed=True)


@router.get(
    path="/schedules/followups/{schedule_key}/cadence",
    summary="How often a scheduled chat runs, and how it could run less often",
    operation_id="getScheduledChatCadence",
)
async def get_scheduled_chat_cadence(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS),
    ],
    schedule_key: str = Path(
        ..., description="The routine's id, or the follow-up's schedule id"
    ),
) -> cadence_service.ScheduledChatCadence:
    if schedule_key not in await _scheduled_chat_scopes(user_id, ctx):
        raise _cadence_error(
            cadence_service.CadenceNotFound(schedule_key), schedule_key
        )
    try:
        return await cadence_service.get_cadence(user_id, schedule_key)
    except Exception as e:
        raise _cadence_error(e, schedule_key) from e


@router.post(
    dependencies=[Depends(enforce_payment_paywall)],
    path="/schedules/followups/{schedule_key}/cadence",
    summary="Run a scheduled chat less often",
    operation_id="setScheduledChatCadence",
)
async def set_scheduled_chat_cadence(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS
        ),
    ],
    body: SetScheduledChatCadenceRequest,
    schedule_key: str = Path(
        ..., description="The routine's id, or the follow-up's schedule id"
    ),
) -> cadence_service.ScheduledChatCadence:
    scope = (await _scheduled_chat_scopes(user_id, ctx)).get(schedule_key)
    if scope is None:
        raise _cadence_error(
            cadence_service.CadenceNotFound(schedule_key), schedule_key
        )
    organization_id, team_id = scope
    async with live_resource_access_barrier(
        user_id, organization_id, team_id, "create"
    ) as allowed:
        if not allowed:
            raise HTTPException(403, detail="Resource scope is inactive")
        try:
            return await cadence_service.set_cadence(user_id, schedule_key, body.cron)
        except Exception as e:
            raise _cadence_error(e, schedule_key) from e


def _cadence_error(error: Exception, schedule_key: str) -> HTTPException:
    if isinstance(error, cadence_service.CadenceNotOffered):
        return HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="That cadence is not one this scheduled chat can switch to.",
        )
    if isinstance(error, cadence_service.CadenceNotFound):
        return HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail="This scheduled chat has no cadence to change.",
        )
    if isinstance(error, cadence_service.ScheduledChatPaused):
        return HTTPException(
            status_code=HTTP_409_CONFLICT,
            detail="This scheduled chat is paused. Resume it first.",
        )
    logger.exception("Could not change the cadence of scheduled chat %s", schedule_key)
    return HTTPException(
        status_code=HTTP_503_SERVICE_UNAVAILABLE,
        detail="Its cadence could not be changed just now. Try again.",
    )


@router.delete(
    path="/schedules/{schedule_id}",
    summary="Delete execution schedule",
)
async def delete_graph_execution_schedule(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS
        ),
    ],
    schedule_id: str = Path(..., description="ID of the schedule to delete"),
) -> dict[str, Any]:
    try:
        scheduler_client = get_scheduler_client()
        candidate = await scheduler_client.get_execution_schedule(
            schedule_id,
            user_id=user_id,
            organization_id=ctx.org_id,
            team_ids=await _schedule_team_ids(user_id, ctx),
        )
        target_team_ids = [candidate.team_id] if candidate.team_id is not None else []
        async with live_resource_access_barrier(
            user_id, ctx.org_id, candidate.team_id, "create"
        ) as allowed:
            if not allowed:
                raise HTTPException(403, detail="Resource scope is inactive")
            await scheduler_client.get_execution_schedule(
                schedule_id,
                user_id=user_id,
                organization_id=ctx.org_id,
                team_ids=target_team_ids,
            )
            await scheduler_client.delete_schedule(
                schedule_id,
                user_id=user_id,
                organization_id=ctx.org_id,
                team_ids=target_team_ids,
            )
    except NotFoundError:
        # Gone already, perhaps by a delete whose streak cleanup failed: a
        # retry still clears the streak, or the chat stays listed as paused.
        # One that is only outside this workspace still runs, so its streak
        # stays with it.
        if not await _schedule_exists(user_id, schedule_id):
            await _forget_scheduled_chat(user_id, schedule_id)
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Schedule #{schedule_id} not found",
        )
    # A follow-up's failure streak is keyed by its schedule id.
    await _forget_scheduled_chat(user_id, schedule_id)
    return {"id": schedule_id}


async def _schedule_exists(user_id: str, schedule_id: str) -> bool:
    """Whether the caller still has this schedule in any workspace."""
    try:
        await get_scheduler_client().get_execution_schedule(
            schedule_id, user_id=user_id
        )
    except (NotFoundError, NotAuthorizedError):
        return False
    return True


async def _forget_scheduled_chat(user_id: str, schedule_id: str) -> None:
    """The schedule is gone whatever this does, so a failure is logged, not
    answered: a 500 would say the delete failed. A streak left behind keeps
    the chat listed as paused, and deleting it from there again clears it."""
    try:
        await forget.forget_scheduled_chat(user_id, schedule_id)
    except Exception:
        logger.warning(
            "Could not clear the failure streak of deleted schedule %s",
            schedule_id,
            exc_info=True,
        )


@router.patch(
    dependencies=[Depends(enforce_payment_paywall)],
    path="/schedules/{schedule_id}",
    summary="Update execution schedule",
)
async def update_graph_execution_schedule(
    user_id: Annotated[str, Security(get_user_id)],
    ctx: Annotated[
        RequestContext,
        requires_resource_permission(
            OrgAction.CREATE_RESOURCES, TeamAction.CREATE_AGENTS
        ),
    ],
    schedule_id: str = Path(..., description="ID of the schedule to update"),
    update: ScheduleUpdateRequest = Body(),
) -> scheduler.GraphExecutionJobInfo:
    """Change an agent schedule's name, timing or inputs. Only the schedule's
    owner can edit it; anyone else gets the same 404 as a missing schedule."""
    try:
        scheduler_client = get_scheduler_client()
        candidate = await scheduler_client.get_execution_schedule(
            schedule_id,
            user_id=user_id,
            organization_id=ctx.org_id,
            team_ids=await _schedule_team_ids(user_id, ctx),
        )
        async with live_resource_access_barrier(
            user_id, ctx.org_id, candidate.team_id, "create"
        ) as allowed:
            if not allowed:
                raise HTTPException(403, detail="Resource scope is inactive")
            result = await scheduler_client.update_execution_schedule(
                schedule_id=schedule_id,
                user_id=user_id,
                name=update.name,
                cron=update.cron,
                user_timezone=update.timezone,
                input_data=update.inputs,
                input_credentials=update.credentials,
                organization_id=ctx.org_id,
                team_ids=[candidate.team_id] if candidate.team_id is not None else [],
            )
    except (NotFoundError, NotAuthorizedError):
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND,
            detail=f"Schedule #{schedule_id} not found",
        )

    if result.next_run_time:
        result.next_run_time = convert_utc_time_to_user_timezone(
            result.next_run_time, result.timezone
        )
    return result
