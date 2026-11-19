"""The caller's tasks: create, read, update, archive, and link them to chats.

Every route resolves the user from the authenticated caller, and the data
layer scopes every query to that user, so a task, expert or chat someone else
owns reads as 404.
"""

from typing import Annotated

from autogpt_libs.auth import (
    RequestContext,
    get_request_context,
    get_user_id,
    requires_user,
)
from fastapi import APIRouter, HTTPException, Path, Query, Request, Security
from prisma.enums import TaskActivityKind, TaskActor, TaskCreator, TaskStatus
from starlette.status import HTTP_201_CREATED, HTTP_204_NO_CONTENT

from backend.api.features.chat.routes import enforce_platform_allowance
from backend.copilot.active_turns import inflight_turn_limit_message
from backend.copilot.llm_attribution import is_impersonated_request
from backend.copilot.rate_limit import enforce_payment_paywall
from backend.copilot.task_due import start_scheduled_now
from backend.copilot.task_run import cancel_task_run, start_task_run
from backend.copilot.turn_queue import InflightCapExceeded
from backend.data.task import (
    LIST_LIMIT_MAX,
    SETTLED_STATUSES,
    Task,
    TaskPatch,
    archive_task,
    attach_task_session,
    create_task,
    detach_task_session,
    get_task,
    list_tasks,
    update_task,
)
from backend.data.task_record import (
    ActivityInput,
    TaskRecord,
    add_activity,
    get_task_record,
)
from backend.data.task_schedule import (
    clear_task_repeat,
    pause_task_repeat,
    resume_task_repeat,
    set_task_start,
    stop_task_repeat,
)
from backend.util.exceptions import NotFoundError

from .model import (
    TaskCancelRequest,
    TaskCreateRequest,
    TaskListResponse,
    TaskRunRequest,
    TaskUpdateRequest,
)
from .start_time import requested_start, start_change_line

router = APIRouter(
    prefix="/tasks",
    tags=["tasks"],
    dependencies=[Security(requires_user)],
)

_TASK_ID = Path(min_length=1, max_length=128, description="A task the caller owns")
_SESSION_ID = Path(min_length=1, max_length=128, description="A chat the caller owns")
UserId = Annotated[str, Security(get_user_id)]


@router.post("", operation_id="create_task", status_code=HTTP_201_CREATED)
async def create_my_task(user_id: UserId, body: TaskCreateRequest) -> Task:
    return await create_task(
        user_id=user_id,
        title=body.title,
        description=body.description,
        owner_expert_id=body.owner_expert_id,
        session_id=body.session_id,
        created_by=TaskCreator.USER,
        goal=body.goal,
        done_when=body.done_when,
    )


@router.get("", operation_id="list_tasks")
async def list_my_tasks(
    user_id: UserId,
    status: TaskStatus | None = None,
    owner_expert_id: str | None = None,
    session_id: Annotated[
        str | None, Query(description="Only the tasks linked to this chat")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=LIST_LIMIT_MAX)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    include_subtasks: Annotated[
        bool, Query(description="Also list subtasks as rows of their own")
    ] = False,
    scheduled: Annotated[
        bool,
        Query(description="Only the tasks waiting for a start time, soonest first"),
    ] = False,
) -> TaskListResponse:
    tasks = await list_tasks(
        user_id=user_id,
        status=status,
        owner_expert_id=owner_expert_id,
        session_id=session_id,
        limit=limit,
        offset=offset,
        include_subtasks=include_subtasks,
        scheduled=scheduled,
    )
    return TaskListResponse(tasks=tasks)


@router.get("/{task_id}", operation_id="get_task")
async def get_my_task(user_id: UserId, task_id: Annotated[str, _TASK_ID]) -> Task:
    task = await get_task(task_id, user_id)
    if task is None:
        raise NotFoundError(f"Task {task_id} not found")
    return task


@router.get("/{task_id}/record", operation_id="get_task_record")
async def get_my_task_record(
    user_id: UserId, task_id: Annotated[str, _TASK_ID]
) -> TaskRecord:
    """The task with its subtasks, activity, asks, artifacts and evidence."""
    record = await get_task_record(task_id, user_id)
    if record is None:
        raise NotFoundError(f"Task {task_id} not found")
    return record


@router.patch("/{task_id}", operation_id="update_task")
async def update_my_task(
    user_id: UserId, task_id: Annotated[str, _TASK_ID], body: TaskUpdateRequest
) -> Task:
    """Edit the task. ``start_at`` schedules a task that has not started,
    moves its start or, sent as null, removes it."""
    before = await get_task(task_id, user_id)
    retime = "start_at" in body.model_fields_set
    start_at = requested_start(before, body) if retime else None
    task = await update_task(
        task_id, user_id, TaskPatch(**body.model_dump(exclude={"start_at"}))
    )
    if task.repeat and task.status in SETTLED_STATUSES:
        # Marked done or cancelled by hand: it stops repeating.
        await clear_task_repeat(task_id, user_id)
        task = await get_task(task_id, user_id) or task
    if retime:
        task = await set_task_start(task_id, user_id, start_at)
    line = _user_change(before, task) or await start_change_line(before, task)
    if line:
        await add_activity(
            task_id,
            user_id,
            ActivityInput(
                kind=TaskActivityKind.STATUS,
                text=line,
                actor_type=TaskActor.USER,
                is_milestone=True,
            ),
        )
    return task


def _user_change(before: Task | None, after: Task) -> str | None:
    """What the user just changed, as one activity line."""
    if before is None:
        return None
    if before.status != after.status:
        words = after.status.value.replace("_", " ").lower()
        return f"You marked it {words}"
    if before.goal != after.goal or before.done_when != after.done_when:
        return "You changed the goal"
    if before.title != after.title:
        return f"You renamed it to {after.title}"
    if before.owner_expert_id != after.owner_expert_id:
        return "You changed who owns it"
    return None


@router.delete(
    "/{task_id}", operation_id="archive_task", status_code=HTTP_204_NO_CONTENT
)
async def archive_my_task(user_id: UserId, task_id: Annotated[str, _TASK_ID]) -> None:
    if not await archive_task(task_id, user_id):
        raise NotFoundError(f"Task {task_id} not found")
    # An archived task no longer repeats or watches.
    await clear_task_repeat(task_id, user_id)


@router.post(
    "/{task_id}/run",
    operation_id="run_task",
    responses={
        402: {"description": "Subscription required"},
        404: {"description": "Task not found"},
        409: {"description": "The task is already being worked on"},
        429: {"description": "Usage limit or concurrent-turn limit reached"},
    },
)
async def run_my_task(
    user_id: UserId,
    task_id: Annotated[str, _TASK_ID],
    http_request: Request,
    ctx: Annotated[RequestContext, Security(get_request_context)],
    body: TaskRunRequest | None = None,
) -> Task:
    """Start work on a task that is not being worked on, in a chat of its
    own; the task settles when that chat's work ends. A scheduled or missed
    task starts now and loses its start time."""
    await enforce_payment_paywall(user_id)
    await enforce_platform_allowance(user_id)
    instructions = body.instructions if body else None
    try:
        if not instructions:
            started = await start_scheduled_now(user_id, task_id)
            if started is not None:
                return started
        return await start_task_run(
            user_id,
            task_id,
            instructions,
            organization_id=ctx.org_id,
            team_id=ctx.team_id,
            impersonated=is_impersonated_request(http_request),
        )
    except InflightCapExceeded as exc:
        raise HTTPException(
            status_code=429, detail=inflight_turn_limit_message(exc.cap)
        ) from exc


@router.post("/{task_id}/cancel", operation_id="cancel_task")
async def cancel_my_task(
    user_id: UserId,
    task_id: Annotated[str, _TASK_ID],
    body: TaskCancelRequest | None = None,
) -> Task:
    """Stop the work running for the task and mark it cancelled; a task
    waiting for its start time is cancelled before it starts, a repeating one
    stops repeating, and one that is already settled is returned as it is.
    ``this_run_only`` stops only a repeating task's run going now."""
    return await cancel_task_run(
        user_id, task_id, this_run_only=bool(body and body.this_run_only)
    )


@router.post("/{task_id}/repeat/pause", operation_id="pause_task_repeat")
async def pause_my_task_repeat(
    user_id: UserId, task_id: Annotated[str, _TASK_ID]
) -> Task:
    """Pause a repeating or watching task: it keeps its rhythm and starts no
    new run; a paused watching task ignores events."""
    task = await pause_task_repeat(task_id, user_id)
    await _user_line(task_id, user_id, "You paused it")
    return task


@router.post("/{task_id}/repeat/resume", operation_id="resume_task_repeat")
async def resume_my_task_repeat(
    user_id: UserId, task_id: Annotated[str, _TASK_ID]
) -> Task:
    """Resume a paused repeating task at its next fire time; a watching one
    waits for its next event."""
    task = await resume_task_repeat(task_id, user_id)
    await _user_line(task_id, user_id, "You resumed it")
    return task


@router.delete("/{task_id}/repeat", operation_id="stop_task_repeat")
async def stop_my_task_repeat(
    user_id: UserId, task_id: Annotated[str, _TASK_ID]
) -> Task:
    """Stop a task repeating, or watching for its event; a run going now
    still finishes."""
    before = await get_task(task_id, user_id)
    watching = bool(before and before.repeat and before.repeat.watch)
    task = await stop_task_repeat(task_id, user_id)
    line = "You stopped it watching" if watching else "You stopped it repeating"
    await _user_line(task_id, user_id, line)
    return task


async def _user_line(task_id: str, user_id: str, text: str) -> None:
    await add_activity(
        task_id,
        user_id,
        ActivityInput(
            kind=TaskActivityKind.STATUS,
            text=text,
            actor_type=TaskActor.USER,
            is_milestone=True,
        ),
    )


@router.post("/{task_id}/sessions/{session_id}", operation_id="attach_task_session")
async def attach_my_task_session(
    user_id: UserId,
    task_id: Annotated[str, _TASK_ID],
    session_id: Annotated[str, _SESSION_ID],
) -> Task:
    return await attach_task_session(task_id, session_id, user_id)


@router.delete("/{task_id}/sessions/{session_id}", operation_id="detach_task_session")
async def detach_my_task_session(
    user_id: UserId,
    task_id: Annotated[str, _TASK_ID],
    session_id: Annotated[str, _SESSION_ID],
) -> Task:
    return await detach_task_session(task_id, session_id, user_id)
