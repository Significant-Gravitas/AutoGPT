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
from backend.copilot.task_run import cancel_task_run, settle_by_hand, start_task_run
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
    update_task,
)
from backend.data.task_list import SEARCH_MAX_CHARS, TaskFilter, list_task_page
from backend.data.task_record import (
    ActivityInput,
    TaskRecord,
    add_activity,
    get_task_record,
)
from backend.data.task_review import set_task_review
from backend.data.task_schedule import (
    clear_task_repeat,
    pause_task_repeat,
    resume_task_repeat,
    set_task_start,
    stop_task_repeat,
)
from backend.util.exceptions import NotFoundError

from .changes import WORKED_STATUSES, check_change, stops_work, user_change_lines
from .creation import plan_new_start, schedule_new_task
from .model import (
    TaskCancelRequest,
    TaskCreateRequest,
    TaskListResponse,
    TaskReviewRequest,
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
    """Create a task. With ``start_at`` it starts by itself then; with
    ``repeat`` it runs again and again from its next fire time."""
    start = await plan_new_start(user_id, body)
    task = await create_task(
        user_id=user_id,
        title=body.title,
        description=body.description,
        owner_expert_id=body.owner_expert_id,
        session_id=body.session_id,
        created_by=TaskCreator.USER,
        goal=body.goal,
        done_when=body.done_when,
    )
    if start is None:
        return task
    return await schedule_new_task(task, user_id, start)


@router.get("", operation_id="list_tasks")
async def list_my_tasks(
    user_id: UserId,
    status: TaskStatus | None = None,
    statuses: Annotated[
        list[TaskStatus] | None,
        Query(description="Only tasks in any of these statuses (a board lane)"),
    ] = None,
    owner_expert_id: str | None = None,
    owned_by_otto: Annotated[
        bool, Query(description="Only the tasks Otto owns (no expert)")
    ] = False,
    q: Annotated[
        str | None,
        Query(max_length=SEARCH_MAX_CHARS, description="Words in the title"),
    ] = None,
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
    """A page of the caller's tasks, most recently updated first; page on
    with ``offset`` = ``next_offset`` while ``has_more``."""
    page = await list_task_page(
        user_id,
        TaskFilter(
            statuses=statuses or ([status] if status else []),
            owner_expert_id=owner_expert_id,
            owned_by_otto=owned_by_otto,
            session_id=session_id,
            include_subtasks=include_subtasks,
            scheduled=scheduled,
            search=q,
        ),
        limit,
        offset,
    )
    return TaskListResponse(
        tasks=page.tasks, has_more=page.has_more, next_offset=page.next_offset
    )


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
    """Edit the task: its title, description, goal, what done looks like, its
    owner, and its status (To do, Done or Cancelled; Done or Cancelled on a
    task being worked on stops its work first). ``start_at`` schedules a task
    that has not started, moves its start or, sent as null, removes it."""
    before = await get_task(task_id, user_id)
    if before is None:
        raise NotFoundError(f"Task {task_id} not found")
    check_change(before, body)
    retime = "start_at" in body.model_fields_set
    start_at = requested_start(before, body) if retime else None
    task = await _apply_edit(user_id, task_id, before, body)
    if task.repeat and task.status in SETTLED_STATUSES:
        # Marked done or cancelled by hand: it stops repeating.
        await clear_task_repeat(task_id, user_id)
        task = await get_task(task_id, user_id) or task
    if retime:
        task = await set_task_start(task_id, user_id, start_at)
    start_line = await start_change_line(before, task)
    for line in [*user_change_lines(before, task), *filter(None, [start_line])]:
        await _user_line(task_id, user_id, line)
    return task


async def _apply_edit(
    user_id: str, task_id: str, before: Task, body: TaskUpdateRequest
) -> Task:
    """Write the edit. Done or Cancelled on a task being worked on settles it
    and stops its run first (``settle_by_hand``), so the run cannot flip it
    back; the rest of the edit follows."""
    patch = TaskPatch(**body.model_dump(exclude={"start_at"}))
    if body.status is None or not stops_work(before, body):
        return await update_task(task_id, user_id, patch)
    task = await settle_by_hand(user_id, task_id, body.status)
    rest = patch.model_copy(update={"status": None})
    if rest == TaskPatch():
        return task
    return await update_task(task_id, user_id, rest)


@router.delete(
    "/{task_id}", operation_id="archive_task", status_code=HTTP_204_NO_CONTENT
)
async def archive_my_task(user_id: UserId, task_id: Annotated[str, _TASK_ID]) -> None:
    """Archive the task; one being worked on is stopped first, so nothing
    keeps working on a task the user can no longer see."""
    before = await get_task(task_id, user_id)
    if before is None:
        raise NotFoundError(f"Task {task_id} not found")
    if before.status in WORKED_STATUSES:
        await cancel_task_run(user_id, task_id)
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
    instructions = body.instructions if body else None
    return await _run(user_id, task_id, instructions, http_request, ctx)


@router.post(
    "/{task_id}/review",
    operation_id="review_task",
    responses={
        402: {"description": "Subscription required"},
        404: {"description": "Task not found"},
        409: {"description": "The task's result cannot be reviewed now"},
        422: {"description": "Changes were asked for without saying what"},
        429: {"description": "Usage limit or concurrent-turn limit reached"},
    },
)
async def review_my_task(
    user_id: UserId,
    task_id: Annotated[str, _TASK_ID],
    body: TaskReviewRequest,
    http_request: Request,
    ctx: Annotated[RequestContext, Security(get_request_context)],
) -> Task:
    """Accept the task's result, or ask for changes: the note is kept on the
    task and the task runs again with it as what to change."""
    if body.decision == "accept":
        task = await set_task_review(task_id, user_id, "accepted")
        await _user_line(task_id, user_id, "You accepted the result")
        return task
    note = " ".join((body.note or "").split())
    if not note:
        raise HTTPException(status_code=422, detail="Say what to change.")
    await set_task_review(task_id, user_id, "changes_requested", note)
    await _user_line(task_id, user_id, f"You asked for changes: {note}")
    return await _run(user_id, task_id, note, http_request, ctx)


async def _run(
    user_id: str,
    task_id: str,
    instructions: str | None,
    http_request: Request,
    ctx: RequestContext,
) -> Task:
    await enforce_payment_paywall(user_id)
    await enforce_platform_allowance(user_id)
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
