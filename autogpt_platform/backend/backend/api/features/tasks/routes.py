"""The caller's tasks: create, read, update, archive, and link them to chats.

Every route resolves the user from the authenticated caller, and the data
layer scopes every query to that user, so a task, expert or chat someone else
owns reads as 404.
"""

from typing import Annotated

from autogpt_libs.auth import get_user_id, requires_user
from fastapi import APIRouter, Path, Query, Security
from prisma.enums import TaskActivityKind, TaskActor, TaskCreator, TaskStatus
from starlette.status import HTTP_201_CREATED, HTTP_204_NO_CONTENT

from backend.data.task import (
    LIST_LIMIT_MAX,
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
from backend.util.exceptions import NotFoundError

from .model import TaskCreateRequest, TaskListResponse, TaskUpdateRequest

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
) -> TaskListResponse:
    tasks = await list_tasks(
        user_id=user_id,
        status=status,
        owner_expert_id=owner_expert_id,
        session_id=session_id,
        limit=limit,
        offset=offset,
        include_subtasks=include_subtasks,
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
    before = await get_task(task_id, user_id)
    task = await update_task(task_id, user_id, TaskPatch(**body.model_dump()))
    line = _user_change(before, task)
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
