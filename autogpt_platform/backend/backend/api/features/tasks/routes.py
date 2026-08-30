import autogpt_libs.auth as autogpt_auth_lib
import fastapi
from fastapi import APIRouter, Query, Security

from backend.api.features.tasks import tasks_db
from backend.api.features.tasks.errors import DelegatedTaskNotFoundError
from backend.api.features.tasks.models import (
    MAX_TASKS_PER_PAGE,
    DelegatedTask,
    DelegatedTaskDetail,
    TaskStatus,
)

router = APIRouter(
    prefix="/tasks",
    tags=["tasks", "private"],
    dependencies=[Security(autogpt_auth_lib.requires_user)],
)


@router.get("", operation_id="list_tasks")
async def list_tasks(
    expert_id: str | None = Query(
        default=None, description="Only tasks owned by this expert"
    ),
    status: TaskStatus | None = Query(default=None, description="Only this status"),
    limit: int = Query(default=MAX_TASKS_PER_PAGE, ge=1, le=MAX_TASKS_PER_PAGE),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> list[DelegatedTask]:
    """The caller's delegated tasks, newest first."""
    return await tasks_db.list_tasks(
        user_id, expert_id=expert_id, status=status, limit=limit
    )


@router.get(
    "/{task_id}",
    operation_id="get_task",
    responses={404: {"description": "Task not found"}},
)
async def get_task(
    task_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> DelegatedTaskDetail:
    """One task with its direct children, for the detail drawer."""
    detail = await tasks_db.get_task(user_id, task_id)
    if detail is None:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    return detail


@router.post(
    "/{task_id}/cancel",
    operation_id="cancel_task",
    responses={404: {"description": "Task not found"}},
)
async def cancel_task(
    task_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> DelegatedTaskDetail:
    """Cancel the task and every open task beneath it, stopping the runs they
    were driving. Already-terminal tasks are left alone, so this is safe to
    retry."""
    try:
        return await tasks_db.cancel_task(user_id, task_id)
    except DelegatedTaskNotFoundError:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
