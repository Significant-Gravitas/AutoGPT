import logging
from typing import Optional

import autogpt_libs.auth as autogpt_auth_lib
import fastapi
from fastapi import APIRouter, Depends, Query, Security

from backend.api.features.tasks import db, dispatch, events
from backend.api.features.tasks.models import (
    ActivityEvent,
    ActivityListResponse,
    Actor,
    AnswerCreate,
    AssigneeType,
    Attachment,
    AttachmentCreate,
    CommentCreate,
    ExpertListResponse,
    Plan,
    PlanConfirm,
    PlanConfirmResponse,
    Task,
    TaskCreate,
    TaskDetail,
    TaskFilters,
    TaskListResponse,
    TaskPriority,
    TasksEnabledResponse,
    TaskStatus,
    TaskUpdate,
)
from backend.util.feature_flag import Flag, is_feature_enabled

logger = logging.getLogger(__name__)

_WORKER_TYPES = (AssigneeType.EXPERT, AssigneeType.OTTO)
_RESTARTABLE = (
    TaskStatus.BACKLOG,
    TaskStatus.CANCELLED,
    TaskStatus.DONE,
    TaskStatus.IN_REVIEW,
)


async def require_tasks_enabled(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> str:
    if not await is_feature_enabled(Flag.TASKS_V1, user_id, default=False):
        raise fastapi.HTTPException(status_code=404, detail="Not found")
    return user_id


router = APIRouter(dependencies=[Security(autogpt_auth_lib.requires_user)])
_GATE = [Depends(require_tasks_enabled)]


@router.get(
    "/enabled", operation_id="get_tasks_enabled", response_model=TasksEnabledResponse
)
async def get_tasks_enabled(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TasksEnabledResponse:
    try:
        enabled = await is_feature_enabled(Flag.TASKS_V1, user_id, default=False)
    except Exception:
        logger.exception("Tasks flag check failed")
        enabled = False
    return TasksEnabledResponse(enabled=enabled)


def _snapshot(task: Optional[Task]) -> Optional[Task]:
    if task is None:
        return None
    return Task.model_validate(
        task.model_dump(exclude={"subtasks", "attachments", "parent"})
    )


async def _emit(
    user_id: str,
    event: str,
    task: Optional[Task],
    activity: Optional[ActivityEvent] = None,
    task_id: Optional[str] = None,
) -> None:
    try:
        await events.publish_task_event(
            user_id, event, _snapshot(task), activity, task_id=task_id
        )
    except Exception:
        logger.exception(f"Failed to publish task event {event}")


async def _on_assigned(user_id: str, task: Task) -> None:
    if task.assignee is None or task.assignee.type not in _WORKER_TYPES:
        return
    try:
        await dispatch.on_task_assigned(user_id, task.id)
    except Exception:
        logger.exception(f"Failed to start work on task {task.id}")


async def _resume(user_id: str, task_id: str, message: str) -> None:
    try:
        await dispatch.resume_task(user_id, task_id, message)
    except Exception:
        logger.exception(f"Failed to resume task {task_id}")


async def _fresh_detail(user_id: str, task_id: str) -> TaskDetail:
    detail = await db.get_task_detail(user_id, task_id)
    if detail is None:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    return detail


def _bad_request(error: Exception) -> fastapi.HTTPException:
    return fastapi.HTTPException(status_code=400, detail=str(error))


def _split(raw: Optional[str]) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


@router.get(
    "", dependencies=_GATE, operation_id="list_tasks", response_model=TaskListResponse
)
async def list_tasks(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    status: Optional[str] = Query(default=None),
    assignee_type: Optional[AssigneeType] = Query(default=None),
    assignee_id: Optional[str] = Query(default=None),
    priority: Optional[str] = Query(default=None),
    due: Optional[str] = Query(default=None, pattern="^(overdue|today|week|none)$"),
    parent_id: Optional[str] = Query(default=None),
    top_level: bool = Query(default=False),
    mine: bool = Query(default=False),
    include_hidden: bool = Query(default=False),
    q: Optional[str] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=500),
) -> TaskListResponse:
    try:
        filters = TaskFilters(
            status=[TaskStatus(s) for s in _split(status)] or None,
            assignee_type=assignee_type,
            assignee_id=assignee_id,
            priority=[TaskPriority(p) for p in _split(priority)] or None,
            due=due,
            parent_id=parent_id,
            top_level=top_level,
            mine=mine,
            include_hidden=include_hidden,
            q=q,
            limit=limit,
        )
    except ValueError as e:
        raise _bad_request(e)
    return TaskListResponse(tasks=await db.list_tasks(user_id, filters))


@router.post(
    "",
    dependencies=_GATE,
    operation_id="create_task",
    response_model=TaskDetail,
    status_code=201,
)
async def create_task(
    body: TaskCreate,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskDetail:
    return await _create(user_id, body)


async def _create(user_id: str, body: TaskCreate) -> TaskDetail:
    try:
        detail = await db.create_task(user_id, body, Actor.user(user_id))
    except ValueError as e:
        raise _bad_request(e)
    await _emit(user_id, "task_created", detail)
    if detail.parent_id:
        parent = await db.get_task(user_id, detail.parent_id)
        await _emit(user_id, "task_updated", parent)
    await _on_assigned(user_id, detail)
    return await _fresh_detail(user_id, detail.id)


@router.get(
    "/needs-you",
    dependencies=_GATE,
    operation_id="list_tasks_needing_me",
    response_model=TaskListResponse,
)
async def list_tasks_needing_me(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskListResponse:
    return TaskListResponse(tasks=await db.list_tasks_needing_user(user_id))


@router.get(
    "/experts",
    dependencies=_GATE,
    operation_id="list_task_experts",
    response_model=ExpertListResponse,
)
async def list_task_experts(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ExpertListResponse:
    return ExpertListResponse(experts=await db.list_experts_for_user(user_id))


@router.get(
    "/plans/{plan_id}",
    dependencies=_GATE,
    operation_id="get_task_plan",
    response_model=Plan,
)
async def get_task_plan(
    plan_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Plan:
    plan = await db.get_plan(user_id, plan_id)
    if plan is None:
        raise fastapi.HTTPException(status_code=404, detail="Plan not found")
    return plan


@router.post(
    "/plans/{plan_id}/confirm",
    dependencies=_GATE,
    operation_id="confirm_task_plan",
    response_model=PlanConfirmResponse,
)
async def confirm_task_plan(
    plan_id: str,
    body: Optional[PlanConfirm] = None,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> PlanConfirmResponse:
    try:
        plan, parent = await db.confirm_plan(
            user_id, plan_id, body.steps if body else None
        )
    except LookupError:
        raise fastapi.HTTPException(status_code=404, detail="Plan not found")
    except ValueError as e:
        raise _bad_request(e)
    await _emit(user_id, "task_created", parent)
    for sub in parent.subtasks:
        await _emit(user_id, "task_created", sub)
        await _on_assigned(user_id, sub)
    return PlanConfirmResponse(
        plan=plan, parent_task=await _fresh_detail(user_id, parent.id)
    )


@router.post(
    "/plans/{plan_id}/approve",
    dependencies=_GATE,
    operation_id="approve_task_plan",
    response_model=PlanConfirmResponse,
)
async def approve_task_plan(
    plan_id: str,
    body: Optional[PlanConfirm] = None,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> PlanConfirmResponse:
    return await confirm_task_plan(plan_id, body, user_id)


@router.post(
    "/plans/{plan_id}/cancel",
    dependencies=_GATE,
    operation_id="cancel_task_plan",
    response_model=Plan,
)
async def cancel_task_plan(
    plan_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Plan:
    plan = await db.cancel_plan(user_id, plan_id)
    if plan is None:
        raise fastapi.HTTPException(status_code=404, detail="Plan not found")
    return plan


@router.get(
    "/{task_id}", dependencies=_GATE, operation_id="get_task", response_model=TaskDetail
)
async def get_task(
    task_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskDetail:
    return await _fresh_detail(user_id, task_id)


@router.patch(
    "/{task_id}",
    dependencies=_GATE,
    operation_id="update_task",
    response_model=TaskDetail,
)
async def update_task(
    task_id: str,
    body: TaskUpdate,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskDetail:
    before = await _fresh_detail(user_id, task_id)
    try:
        detail = await db.update_task(user_id, task_id, body, Actor.user(user_id))
    except LookupError:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    except ValueError as e:
        raise _bad_request(e)
    became_done = detail.status == TaskStatus.DONE != before.status
    await _emit(user_id, "task_completed" if became_done else "task_updated", detail)
    if _assignee_changed(before, detail):
        await _on_assigned(user_id, detail)
    return await _fresh_detail(user_id, task_id)


def _assignee_changed(before: Task, after: Task) -> bool:
    a, b = before.assignee, after.assignee
    before_ref = (a.type, a.id) if a else None
    after_ref = (b.type, b.id) if b else None
    return before_ref != after_ref


@router.delete(
    "/{task_id}", dependencies=_GATE, operation_id="delete_task", status_code=204
)
async def delete_task(
    task_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> fastapi.Response:
    task = await db.get_task(user_id, task_id)
    if task is None or not await db.delete_task(user_id, task_id):
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    await _emit(user_id, "task_deleted", None, task_id=task_id)
    return fastapi.Response(status_code=204)


@router.post(
    "/{task_id}/subtasks",
    dependencies=_GATE,
    operation_id="create_subtask",
    response_model=TaskDetail,
)
async def create_subtask(
    task_id: str,
    body: TaskCreate,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskDetail:
    parent = await _fresh_detail(user_id, task_id)
    if parent.parent_id:
        raise fastapi.HTTPException(
            status_code=400, detail="Sub-tasks can't have their own sub-tasks"
        )
    return await _create(user_id, body.model_copy(update={"parent_id": task_id}))


@router.get(
    "/{task_id}/activity",
    dependencies=_GATE,
    operation_id="list_task_activity",
    response_model=ActivityListResponse,
)
async def list_task_activity(
    task_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ActivityListResponse:
    await _fresh_detail(user_id, task_id)
    return ActivityListResponse(events=await db.list_activity(user_id, task_id))


@router.post(
    "/{task_id}/comments",
    dependencies=_GATE,
    operation_id="create_task_comment",
    response_model=ActivityEvent,
)
async def create_task_comment(
    task_id: str,
    body: CommentCreate,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ActivityEvent:
    task = await _fresh_detail(user_id, task_id)
    event = await db.add_comment(user_id, task_id, body.body, Actor.user(user_id))
    await _emit(user_id, "task_activity", task, event)
    mentioned = {m.get("expert_id") for m in event.data.get("mentions", [])}
    assignee = task.assignee
    if assignee and assignee.type == AssigneeType.EXPERT and assignee.id in mentioned:
        await _resume(user_id, task_id, f"The user commented: {body.body}")
    return event


@router.post(
    "/{task_id}/answer",
    dependencies=_GATE,
    operation_id="answer_task_input",
    response_model=TaskDetail,
)
async def answer_task_input(
    task_id: str,
    body: AnswerCreate,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskDetail:
    try:
        await db.answer_input(
            user_id, task_id, body.answer, body.input_id, Actor.user(user_id)
        )
    except LookupError:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    except ValueError as e:
        raise _bad_request(e)
    detail = await _fresh_detail(user_id, task_id)
    await _emit(user_id, "task_updated", detail)
    await _resume(user_id, task_id, f"The user answered: {body.answer}")
    return await _fresh_detail(user_id, task_id)


@router.post(
    "/{task_id}/start",
    dependencies=_GATE,
    operation_id="start_task",
    response_model=TaskDetail,
)
async def start_task(
    task_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskDetail:
    task = await _fresh_detail(user_id, task_id)
    if task.assignee is None or task.assignee.type not in _WORKER_TYPES:
        raise fastapi.HTTPException(
            status_code=400, detail="Assign this task to an expert or Otto first"
        )
    if task.status in _RESTARTABLE:
        updated = await db.set_task_status(
            user_id, task_id, TaskStatus.TODO, Actor.user(user_id)
        )
        await _emit(user_id, "task_updated", updated)
    await _on_assigned(user_id, task)
    return await _fresh_detail(user_id, task_id)


@router.post(
    "/{task_id}/attachments",
    dependencies=_GATE,
    operation_id="add_task_attachment",
    response_model=Attachment,
)
async def add_task_attachment(
    task_id: str,
    body: AttachmentCreate,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> Attachment:
    try:
        attachment = await db.add_attachment(
            user_id, task_id, body, Actor.user(user_id)
        )
    except LookupError:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    await _emit(user_id, "task_updated", await db.get_task(user_id, task_id))
    return attachment
