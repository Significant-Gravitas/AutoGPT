import logging

import autogpt_libs.auth as autogpt_auth_lib
import fastapi
from fastapi import APIRouter, Depends, Query, Security

from backend.api.features.tasks import contract_db, db, dispatch, events, inbox
from backend.api.features.tasks.approvals_answer import answer_cards
from backend.api.features.tasks.contract import brief
from backend.api.features.tasks.contract_db import EvidenceDetail, TaskSummary
from backend.api.features.tasks.models import (
    ActivityEvent,
    ActivityListResponse,
    Actor,
    AnswerCreate,
    ApprovalsAnswer,
    AskAnswer,
    AssigneeType,
    Attachment,
    AttachmentCreate,
    CommentCreate,
    ExpertListResponse,
    Plan,
    PlanConfirm,
    PlanConfirmResponse,
    StepStatus,
    Task,
    TaskActionName,
    TaskActionRequest,
    TaskCounts,
    TaskCreate,
    TaskDetail,
    TaskFilters,
    TaskListResponse,
    TaskPriority,
    TasksEnabledResponse,
    TaskStatus,
    TaskUpdate,
)
from backend.copilot import turn_queue
from backend.copilot.executor.utils import enqueue_cancel_task
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


def _snapshot(task: Task | None) -> Task | None:
    if task is None:
        return None
    return Task.model_validate(
        task.model_dump(exclude={"subtasks", "attachments", "parent"})
    )


async def _emit(
    user_id: str,
    event: str,
    task: Task | None,
    activity: ActivityEvent | None = None,
    task_id: str | None = None,
) -> None:
    try:
        snapshot = _snapshot(task)
        if snapshot is not None:
            snapshot = await inbox.with_inbox_one(snapshot)
        await events.publish_task_event(
            user_id, event, snapshot, activity, task_id=task_id
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
    decorated = await inbox.with_inbox([detail, *detail.subtasks])
    detail.subtasks = decorated[1:]
    return detail


def _bad_request(error: Exception) -> fastapi.HTTPException:
    return fastapi.HTTPException(status_code=400, detail=str(error))


def _split(raw: str | None) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


@router.get(
    "", dependencies=_GATE, operation_id="list_tasks", response_model=TaskListResponse
)
async def list_tasks(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    status: str | None = Query(default=None),
    assignee_type: AssigneeType | None = Query(default=None),
    assignee_id: str | None = Query(default=None),
    priority: str | None = Query(default=None),
    due: str | None = Query(default=None, pattern="^(overdue|today|week|none)$"),
    parent_id: str | None = Query(default=None),
    top_level: bool = Query(default=False),
    mine: bool = Query(default=False),
    include_hidden: bool = Query(default=False),
    q: str | None = Query(default=None),
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
    tasks = await inbox.with_inbox(await db.list_tasks(user_id, filters))
    counts = await inbox.count_tasks(user_id, filters)
    return TaskListResponse(tasks=tasks, counts=counts)


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
    "/counts",
    dependencies=_GATE,
    operation_id="count_tasks",
    response_model=TaskCounts,
)
async def count_tasks(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
    assignee_type: AssigneeType | None = Query(default=None),
    assignee_id: str | None = Query(default=None),
    priority: str | None = Query(default=None),
    due: str | None = Query(default=None, pattern="^(overdue|today|week|none)$"),
    parent_id: str | None = Query(default=None),
    top_level: bool = Query(default=False),
    mine: bool = Query(default=False),
    q: str | None = Query(default=None),
) -> TaskCounts:
    try:
        filters = TaskFilters(
            assignee_type=assignee_type,
            assignee_id=assignee_id,
            priority=[TaskPriority(p) for p in _split(priority)] or None,
            due=due,
            parent_id=parent_id,
            top_level=top_level,
            mine=mine,
            q=q,
        )
    except ValueError as e:
        raise _bad_request(e)
    return await inbox.count_tasks(user_id, filters)


@router.get(
    "/needs-you",
    dependencies=_GATE,
    operation_id="list_tasks_needing_me",
    response_model=TaskListResponse,
)
async def list_tasks_needing_me(
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskListResponse:
    return TaskListResponse(
        tasks=await inbox.with_inbox(await db.list_tasks_needing_user(user_id))
    )


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
    body: PlanConfirm | None = None,
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
    if not plan.split:
        await _on_assigned(user_id, parent)
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
    body: PlanConfirm | None = None,
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
    cursor: str | None = Query(default=None),
    limit: int | None = Query(default=None, ge=1, le=200),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> ActivityListResponse:
    """The full log, oldest first. Pass ``limit`` (and then the returned
    ``next_cursor``) to page through it; evidence content is left out and
    fetched with the evidence endpoint."""
    if await db.get_task(user_id, task_id) is None:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    if cursor is None and limit is None:
        events, next_cursor = await db.list_activity(user_id, task_id), None
    else:
        events, next_cursor = await contract_db.list_activity_page(
            user_id, task_id, cursor, limit or 50
        )
    return ActivityListResponse(
        events=[_without_content(e) for e in events], next_cursor=next_cursor
    )


def _without_content(event: ActivityEvent) -> ActivityEvent:
    if not event.data.get("content"):
        return event
    return event.model_copy(update={"data": {**event.data, "content": None}})


@router.get(
    "/{task_id}/summary",
    dependencies=_GATE,
    operation_id="get_task_summary",
    response_model=TaskSummary,
)
async def get_task_summary(
    task_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskSummary:
    """The task's contract and derived state: what the panel shows."""
    summary = await contract_db.get_summary(user_id, task_id)
    if summary is None:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    return summary


@router.get(
    "/{task_id}/evidence/{evidence_id}",
    dependencies=_GATE,
    operation_id="get_task_evidence",
    response_model=EvidenceDetail,
)
async def get_task_evidence(
    task_id: str,
    evidence_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> EvidenceDetail:
    """One proof item in full: its attachment and the activity row holding
    any raw content."""
    detail = await contract_db.get_evidence(user_id, task_id, evidence_id)
    if detail is None:
        raise fastapi.HTTPException(status_code=404, detail="Evidence not found")
    return detail


@router.post(
    "/{task_id}/ask/answer",
    dependencies=_GATE,
    operation_id="answer_task_ask",
    response_model=TaskSummary,
)
async def answer_task_ask(
    task_id: str,
    body: AskAnswer,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskSummary:
    try:
        await db.answer_input(
            user_id, task_id, body.answer, body.ask_id, Actor.user(user_id)
        )
    except LookupError:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    except ValueError as e:
        raise _bad_request(e)
    await _emit(user_id, "task_updated", await db.get_task(user_id, task_id))
    await _resume(user_id, task_id, f"The user answered: {body.answer}")
    return await get_task_summary(task_id, user_id)


@router.post(
    "/{task_id}/approvals",
    dependencies=_GATE,
    operation_id="answer_task_approvals",
    response_model=TaskSummary,
)
async def answer_task_approvals(
    task_id: str,
    body: ApprovalsAnswer,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskSummary:
    """Approve or decline the hire and raise cards the task waits on, the
    same as clicking them in the task's chat, and let the worker go on."""
    try:
        message = await answer_cards(user_id, task_id, body.decisions)
    except LookupError:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    except ValueError as e:
        raise _bad_request(e)
    await _emit(user_id, "task_updated", await db.get_task(user_id, task_id))
    await _resume(user_id, task_id, message)
    return await get_task_summary(task_id, user_id)


@router.post(
    "/{task_id}/actions/{action}",
    dependencies=_GATE,
    operation_id="take_task_action",
    response_model=TaskSummary,
)
async def take_task_action(
    task_id: str,
    action: str,
    body: TaskActionRequest | None = None,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskSummary:
    """What the user does about a stuck or finished task. ``retry_failed_part``,
    ``allow``, ``change_goal``, ``take_over``, ``stop`` and ``accept`` have
    their own handling; any other action is passed to the worker as the
    user's instruction, with ``note``."""
    body = body or TaskActionRequest()
    task = await db.get_task(user_id, task_id)
    if task is None:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    try:
        await _apply_action(user_id, task, action.strip().lower(), body)
    except ValueError as e:
        raise _bad_request(e)
    return await get_task_summary(task_id, user_id)


async def _stop_work_turn(user_id: str, task: Task) -> None:
    """Stop the expert's queued or running turn too, the way Stop in the chat
    does; otherwise it keeps working and can mark a cancelled task done."""
    session_id = task.work_session_id
    if not session_id:
        return
    try:
        if await turn_queue.cancel_queued_turn(user_id=user_id, session_id=session_id):
            return
        await enqueue_cancel_task(session_id)
    except Exception:
        logger.exception(f"Could not stop the work turn of task {task.id}")


async def _apply_action(
    user_id: str, task: Task, action: str, body: TaskActionRequest
) -> None:
    me = Actor.user(user_id)
    contract = await contract_db.get_contract(user_id, task.id)
    stopped = contract.blocker.what_stopped_it if contract.blocker else None
    note = f" {body.note.strip()}" if body.note and body.note.strip() else ""

    async def log(text: str) -> None:
        activity = await contract_db.log_action(user_id, task.id, action, me, text)
        await _emit(user_id, "task_activity", task, activity)

    async def unblock() -> None:
        update = await contract_db.clear_blocker(user_id, task.id)
        await events.publish_contract_update(user_id, update)

    async def settle(status: TaskStatus) -> None:
        await db.set_live(task.id, False)
        updated = await db.set_task_status(user_id, task.id, status, me)
        await _emit(user_id, "task_updated", updated)

    if action == TaskActionName.STOP:
        await log(f"You stopped it.{note}")
        await settle(TaskStatus.CANCELLED)
        await _stop_work_turn(user_id, task)
        return
    if action == TaskActionName.ACCEPT:
        if contract.result is None:
            summary = task.outcome_summary or "Accepted as it is."
            update = await contract_db.update_task(
                user_id, task.id, {"result": {"summary": summary}}, me
            )
            await events.publish_contract_update(user_id, update)
        await log(f"You accepted it.{note}")
        await settle(TaskStatus.DONE)
        return
    if action == TaskActionName.TAKE_OVER:
        await unblock()
        await log(f"You took it over; the work so far is kept.{note}")
        await db.unassign(user_id, task.id)
        await settle(TaskStatus.TODO)
        return
    if action == TaskActionName.CHANGE_GOAL:
        if not body.goal:
            raise ValueError("Give the new goal.")
        update = await contract_db.change_goal(user_id, task.id, body.goal, me)
        await events.publish_contract_update(user_id, update)
        await log(f"You changed the goal to: {update.contract.goal}{note}")
        await _resume(
            user_id,
            task.id,
            f"The user changed the goal to: {update.contract.goal}.{note} Revise "
            "the plan and checks with update_task, keep the finished work, and "
            "go on.",
        )
        return
    if action in (TaskActionName.ALLOW, TaskActionName.DO_IN_CHAT):
        if task.live.running:
            raise ValueError("Wait for the current turn to end, then allow it.")
        await unblock()
        await log(f"You allowed it{f': {stopped}' if stopped else ''}.{note}")
        # A fresh chat, which now counts as one the user drives
        # (dispatch._user_drives), carrying the record of the work so far.
        await db.clear_work_session(task.id)
        fresh = await db.get_task(user_id, task.id)
        if fresh is not None:
            await dispatch.start_task_turn(
                user_id,
                fresh,
                dispatch.resume_brief(
                    fresh,
                    brief(contract),
                    f"The user allowed what stopped you"
                    f"{f' ({stopped})' if stopped else ''}. Try it again.{note}",
                ),
                dispatch.kickoff_metadata(fresh),
            )
        return
    if action == TaskActionName.SKIP:
        failed = next(
            (
                s
                for s in contract.plan
                if s.status in (StepStatus.FAILED, StepStatus.RUNNING)
            ),
            None,
        )
        if failed is not None:
            update = await contract_db.update_task(
                user_id,
                task.id,
                {"steps": [{"id": failed.id, "status": "skipped"}]},
                me,
            )
            await events.publish_contract_update(user_id, update)
        await unblock()
        part = f' "{failed.title}"' if failed else " the part that failed"
        await log(f"You skipped{part}.{note}")
        await _resume(
            user_id,
            task.id,
            f"The user chose to skip{part}.{note} Go on without it and say in "
            "the result what was left out.",
        )
        return
    await unblock()
    if action == TaskActionName.RETRY_FAILED_PART:
        await log(f"You asked to retry the part that failed.{note}")
        message = (
            f"The user asked you to retry the part that failed"
            f"{f': {stopped}' if stopped else ''}.{note} Keep the finished "
            "work and evidence; redo only what failed."
        )
    else:
        await log(f"You chose: {action}.{note}")
        message = f"The user chose: {action}.{note} Act on it and go on."
    await _resume(user_id, task.id, message)


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
