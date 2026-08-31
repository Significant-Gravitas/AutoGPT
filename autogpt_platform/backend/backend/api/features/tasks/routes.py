import logging
from datetime import datetime

import autogpt_libs.auth as autogpt_auth_lib
import fastapi
from fastapi import APIRouter, Query, Security

from backend.api.features.tasks import task_actions, task_review, tasks_db
from backend.api.features.tasks.errors import DelegatedTaskNotFoundError
from backend.api.features.tasks.models import (
    MAX_TASKS_PER_PAGE,
    AnswerTaskRequest,
    DelegatedTask,
    DelegatedTaskDetail,
    RejectTaskRequest,
    TaskEventsResponse,
    TaskReviewResult,
    TaskStatus,
)
from backend.copilot.executor.utils import schedule_chat_turn
from backend.copilot.pending_message_helpers import queue_user_message
from backend.util.exceptions import TaskDelegationRefusedError, TaskUpdateConflictError

logger = logging.getLogger(__name__)

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


# Declared before "/{task_id}" so "/tasks/events" is not swallowed by the
# task-detail path parameter.
@router.get("/events", operation_id="list_task_events")
async def list_task_events(
    since: datetime | None = Query(
        default=None,
        description="Only events on tasks updated after this ISO 8601 instant",
    ),
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskEventsResponse:
    """Lightweight polling feed for the office view: one event per task
    updated since ``since``, carrying the task's current lowercase status."""
    return TaskEventsResponse(
        events=await tasks_db.list_task_events(user_id, since=since)
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
    "/{task_id}/answer",
    operation_id="answer_task",
    responses={
        404: {"description": "Task not found"},
        409: {"description": "Task is not waiting on an answer"},
    },
)
async def answer_task(
    task_id: str,
    request: AnswerTaskRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> DelegatedTask:
    """Answer a task escalation. The task returns to WORKING and the answer
    is delivered into the escalating expert's session, resuming the work."""
    try:
        task, worker_session_id = await task_actions.answer_delegated_task(
            user_id, task_id, answer=request.answer
        )
    except DelegatedTaskNotFoundError:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    except (TaskDelegationRefusedError, TaskUpdateConflictError) as e:
        raise fastapi.HTTPException(status_code=409, detail=str(e))

    await _deliver_answer(
        user_id=user_id,
        task=task,
        worker_session_id=worker_session_id,
        answer=request.answer,
    )
    return task


async def _deliver_answer(
    *,
    user_id: str,
    task: DelegatedTask,
    worker_session_id: str | None,
    answer: str,
) -> None:
    """Hand the answer to the session that escalated, resuming it.

    A running turn gets it injected mid-flight via the pending buffer; an
    idle session gets a fresh turn scheduled. Best-effort — the answer is
    already on the task's timeline, so a failed delivery degrades to the
    expert reading it next time the thread runs, not to losing it.
    """
    if worker_session_id is None:
        return
    message = (
        f"[Answering your escalated question on task '{task.title}' "
        f"(task_id: {task.id})]\n\n{answer}"
    )
    try:
        queued = await queue_user_message(
            session_id=worker_session_id,
            message=message,
            require_turn_in_flight=True,
        )
        if not queued.turn_in_flight:
            await schedule_chat_turn(
                session_id=worker_session_id,
                user_id=user_id,
                message=message,
            )
    except Exception:
        logger.warning(
            "Failed to deliver escalation answer for task #%s to session #%s",
            task.id,
            worker_session_id,
            exc_info=True,
        )


@router.post(
    "/{task_id}/accept",
    operation_id="accept_task",
    responses={
        404: {"description": "Task not found"},
        409: {"description": "Task is not finished"},
    },
)
async def accept_task(
    task_id: str,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskReviewResult:
    """Approve a finished task's outcome."""
    try:
        task = await task_review.accept_delegated_task(user_id, task_id)
    except DelegatedTaskNotFoundError:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    except (TaskDelegationRefusedError, TaskUpdateConflictError) as e:
        raise fastapi.HTTPException(status_code=409, detail=str(e))
    return TaskReviewResult(task=task, message="Outcome accepted.")


@router.post(
    "/{task_id}/reject",
    operation_id="reject_task",
    responses={
        404: {"description": "Task not found"},
        409: {"description": "Task is not finished"},
    },
)
async def reject_task(
    task_id: str,
    request: RejectTaskRequest,
    user_id: str = Security(autogpt_auth_lib.get_user_id),
) -> TaskReviewResult:
    """Reject a finished task's outcome. Under the revision cap this opens a
    revision subtask for the same owner and nudges their session; at the cap
    it asks the user to clarify in chat instead of looping again."""
    try:
        task, revision = await task_review.reject_delegated_task(
            user_id, task_id, note=request.note
        )
    except DelegatedTaskNotFoundError:
        raise fastapi.HTTPException(status_code=404, detail="Task not found")
    except (TaskDelegationRefusedError, TaskUpdateConflictError) as e:
        raise fastapi.HTTPException(status_code=409, detail=str(e))

    if revision is None:
        return TaskReviewResult(
            task=task,
            escalated=True,
            message=(
                "We've iterated twice on this task already. Please clarify "
                "what you need in chat so the next attempt lands."
            ),
        )

    await _deliver_revision(user_id=user_id, task=task, revision=revision)
    return TaskReviewResult(
        task=task,
        revision_task=revision,
        message=f"Revision requested — {_owner_name(task)} is on it.",
    )


def _owner_name(task: DelegatedTask) -> str:
    return task.owner.name if task.owner else "Autopilot"


async def _deliver_revision(
    *, user_id: str, task: DelegatedTask, revision: DelegatedTask
) -> None:
    """Nudge the owner's working session with the revision ask. Best-effort,
    mirroring ``_deliver_answer`` — the revision subtask already exists, so a
    failed delivery degrades to the owner finding it next turn."""
    if revision.origin_session_id is None:
        return
    message = (
        f"[Revision requested on task '{task.title}' (task_id: {task.id})]\n\n"
        f"The user rejected the outcome and asked for changes:\n"
        f"{revision.spec}\n\n"
        f"Do the revision, then report the revision task done with "
        f"report_task (task_id: {revision.id})."
    )
    try:
        queued = await queue_user_message(
            session_id=revision.origin_session_id,
            message=message,
            require_turn_in_flight=True,
        )
        if not queued.turn_in_flight:
            await schedule_chat_turn(
                session_id=revision.origin_session_id,
                user_id=user_id,
                message=message,
            )
    except Exception:
        logger.warning(
            "Failed to deliver revision ask for task #%s to session #%s",
            task.id,
            revision.origin_session_id,
            exc_info=True,
        )


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
