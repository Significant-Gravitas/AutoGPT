"""Recover accepted work and wake the chat once to carry on with the result."""

import logging
from collections.abc import Awaitable
from datetime import UTC, datetime, timedelta
from typing import TypeGuard, cast

from prisma.enums import TaskStatus

from backend.copilot import stream_registry
from backend.copilot.active_turns import ORPHAN_RUNNING_RESET_SECONDS
from backend.copilot.background_task import BackgroundTask, task_metadata
from backend.copilot.executor.utils import enqueue_cancel_task, get_session_lock_key
from backend.copilot.model import _get_session_lock, invalidate_session_cache
from backend.copilot.tree import TurnEnvelope, keep_tree_alive
from backend.copilot.turn_queue import cancel_queued_turn, dispatch_next_for_user
from backend.data.db_accessors import (
    background_task_db,
    chat_db,
    task_record_db,
    task_result_db,
)
from backend.data.redis_client import get_redis_async
from backend.data.task_result import TaskOutcome

logger = logging.getLogger(__name__)


async def reconcile_background_subtasks() -> None:
    for task in await background_task_db().list_background_tasks():
        try:
            await _reconcile(task)
        except Exception:
            logger.exception(f"Background subtask recovery failed for {task.id}")


async def cancel_background_subtasks(
    user_id: str,
    session_id: str,
    launch_turn_id: str | None = None,
) -> None:
    try:
        tasks = await background_task_db().cancel_background_tasks(
            user_id, session_id, launch_turn_id
        )
    except Exception:
        logger.exception(f"Could not persist background cancellation for {session_id}")
        return
    await stop_cancelled_subtasks(tasks)


async def stop_cancelled_subtasks(tasks: list[BackgroundTask]) -> None:
    """Stop the children of handoffs just recorded as cancelled, and refresh
    their chats."""
    for task in tasks:
        await _stop_child(task)
        await invalidate_session_cache(task.child_session_id)


async def cancel_orphaned_background_subtasks(
    user_id: str, session_id: str, expected_updated_at: datetime
) -> bool:
    tasks = await background_task_db().cancel_orphaned_background_tasks(
        user_id, session_id, expected_updated_at
    )
    if tasks is None:
        return False
    await stop_cancelled_subtasks(tasks)
    await invalidate_session_cache(session_id)
    return True


async def _reconcile(task: BackgroundTask) -> None:
    status = await chat_db().get_chat_session_status(task.child_session_id)
    if task.wake_suppressed:
        await _stop_child(task)
        if await chat_db().get_chat_session_status(task.child_session_id) == "idle":
            await background_task_db().suppress_background_followup(
                task.id, task.user_id
            )
        return
    # Open work keeps its tree, and the counts the tree holds, from expiring.
    await keep_tree_alive(task.context.child_envelope)
    if task.outcome is None:
        settled = await _settle_child(task, status)
        if settled is None:
            return
        task = settled
    if task.outcome in ("cancelled", "timed_out"):
        await _stop_child(task)
    if task.followup_state == "queued":
        await _recover_followup(task)
    elif task.followup_state == "waiting":
        await _deliver(task)


async def _settle_child(
    task: BackgroundTask, status: str | None
) -> BackgroundTask | None:
    """Record how a child without a recorded outcome ended, from what its
    chat and stream show, and return the refreshed record; None while the
    child is still queued or running, or once its record is gone."""
    if status == "idle" and await background_task_db().has_open_handoffs(
        task.user_id, task.child_session_id
    ):
        # The child handed work off itself: its record waits for the last of
        # that work to come back to it, and its chat carries on meanwhile.
        await dispatch_next_for_user(task.user_id)
        return None
    completion = (
        await stream_registry.get_session(task.child_session_id)
        if status == "idle" and task.started_at is not None
        else None
    )
    if completion and (
        completion.user_id,
        completion.turn_id,
        completion.status,
    ) == (task.user_id, task.child_turn_id, "completed"):
        await background_task_db().finish_background_turn(
            task.user_id, task.child_session_id, task.child_turn_id, None
        )
    elif status == "idle":
        # The chat went idle without the worker recording an outcome: the
        # turn was interrupted, so the parent carries on from that.
        await background_task_db().complete_background_task(
            task.id, task.user_id, "failed"
        )
    elif (
        status == "queued"
        and task.deadline is not None
        and datetime.now(UTC) >= task.deadline
    ):
        # Only a deadline the launching turn already had: a child still
        # queued past it would be refused at pickup, so close it here
        # instead of nudging the queue forever. Once promoted, the
        # worker's own clock bounds it.
        await background_task_db().complete_background_task(
            task.id, task.user_id, "timed_out"
        )
        await _stop_child(task)
    else:
        if status == "running" and task.started_at is not None:
            await _release_if_abandoned(task, task.child_session_id, task.child_turn_id)
        await dispatch_next_for_user(task.user_id)
        return None
    return await background_task_db().get_background_task(task.id, task.user_id)


# Where a finished worker leaves the tasks linked to its chat.
_TASK_STATUS_BY_OUTCOME: dict[str, TaskStatus] = {
    "completed": TaskStatus.DONE,
    "failed": TaskStatus.FAILED,
    "timed_out": TaskStatus.FAILED,
    "cancelled": TaskStatus.CANCELLED,
}


# Why a task failed, in words for the user.
_FAILURE_REASON_BY_OUTCOME: dict[str, str] = {
    "failed": "The work stopped before it finished.",
    "timed_out": "The work ran out of time.",
}


async def settle_worker_tasks(task: BackgroundTask) -> None:
    """Close the tasks a finished worker was running (``start_task``).

    Only open ones move: a task waiting on the user (NEEDS_INPUT) stays
    open, since the worker ended by asking. Best-effort and idempotent, so
    a delivery retried later moves nothing twice.
    """
    status = _TASK_STATUS_BY_OUTCOME.get(task.outcome or "")
    if status is None:
        return
    try:
        result = None
        if status == TaskStatus.DONE:
            child = await background_task_db().background_child_result(
                task.child_session_id, task.user_id
            )
            result = child.response_text if child else None
        await task_record_db().settle_work_tasks(
            task.child_session_id,
            task.user_id,
            status,
            result,
            _FAILURE_REASON_BY_OUTCOME.get(task.outcome or ""),
        )
    except Exception:
        logger.warning(f"Could not settle tasks for background task {task.id}")


async def worker_task_outcome(task: BackgroundTask) -> TaskOutcome | None:
    """What the worker's task came to, so the chat that started it can be
    told in the wake itself. Best-effort: without it the chat reads the
    worker's own reply."""
    try:
        return await task_result_db().task_outcome(task.child_session_id, task.user_id)
    except Exception:
        logger.warning(f"Could not read the task of background task {task.id}")
        return None


async def _deliver(task: BackgroundTask) -> None:
    await settle_worker_tasks(task)
    outcome = await worker_task_outcome(task)
    async with _get_session_lock(task.parent_session_id) as locked:
        if not locked:
            return
        if await chat_db().get_chat_session_status(task.parent_session_id) != "idle":
            return
        envelope = continuation_envelope(task)
        metadata = task_metadata(task, "followup", envelope)
        metadata["background_task_notice"] = task.id
        metadata["display_content"] = (
            f"Background subtask {outcome_label(task)}. Otto is following up."
        )
        if _is_handed_over(outcome):
            # The chat shows this task's result itself, from its record.
            metadata["task_handoff"] = outcome.task_id
        queued = await background_task_db().enqueue_background_followup(
            task.id, task.user_id, continuation_message(task, outcome), metadata
        )
    if queued in ("queued", "delivered"):
        await invalidate_session_cache(task.parent_session_id)
    if queued == "queued":
        await dispatch_next_for_user(task.user_id)


def outcome_label(task: BackgroundTask) -> str:
    return "timed out" if task.outcome == "timed_out" else task.outcome or "finished"


def continuation_envelope(task: BackgroundTask) -> TurnEnvelope:
    """The launching turn's own envelope: the chat carries on with the tools,
    spawn rights and spend ledger it had, and with no clock of its own."""
    return task.context.envelope


# The user skims: the chat that started the work answers in a few lines and
# points at the task for the rest.
_SHORT_ANSWER = (
    "Tell the user now in at most 3 short lines: what was done, the one "
    "thing they should know, and the link to the full result when there is "
    "one. Do not paste or retell the document and do not list steps; the "
    "task holds the detail."
)


# A finished task's result and document are shown in the chat as they are, so
# the chat that started the work adds only what they do not already say.
_HANDED_OVER = (
    "The user already sees this result, and its document when it has one, in "
    "the chat, handed over by whoever did the work. Do not repeat or retell "
    "any of it, names and numbers included, and do not link it. Reply in at "
    "most 2 short sentences: the one thing they should know or decide next, "
    "or what you do now."
)


def _is_handed_over(outcome: TaskOutcome | None) -> TypeGuard[TaskOutcome]:
    """A task that finished with a result is shown to the user as it is."""
    return bool(outcome and outcome.status == TaskStatus.DONE and outcome.result)


def continuation_message(
    task: BackgroundTask, outcome: TaskOutcome | None = None
) -> str:
    """What wakes the chat that started the work."""
    return continuation_text(task.child_session_id, outcome_label(task), outcome)


def continuation_text(
    child_session_id: str, label: str, outcome: TaskOutcome | None = None
) -> str:
    """A task that left a result is told here, with its document, so the chat
    can answer at once; otherwise the subtask's output stays in its own chat
    and reaches the model as a tool result, as a foreground subtask's does."""
    read_more = f'get_sub_session_result(sub_session_id="{child_session_id}")'
    if _is_handed_over(outcome):
        document = (
            f" Full result: [{outcome.document_title}]"
            f"(workspace://{outcome.document_file_id})."
            if outcome.document_title and outcome.document_file_id
            else ""
        )
        return (
            f"The task '{outcome.title}' is done (task id {outcome.task_id}). "
            f"Result: {outcome.result}{document}\n\n{_HANDED_OVER} If the "
            "user's request has more steps, carry on with them; read the "
            f"worker's chat with {read_more} only if you need more than this."
        )
    return (
        f"Background subtask {label}: "
        f"/copilot?sessionId={child_session_id}. Read what it produced with "
        f"{read_more} and "
        "carry on with the user's request from where you left off. If it did "
        f"not complete, say so rather than claiming a result. {_SHORT_ANSWER}"
    )


async def _recover_followup(task: BackgroundTask) -> None:
    status = await chat_db().get_chat_session_status(task.parent_session_id)
    if status == "idle":
        # The turn may have answered and only failed to record that, so a
        # completion Redis still holds for the wake counts, as for a child.
        completion = (
            await stream_registry.get_session(task.parent_session_id)
            if task.followup_started_at is not None
            else None
        )
        delivered = completion is not None and (
            completion.user_id,
            completion.turn_id,
            completion.status,
        ) == (task.user_id, task.wake_id, "completed")
        await background_task_db().finish_background_turn(
            task.user_id,
            task.parent_session_id,
            task.wake_id,
            None if delivered else "Follow-up interrupted",
        )
        return
    if status == "running" and task.followup_started_at is not None:
        await _release_if_abandoned(task, task.parent_session_id, task.wake_id)
    await dispatch_next_for_user(task.user_id)


async def _release_if_abandoned(
    task: BackgroundTask, session_id: str, turn_id: str
) -> None:
    """A chat still running the task's turn with no live stream for it lost
    its worker. The chat routes apply this rule when the chat is read; the
    sweep applies it so a chat nobody has open recovers too, and the next
    pass records the interrupted outcome."""
    active, _ = await stream_registry.get_active_session(session_id, task.user_id)
    if active is not None and active.turn_id == turn_id:
        return
    redis = await get_redis_async()
    if await cast(Awaitable[int], redis.exists(get_session_lock_key(session_id))):
        # The worker refreshes the chat's lock while its turn runs, whatever
        # became of the stream metadata; only the lock's expiry says it died.
        return
    before = datetime.now(UTC) - timedelta(seconds=ORPHAN_RUNNING_RESET_SECONDS)
    if await background_task_db().release_abandoned_background_turn(
        task, session_id, turn_id, before
    ):
        await invalidate_session_cache(session_id)


async def _stop_child(task: BackgroundTask) -> None:
    if await cancel_queued_turn(
        user_id=task.user_id,
        session_id=task.child_session_id,
        expected_turn_id=task.child_turn_id,
    ):
        return
    active = await stream_registry.get_session(task.child_session_id)
    if active and active.status == "running" and active.turn_id == task.child_turn_id:
        await enqueue_cancel_task(task.child_session_id, task.child_turn_id)
