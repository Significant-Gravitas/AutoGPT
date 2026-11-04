"""Starts and resumes expert work on tasks.

An expert works a task by running a normal unattended copilot turn in a
dedicated work chat (``Task.work_session_id``), queued the same way a
delegation or sub-session turn is (``run_copilot_turn_via_queue`` with
``timeout=0``). The turn-finish hook in ``stream_registry`` calls
``on_work_turn_finished`` (see ``dispatch_finish``).

Every public function here is best-effort and never raises.
"""

import logging
import time
from datetime import UTC, datetime

from backend.api.features.tasks.access import tasks_store
from backend.api.features.tasks.events import publish_task_event
from backend.api.features.tasks.live_step import remember_work_session
from backend.api.features.tasks.models import (
    DEFAULT_BUDGET_CAP_USD,
    ActivityKind,
    Actor,
    AssigneeType,
    Task,
    TaskStatus,
)
from backend.copilot.model import create_chat_session, get_chat_session
from backend.copilot.sdk.session_waiter import run_copilot_turn_via_queue

logger = logging.getLogger(__name__)

MAX_TURNS_PER_TASK = 20
RESUME_WINDOW_SECONDS = 120
MAX_RESUMES_PER_WINDOW = 6
_STARTABLE = (TaskStatus.BACKLOG, TaskStatus.TODO)
_recent_starts: dict[str, list[float]] = {}

TOOL_HINT = (
    "Task tools: list_my_tasks, update_task_status, comment_on_task, "
    "request_input, link_artifact, create_task. If a tool is not in your tool "
    "list, call it through run_capability with id 'tool:<name>'."
)


async def on_task_assigned(user_id: str, task_id: str) -> None:
    """Start work if the task is assigned to an expert or Otto and not started."""
    try:
        task = await tasks_store().get_task(user_id, task_id)
        if task is None or not _worked_by_ai(task):
            return
        if task.status not in _STARTABLE and not (
            task.status == TaskStatus.IN_PROGRESS and not task.live.running
        ):
            return
        if await _wait_for_expert(user_id, task):
            return
        await start_task_turn(user_id, task, kickoff_message(task))
    except Exception:
        logger.exception("on_task_assigned failed for task %s", task_id)


async def resume_task(user_id: str, task_id: str, message: str) -> None:
    """Run another turn in the task's work chat (an answer, a mention)."""
    try:
        task = await tasks_store().get_task(user_id, task_id)
        if task is None or not _worked_by_ai(task):
            return
        if task.status in (TaskStatus.DONE, TaskStatus.CANCELLED):
            return
        if task.work_session_id is None:
            await on_task_assigned(user_id, task_id)
            return
        await start_task_turn(user_id, task, message)
    except Exception:
        logger.exception("resume_task failed for task %s", task_id)


async def on_work_turn_finished(session_id: str, status: str) -> None:
    from backend.api.features.tasks.dispatch_finish import finish_work_turn

    try:
        await finish_work_turn(session_id, status)
    except Exception:
        logger.exception("on_work_turn_finished failed for session %s", session_id)


async def start_task_turn(user_id: str, task: Task, message: str) -> bool:
    """Guard, open the work chat if needed, and queue one turn."""
    store = tasks_store()
    refusal = _guard_refusal(task)
    if refusal:
        await _block(user_id, task, refusal)
        return False
    if task.status != TaskStatus.IN_PROGRESS:
        task = await store.set_task_status(
            user_id, task.id, TaskStatus.IN_PROGRESS, Actor.system()
        )
    session_id = await _ensure_work_session(user_id, task)
    session = await get_chat_session(session_id, user_id)
    now = datetime.now(UTC)
    prompt_tokens = sum(u.prompt_tokens for u in session.usage) if session else 0
    completion_tokens = (
        sum(u.completion_tokens for u in session.usage) if session else 0
    )
    activity = await store.add_activity(
        user_id,
        task.id,
        ActivityKind.TURN,
        assignee_actor(task),
        data={
            "session_id": session_id,
            "turn_index": task.metrics.turns + 1,
            "status": "running",
            "started_at": now.isoformat(),
            "start_message_count": len(session.messages) if session else 0,
            "start_prompt_tokens": prompt_tokens,
            "start_completion_tokens": completion_tokens,
            "tool_calls": 0,
            "tokens": 0,
            "cost_usd": 0.0,
            "duration_ms": 0,
            "tool_call_cards": [],
        },
    )
    await store.set_live(task.id, True, "Starting", now)
    await remember_work_session(session_id, user_id, task.id, now)
    _recent_starts.setdefault(task.id, []).append(time.monotonic())
    fresh = await store.get_task(user_id, task.id)
    await publish_task_event(user_id, "task_updated", fresh, task_id=task.id)
    await publish_task_event(user_id, "task_activity", fresh, activity, task.id)
    return await _queue_turn(user_id, task, session_id, message, activity.id)


async def _queue_turn(
    user_id: str, task: Task, session_id: str, message: str, activity_id: str
) -> bool:
    store = tasks_store()
    try:
        outcome, result = await run_copilot_turn_via_queue(
            session_id=session_id,
            user_id=user_id,
            message=message,
            timeout=0,
            tool_call_id=f"task:{task.id}",
            tool_name="task_dispatch",
            allow_queue=True,
        )
    except Exception as e:
        logger.warning("Could not queue a work turn for task %s: %s", task.id, e)
        outcome, result = "failed", None
    if outcome in ("running", "queued", "completed"):
        return True
    reason = (getattr(result, "refusal", None) if result else None) or (
        "Too many chats are running right now. Start it again in a moment."
        if outcome == "rejected_concurrent_turn_cap"
        else "The work turn could not be started."
    )
    await store.update_activity(activity_id, data={"status": "failed"})
    await store.set_live(task.id, False)
    note = await store.add_activity(
        user_id, task.id, ActivityKind.SYSTEM, Actor.system(), body=reason
    )
    fresh = await store.get_task(user_id, task.id)
    await publish_task_event(user_id, "task_activity", fresh, note, task.id)
    return False


def _guard_refusal(task: Task) -> str | None:
    cap = (
        task.budget_cap_usd
        if task.budget_cap_usd is not None
        else DEFAULT_BUDGET_CAP_USD
    )
    if task.metrics.cost_usd >= cap:
        return (
            f"Paused: this task reached its budget of ${cap:.2f} "
            f"(spent ${task.metrics.cost_usd:.2f}). Raise the budget to continue."
        )
    if task.metrics.turns >= MAX_TURNS_PER_TASK:
        return (
            f"Paused after {task.metrics.turns} work turns so it does not loop. "
            "Check the progress and restart it if it should keep going."
        )
    cutoff = time.monotonic() - RESUME_WINDOW_SECONDS
    recent = [t for t in _recent_starts.get(task.id, []) if t >= cutoff]
    _recent_starts[task.id] = recent
    if len(recent) >= MAX_RESUMES_PER_WINDOW:
        return (
            "Paused: this task was restarted too many times in a short window. "
            "Check the conversation and restart it when ready."
        )
    return None


async def _block(user_id: str, task: Task, note: str) -> None:
    store = tasks_store()
    await store.set_live(task.id, False)
    fresh = await store.set_task_status(
        user_id, task.id, TaskStatus.BLOCKED, Actor.system(), note=note
    )
    activity = await store.add_activity(
        user_id, task.id, ActivityKind.SYSTEM, Actor.system(), body=note
    )
    await publish_task_event(user_id, "task_needs_input", fresh, activity, task.id)


async def _wait_for_expert(user_id: str, task: Task) -> bool:
    """Queue behind another task the same expert is working. True if queued."""
    assignee = task.assignee
    if assignee is None or assignee.type != AssigneeType.EXPERT or not assignee.id:
        return False
    store = tasks_store()
    busy = await store.get_busy_task_for_expert(user_id, assignee.id, task.id)
    if busy is None:
        return False
    if task.status != TaskStatus.TODO:
        task = await store.set_task_status(
            user_id, task.id, TaskStatus.TODO, Actor.system()
        )
    note = await store.add_activity(
        user_id,
        task.id,
        ActivityKind.SYSTEM,
        Actor.system(),
        body=f"Waiting for {assignee.name} to finish {busy.key}",
    )
    await publish_task_event(user_id, "task_activity", task, note, task.id)
    return True


async def _ensure_work_session(user_id: str, task: Task) -> str:
    if task.work_session_id:
        existing = await get_chat_session(task.work_session_id, user_id)
        if existing is not None:
            return task.work_session_id
    expert_id = (
        task.assignee.id
        if task.assignee and task.assignee.type == AssigneeType.EXPERT
        else None
    )
    session = await create_chat_session(
        user_id, dry_run=False, expert_id=expert_id, origin="automation"
    )
    await tasks_store().set_work_session(task.id, session.session_id)
    return session.session_id


def _worked_by_ai(task: Task) -> bool:
    return task.assignee is not None and task.assignee.type in (
        AssigneeType.EXPERT,
        AssigneeType.OTTO,
    )


def assignee_actor(task: Task) -> Actor:
    a = task.assignee
    if a is not None and a.type == AssigneeType.EXPERT and a.id:
        return Actor.expert(a.id, a.name, a.avatar_url, a.color)
    return Actor.otto()


def kickoff_message(task: Task) -> str:
    parts = [f"You are assigned task {task.key}: {task.title}"]
    if task.description:
        parts.append(task.description)
    if task.parent_key:
        parts.append(f"This is a sub-task of {task.parent_key}.")
    parts.append(
        "Work it now. Use update_task_status / comment_on_task / request_input "
        "/ link_artifact as you go; when finished call "
        "update_task_status(status='done', outcome_summary=...). If you need "
        "the user, call request_input and stop.\n" + TOOL_HINT
    )
    return "\n\n".join(parts)
