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
from typing import Any

from backend.api.features.tasks.access import tasks_store
from backend.api.features.tasks.events import (
    publish_contract_update,
    publish_task_event,
)
from backend.api.features.tasks.live_step import remember_work_session
from backend.api.features.tasks.models import (
    DEFAULT_BUDGET_CAP_USD,
    ActivityKind,
    Actor,
    ActorType,
    AssigneeType,
    Task,
    TaskStatus,
)
from backend.copilot.model import create_chat_session, get_chat_session
from backend.copilot.sdk.session_waiter import run_copilot_turn_via_queue

logger = logging.getLogger(__name__)

TASK_KICKOFF_KIND = "task_kickoff"
MAX_TURNS_PER_TASK = 20
RESUME_WINDOW_SECONDS = 120
MAX_RESUMES_PER_WINDOW = 6
_STARTABLE = (TaskStatus.BACKLOG, TaskStatus.TODO)
_recent_starts: dict[str, list[float]] = {}

TOOL_HINT = (
    "Task tools: update_task, add_evidence, ask_user, comment_on_task, "
    "list_my_tasks, create_task. If a tool is not in your tool list, call it "
    "through run_capability with id 'tool:<name>'."
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
        await start_task_turn(
            user_id, task, kickoff_message(task), kickoff_metadata(task)
        )
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


async def start_task_turn(
    user_id: str,
    task: Task,
    message: str,
    message_metadata: dict[str, Any] | None = None,
) -> bool:
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
    return await _queue_turn(
        user_id, task, session_id, message, activity.id, message_metadata
    )


async def _queue_turn(
    user_id: str,
    task: Task,
    session_id: str,
    message: str,
    activity_id: str,
    message_metadata: dict[str, Any] | None,
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
            message_metadata=message_metadata,
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
    """Stop the task on a guard, with the guard's reason as its blocker so the
    card says why instead of a generic "stopped"."""
    store = tasks_store()
    await store.set_live(task.id, False)
    update = await store.contract_update_task(
        user_id,
        task.id,
        {"blocker": {"what_stopped_it": note, "cause": "limit"}},
        Actor.system(),
    )
    await publish_contract_update(user_id, update)


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
    origin = "interactive" if await _user_drives(user_id, task) else "automation"
    session = await create_chat_session(
        user_id, dry_run=False, expert_id=expert_id, origin=origin
    )
    await tasks_store().set_work_session(task.id, session.session_id)
    return session.session_id


async def _user_drives(user_id: str, task: Task) -> bool:
    """Whether the task's chat counts as one the user drives, so it may do
    what only a user-driven chat may (propose hires, each still waiting on
    the user's approval). Two things make it so: the user wrote the task and
    gave it to Otto, or the user pressed Allow on a blocker that a
    user-driven chat would get past. A task Otto or an expert wrote stays an
    automation until then."""
    events = await tasks_store().list_activity(user_id, task.id)
    if any(
        e.kind == ActivityKind.ACTION_TAKEN
        and e.data.get("action") == "allow"
        and e.actor.type == ActorType.USER
        for e in events
    ):
        return True
    if task.assignee is None or task.assignee.type != AssigneeType.OTTO:
        return False
    created = next((e for e in events if e.kind == ActivityKind.CREATED), None)
    return created is not None and created.actor.type == ActorType.USER


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
        "Work it now. Start with update_task (goal, done_when, plan), keep its "
        "steps current, record proof with add_evidence, and finish with "
        "update_task(result=...). If you need the user, call ask_user.\n" + TOOL_HINT
    )
    return "\n\n".join(parts)


def resume_brief(task: Task, contract_brief: str, note: str) -> str:
    """The first message of a fresh work chat for a task that already has
    work done: the kickoff, the record so far, and why it starts again."""
    return (
        f"{kickoff_message(task)}\n\nWork so far (keep it; don't redo finished "
        f"steps):\n{contract_brief}\n\n{note}"
    )


def kickoff_metadata(task: Task) -> dict[str, Any]:
    """Lets the chat show the kickoff as the task it starts, not as a message
    the user typed; the model still reads the text."""
    return {
        "kind": TASK_KICKOFF_KIND,
        "task_id": task.id,
        "task_key": task.key,
        "task_title": task.title,
    }
