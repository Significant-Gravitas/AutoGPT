"""What happens when a task's work turn ends (called from stream_registry)."""

import asyncio
import logging
import time
from datetime import UTC, datetime
from typing import Any

from backend.api.features.chat.session_attention import turn_attention
from backend.api.features.tasks.access import tasks_store
from backend.api.features.tasks.contract import (
    tool_error,
    turn_blocker_reason,
    unrecovered_failure,
)
from backend.api.features.tasks.dispatch import on_task_assigned, resume_task
from backend.api.features.tasks.events import (
    publish_contract_update,
    publish_task_event,
)
from backend.api.features.tasks.live_step import step_label
from backend.api.features.tasks.models import (
    ActivityEvent,
    ActivityKind,
    Actor,
    AssigneeType,
    Task,
    TaskStatus,
)
from backend.copilot.model import ChatMessage, get_chat_session

logger = logging.getLogger(__name__)

OTTO_WAKE_INTERVAL_SECONDS = 60
_PREVIEW = 200
_last_wake: dict[str, float] = {}
_pending_children: dict[str, dict[str, str]] = {}
_wake_tasks: dict[str, asyncio.Task] = {}
# A failed call to one of these is the worker fumbling the board, not the
# work failing, so it never becomes the task's blocker.
_BOOKKEEPING_TOOLS = frozenset(
    {
        "update_task",
        "add_evidence",
        "ask_user",
        "update_task_status",
        "comment_on_task",
        "request_input",
        "link_artifact",
        "list_my_tasks",
        "create_task",
        "propose_plan",
        "confirm_task_plan",
    }
)


async def finish_work_turn(session_id: str, status: str) -> None:
    store = tasks_store()
    found = await store.get_task_for_work_session(session_id)
    if found is None:
        return
    user_id, task = found
    turn = await _running_turn(user_id, task.id)
    metrics = await _turn_metrics(session_id, user_id, turn)
    task = await store.add_turn_metrics(
        task.id,
        metrics["tool_calls"],
        metrics["tokens"],
        metrics["cost_usd"],
        metrics["duration_ms"],
    )
    activity = None
    if turn is not None:
        activity = await store.update_activity(
            turn.id,
            data={
                **metrics,
                "status": "completed" if status == "completed" else "failed",
            },
        )
    await store.set_live(task.id, False)
    task = await store.get_task(user_id, task.id) or task
    await publish_task_event(user_id, "task_updated", task, task_id=task.id)
    if activity is not None:
        await publish_task_event(user_id, "task_activity", task, activity, task.id)
    await _fill_contract(user_id, task, session_id, status, turn)
    task = await store.get_task(user_id, task.id) or task
    waiting = await _ended_waiting_on_user(session_id, user_id)
    task = await _settle_unfinished(user_id, task, status, waiting)
    await _start_next_for_expert(user_id, task)
    await _maybe_wake_otto(user_id, task)


async def _fill_contract(
    user_id: str,
    task: Task,
    session_id: str,
    status: str,
    turn: ActivityEvent | None,
) -> None:
    """Bring the task's record up to date from what the turn did: its logged
    effects become evidence, and a failure it never got past (or the turn
    failing) becomes the blocker. Best-effort: never breaks the turn's end."""
    try:
        data = turn.data if turn else {}
        started = data.get("started_at")
        since = datetime.fromisoformat(started) if started else datetime.now(UTC)
        session = await get_chat_session(session_id, user_id)
        messages = (
            session.messages[int(data.get("start_message_count") or 0) :]
            if session
            else []
        )
        failure = unrecovered_failure(_turn_calls(messages), _BOOKKEEPING_TOOLS)
        reason = turn_blocker_reason(status != "completed", failure)
        updates = await tasks_store().contract_fill_from_turn(
            user_id,
            task.id,
            session_id,
            since,
            reason[0] if reason else None,
            reason[1] if reason else None,
        )
        for update in updates:
            await publish_contract_update(user_id, update)
    except Exception:
        logger.warning("Could not fill the contract of %s", task.id, exc_info=True)


def _turn_calls(messages: list[ChatMessage]) -> list[tuple[str, str | None]]:
    """``(tool, error or None)`` for each call the turn made, in order."""
    names: dict[str, str] = {}
    calls: list[tuple[str, str | None]] = []
    for message in messages:
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                fn = call.get("function") or {}
                names[call.get("id") or ""] = fn.get("name") or call.get("name") or ""
        elif message.role == "tool":
            name = names.get(message.tool_call_id or "", "tool")
            calls.append((name, tool_error(message.content)))
    return calls


async def _ended_waiting_on_user(session_id: str, user_id: str) -> bool:
    """Whether the turn stopped on a card or a question for the user."""
    session = await get_chat_session(session_id, user_id)
    if session is None:
        return False
    last_user = max(
        (i for i, m in enumerate(session.messages) if m.role == "user"), default=-1
    )
    turn = [(m.role, m.content) for m in session.messages[last_user + 1 :]]
    return turn_attention(turn) == "needs_you"


async def _settle_unfinished(
    user_id: str, task: Task, status: str, waiting: bool = False
) -> Task:
    """A turn that ended with the task still open: a failed turn needs the
    user, one that stopped on a card or a question waits for them, and a
    quiet expert turn goes to review. A quiet Otto turn stays open, since
    Otto may be waiting on its sub-tasks."""
    a = task.assignee
    if task.status != TaskStatus.IN_PROGRESS or a is None:
        return task
    failed = status != "completed"
    if a.type != AssigneeType.EXPERT and not failed and not waiting:
        return task
    store = tasks_store()
    if failed:
        note = f"{a.name}'s work turn stopped with an error. Restart it when ready."
    elif waiting:
        note = f"{a.name} is waiting for you in the task's chat."
    else:
        note = f"{a.name} finished a turn without marking the task done. Review it."
    new_status = TaskStatus.BLOCKED if failed or waiting else TaskStatus.IN_REVIEW
    fresh = await store.set_task_status(
        user_id, task.id, new_status, Actor.system(), note=note
    )
    activity = await store.add_activity(
        user_id, task.id, ActivityKind.SYSTEM, Actor.system(), body=note
    )
    await publish_task_event(user_id, "task_needs_input", fresh, activity, task.id)
    return fresh


async def _running_turn(user_id: str, task_id: str) -> ActivityEvent | None:
    events = await tasks_store().list_activity(user_id, task_id)
    turns = [
        e
        for e in events
        if e.kind == ActivityKind.TURN and e.data.get("status") == "running"
    ]
    return turns[-1] if turns else None


async def _turn_metrics(
    session_id: str, user_id: str, turn: ActivityEvent | None
) -> dict[str, Any]:
    data = turn.data if turn else {}
    now = datetime.now(UTC)
    started_raw = data.get("started_at")
    started = datetime.fromisoformat(started_raw) if started_raw else now
    session = await get_chat_session(session_id, user_id)
    messages = (
        session.messages[int(data.get("start_message_count") or 0) :] if session else []
    )
    cards = _tool_call_cards(messages)
    prompt = sum(u.prompt_tokens for u in session.usage) if session else 0
    completion = sum(u.completion_tokens for u in session.usage) if session else 0
    input_tokens = max(0, prompt - int(data.get("start_prompt_tokens") or 0))
    output_tokens = max(0, completion - int(data.get("start_completion_tokens") or 0))
    cost_usd = 0.0
    try:
        usage = await tasks_store().get_turn_usage(session_id, started)
        cost_usd = usage.cost_usd
        if not input_tokens and not output_tokens:
            input_tokens, output_tokens = usage.input_tokens, usage.output_tokens
    except Exception:
        logger.warning("Could not read turn cost for %s", session_id, exc_info=True)
    summary = next(
        (m.content for m in reversed(messages) if m.role == "assistant" and m.content),
        None,
    )
    return {
        "tool_calls": len(cards),
        "tokens": input_tokens + output_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost_usd, 6),
        "duration_ms": max(0, int((now - started).total_seconds() * 1000)),
        "summary": summary[:_PREVIEW] if summary else None,
        "tool_call_cards": cards,
    }


def _tool_call_cards(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    results = {m.tool_call_id: m.content for m in messages if m.role == "tool"}
    cards = []
    for message in messages:
        if message.role != "assistant" or not message.tool_calls:
            continue
        for call in message.tool_calls:
            fn = call.get("function") or {}
            name = fn.get("name") or call.get("name") or "tool"
            call_id = call.get("id") or ""
            output = results.get(call_id)
            cards.append(
                {
                    "id": call_id,
                    "name": name,
                    "title": step_label(name),
                    "status": "completed" if call_id in results else "running",
                    "input_preview": _preview(fn.get("arguments")),
                    "output_preview": _preview(output),
                }
            )
    return cards


def _preview(value: Any) -> str | None:
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    return text[:_PREVIEW]


async def _start_next_for_expert(user_id: str, task: Task) -> None:
    a = task.assignee
    if a is None or a.type != AssigneeType.EXPERT or not a.id:
        return
    store = tasks_store()
    if await store.get_busy_task_for_expert(user_id, a.id, None):
        return
    waiting = await store.list_waiting_tasks_for_expert(user_id, a.id)
    if waiting:
        await on_task_assigned(user_id, waiting[0].id)


async def _maybe_wake_otto(user_id: str, task: Task) -> None:
    if task.status != TaskStatus.DONE or not task.parent_id:
        return
    parent = await tasks_store().get_task(user_id, task.parent_id)
    if parent is None or parent.assignee is None:
        return
    if parent.assignee.type != AssigneeType.OTTO:
        return
    if parent.status in (TaskStatus.DONE, TaskStatus.CANCELLED):
        return
    line = f"- {task.key} {task.title}: {task.outcome_summary or 'done'}"
    _pending_children.setdefault(parent.id, {})[task.id] = line
    if parent.id in _wake_tasks and not _wake_tasks[parent.id].done():
        return
    wait = OTTO_WAKE_INTERVAL_SECONDS - (
        time.monotonic() - _last_wake.get(parent.id, 0)
    )
    _wake_tasks[parent.id] = asyncio.create_task(
        _wake_otto_later(user_id, parent.id, max(0.0, wait))
    )


async def _wake_otto_later(user_id: str, parent_id: str, delay: float) -> None:
    try:
        if delay:
            await asyncio.sleep(delay)
        lines = list(_pending_children.pop(parent_id, {}).values())
        if not lines:
            return
        _last_wake[parent_id] = time.monotonic()
        parent = await tasks_store().get_task(user_id, parent_id)
        if parent is None:
            return
        counts = parent.subtask_counts
        message = (
            f"Sub-tasks of {parent.key} finished:\n"
            + "\n".join(lines)
            + f"\n\n{counts.done} of {counts.total} sub-tasks are done. Review the "
            "results. Create follow-up sub-tasks if needed, or, when everything "
            f"is done, call update_task(task_id='{parent.id}', "
            "result={'summary': ...})."
        )
        await resume_task(user_id, parent_id, message)
    except Exception:
        logger.exception("Waking Otto for task %s failed", parent_id)
