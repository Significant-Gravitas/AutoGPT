"""Tasks v1: a proposed hire is one task on the board.

Proposing a hire (``hire_expert``, ``raise_expert``) puts "Hire <name>" on the
board, assigned to Otto and waiting on the user's answer to the hire card;
the user never holds a task. It is never dispatched, since the hire itself
happens in the chat that proposed it. More hires
proposed in the same chat join that task as checklist items instead of
adding tasks. Approving a card ticks its hire off, and the last tick
completes the task. In a chat already working a task the hire is part of
that task, so nothing is added.

The board is bookkeeping here: a failure to track never blocks the hire.
"""

import logging

from backend.api.features.tasks.access import tasks_store
from backend.api.features.tasks.events import (
    publish_contract_update,
    publish_task_event,
)
from backend.api.features.tasks.models import (
    Actor,
    AssigneeType,
    CardApproval,
    Task,
    TaskCreate,
    TaskStatus,
    TaskUpdate,
)
from backend.copilot.model import ChatSession
from backend.data.redis_client import get_redis_async
from backend.util.feature_flag import Flag, is_feature_enabled

from .tasks_helpers import caller_actor, find_work_task

logger = logging.getLogger(__name__)

_KEY_PREFIX = "copilot:hire_task:"
_KEY_TTL_SECONDS = 7 * 24 * 3600
_OPEN = "- [ ] "
_DONE = "- [x] "
_CLOSED = (TaskStatus.DONE, TaskStatus.CANCELLED)


def hire_line(name: str, role: str) -> str:
    return f"Hire {name} ({role})" if role else f"Hire {name}"


def tracked_note(task: Task | None) -> str:
    """What the proposal tells Otto about the board, so it doesn't add a
    second task for the same hire."""
    if task is None:
        return ""
    return f" It is on the task board as {task.key}; don't create another task for it."


async def track_proposed_hire(
    user_id: str, session: ChatSession, name: str, role: str
) -> Task | None:
    """The board task this proposed hire is tracked on, or None when the
    board is off or the chat's own task already covers it."""
    try:
        if not await is_feature_enabled(Flag.TASKS_V1, user_id, default=False):
            return None
        if await find_work_task(user_id, session) is not None:
            return None
        actor = await caller_actor(user_id, session)
        line = hire_line(name, role)
        existing = await _session_hire_task(user_id, session.session_id)
        if existing is not None:
            return await _add_hire(user_id, existing, line, actor)
        return await _new_hire_task(user_id, session.session_id, name, line, actor)
    except Exception:
        logger.exception("Could not put a proposed hire on the task board")
        return None


async def park_on_work_task(
    user_id: str,
    session: ChatSession,
    confirmation_id: str,
    kind: str,
    name: str,
    role: str,
    tagline: str = "",
) -> Task | None:
    """In a chat working a task, keep the card on that task so the user can
    approve it from the board (``tasks/approvals.py``). The task, or None."""
    try:
        if not await is_feature_enabled(Flag.TASKS_V1, user_id, default=False):
            return None
        task = await find_work_task(user_id, session)
        if task is None:
            return None
        card = CardApproval(
            confirmation_id=confirmation_id,
            kind=kind,
            name=name,
            role=role,
            tagline=tagline,
        )
        actor = await caller_actor(user_id, session)
        update = await tasks_store().contract_add_approval(
            user_id, task.id, card, actor
        )
        await publish_contract_update(user_id, update)
        await publish_task_event(user_id, "task_updated", update.task, task_id=task.id)
        return task
    except Exception:
        logger.exception("Could not keep a proposed hire on its task")
        return None


def parked_note(task: Task | None) -> str:
    """What the proposal tells the worker when the card is on its task."""
    if task is None:
        return ""
    return (
        f" The card is also on task {task.key} in the user's Needs you list, "
        "where they can approve it; their answer arrives as a new message "
        "with the Approved line. Don't call ask_user about this card."
    )


async def forget_card(
    user_id: str, session_id: str, confirmation_ids: list[str]
) -> None:
    """Take an answered card off the task its chat works. A hire carries two
    ids (plain / hire and start), so both are passed."""
    try:
        found = await tasks_store().get_task_for_work_session(session_id)
        if found is None:
            return
        owner, task = found
        ids = [i for i in confirmation_ids if i]
        if owner != user_id or not any(
            a.confirmation_id in ids for a in task.approvals
        ):
            return
        update = await tasks_store().contract_drop_approvals(user_id, task.id, ids)
        await publish_contract_update(user_id, update)
        await publish_task_event(user_id, "task_updated", update.task, task_id=task.id)
    except Exception:
        logger.exception("Could not take an answered card off its task")


async def complete_hire(user_id: str, task_id: str, name: str, role: str) -> None:
    """Tick the approved hire off its task; the last tick completes it."""
    try:
        store = tasks_store()
        task: Task | None = await store.get_task(user_id, task_id)
        if task is None or task.status in _CLOSED:
            return
        lines = (task.description or "").splitlines()
        ticked = _tick(lines, hire_line(name, role))
        actor = Actor.system()
        task = await store.update_task(
            user_id, task.id, TaskUpdate(description="\n".join(ticked)), actor
        )
        if not any(line.startswith(_OPEN) for line in ticked):
            hired = [
                line.removeprefix(_DONE) for line in ticked if line.startswith(_DONE)
            ]
            task = await store.set_task_status(
                user_id,
                task_id,
                TaskStatus.DONE,
                actor,
                outcome_summary="; ".join(hired),
            )
        await publish_task_event(user_id, "task_updated", task, task_id=task_id)
    except Exception:
        logger.exception("Could not tick an approved hire off its task")


async def _session_hire_task(user_id: str, session_id: str) -> Task | None:
    redis = await get_redis_async()
    task_id = await redis.get(_KEY_PREFIX + session_id)
    if not task_id:
        return None
    task: Task | None = await tasks_store().get_task(user_id, task_id)
    if task is None or task.status in _CLOSED:
        return None
    return task


async def _new_hire_task(
    user_id: str, session_id: str, name: str, line: str, actor: Actor
) -> Task:
    detail = await tasks_store().create_task(
        user_id,
        TaskCreate(
            title=f"Hire {name}",
            description=_OPEN + line,
            assignee_type=AssigneeType.OTTO,
            status=TaskStatus.IN_REVIEW,
        ),
        actor,
    )
    redis = await get_redis_async()
    await redis.set(_KEY_PREFIX + session_id, detail.id, ex=_KEY_TTL_SECONDS)
    await publish_task_event(user_id, "task_created", detail, task_id=detail.id)
    return detail


async def _add_hire(user_id: str, task: Task, line: str, actor: Actor) -> Task:
    lines = (task.description or "").splitlines()
    if any(existing.endswith(line) for existing in lines):
        return task
    lines.append(_OPEN + line)
    hires = sum(1 for existing in lines if existing.startswith((_OPEN, _DONE)))
    detail = await tasks_store().update_task(
        user_id,
        task.id,
        TaskUpdate(title=f"Hire {hires} experts", description="\n".join(lines)),
        actor,
    )
    await publish_task_event(user_id, "task_updated", detail, task_id=task.id)
    return detail


def _tick(lines: list[str], line: str) -> list[str]:
    ticked = list(lines)
    for i, existing in enumerate(ticked):
        if existing == _OPEN + line:
            ticked[i] = _DONE + line
            break
    return ticked
