"""Tasks v1: each proposed hire is its own task on the board.

Proposing a hire (``hire_expert``, ``raise_expert``) puts "Hire <name>" on the
board, assigned to Otto; the user never holds a task. It is never dispatched,
since the hire itself happens in the chat that proposed it: the task links
that chat, holds the hire card (``tasks/approvals.py``) and asks for the
user's OK on it, so a waiting hire always says what it waits for. Each
proposal is tracked by its confirmation id (``ExpertChangeProposal``'s
``board_task_id``): confirming it completes its own task, and declining it
cancels that task. Proposing the same person again in the same chat keeps
their task and replaces the card. In a chat already working a task the hire
is part of that task, so nothing is added.

The board is bookkeeping here: a failure to track never blocks the hire.
"""

import logging

from backend.api.features.tasks.access import tasks_store
from backend.api.features.tasks.approvals_answer import remember_hire_chat
from backend.api.features.tasks.events import (
    publish_contract_update,
    publish_task_event,
)
from backend.api.features.tasks.models import (
    Actor,
    AssigneeType,
    AttachmentCreate,
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
_CLOSED = (TaskStatus.DONE, TaskStatus.CANCELLED)


def hire_line(name: str, role: str) -> str:
    return f"Hire {name} ({role})" if role else f"Hire {name}"


def waiting_reason(name: str) -> str:
    return f"Waiting for your OK on {name}'s hire card"


def tracked_note(task: Task | None) -> str:
    """What the proposal tells Otto about the board, so it doesn't add a
    second task for the same hire and closes this one when declined."""
    if task is None:
        return ""
    return (
        f" It is on the task board as {task.key}; don't create another task "
        "for it. Confirming this hire completes it. If the user declines the "
        f"card, call update_task_status with task_id {task.id} and status "
        "cancelled."
    )


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
        existing = await _open_hire_task(user_id, session.session_id, name)
        if existing is not None:
            return await _reword(user_id, existing, line, actor)
        return await _new_hire_task(user_id, session.session_id, name, line, actor)
    except Exception:
        logger.exception("Could not put a proposed hire on the task board")
        return None


async def wait_on_hire_card(
    user_id: str,
    session: ChatSession,
    task: Task | None,
    confirmation_id: str,
    kind: str,
    name: str,
    role: str,
    tagline: str = "",
) -> None:
    """Keep the hire card on its own board task, which then waits on the
    user's answer and says so. *task* is ``track_proposed_hire``'s."""
    if task is None:
        return
    try:
        store = tasks_store()
        actor = await caller_actor(user_id, session)
        card = CardApproval(
            confirmation_id=confirmation_id,
            kind=kind,
            name=name,
            role=role,
            tagline=tagline,
        )
        update = await store.contract_add_approval(user_id, task.id, card, actor)
        await publish_contract_update(user_id, update)
        if update.contract.ask is None:
            update = await store.contract_ask(
                user_id,
                task.id,
                waiting_reason(name),
                actor,
                why=f"Otto proposed {hire_line(name, role)}; answer the card "
                "here or in the chat that has it.",
            )
            await publish_contract_update(user_id, update)
        await publish_task_event(user_id, "task_updated", update.task, task_id=task.id)
    except Exception:
        logger.exception("Could not keep a hire card on its board task")


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
    user_id: str,
    session_id: str,
    confirmation_ids: list[str],
    board_task_id: str | None = None,
) -> None:
    """Take an answered card off the task its chat works, and off the hire's
    own board task. A hire carries two ids (plain / hire and start), so both
    are passed."""
    ids = [i for i in confirmation_ids if i]
    try:
        store = tasks_store()
        for task in await _card_holders(user_id, session_id, board_task_id):
            if not any(a.confirmation_id in ids for a in task.approvals):
                continue
            update = await store.contract_drop_approvals(user_id, task.id, ids)
            await publish_contract_update(user_id, update)
            await publish_task_event(
                user_id, "task_updated", update.task, task_id=task.id
            )
    except Exception:
        logger.exception("Could not take an answered card off its task")


async def complete_hire(user_id: str, task_id: str, name: str, role: str) -> None:
    """The confirmed hire's own task is done."""
    try:
        store = tasks_store()
        task: Task | None = await store.get_task(user_id, task_id)
        if task is None or task.status in _CLOSED:
            return
        if task.live.running:
            await store.set_live(task_id, False)
        task = await store.set_task_status(
            user_id,
            task_id,
            TaskStatus.DONE,
            Actor.system(),
            outcome_summary=f"Hired {name} ({role})" if role else f"Hired {name}",
        )
        await publish_task_event(user_id, "task_updated", task, task_id=task_id)
    except Exception:
        logger.exception("Could not complete an approved hire's task")


async def _card_holders(
    user_id: str, session_id: str, board_task_id: str | None
) -> list[Task]:
    store = tasks_store()
    holders: list[Task] = []
    found = await store.get_task_for_work_session(session_id)
    if found is not None and found[0] == user_id:
        holders.append(found[1])
    if board_task_id:
        task: Task | None = await store.get_task(user_id, board_task_id)
        if task is not None:
            holders.append(task)
    return holders


def _hire_key(session_id: str, name: str) -> str:
    return f"{_KEY_PREFIX}{session_id}:{name.strip().casefold()}"


async def _open_hire_task(user_id: str, session_id: str, name: str) -> Task | None:
    redis = await get_redis_async()
    task_id = await redis.get(_hire_key(session_id, name))
    if not task_id:
        return None
    task: Task | None = await tasks_store().get_task(user_id, task_id)
    if task is None or task.status in _CLOSED:
        return None
    return task


async def _new_hire_task(
    user_id: str, session_id: str, name: str, line: str, actor: Actor
) -> Task:
    store = tasks_store()
    detail = await store.create_task(
        user_id,
        TaskCreate(
            title=f"Hire {name}",
            description=line,
            assignee_type=AssigneeType.OTTO,
            status=TaskStatus.IN_REVIEW,
        ),
        actor,
    )
    if session_id:
        await store.add_attachment(
            user_id,
            detail.id,
            AttachmentCreate(
                kind="chat",
                title=f"The chat with {name}'s hire card",
                ref_id=session_id,
            ),
            actor,
        )
        await remember_hire_chat(detail.id, session_id)
        redis = await get_redis_async()
        await redis.set(_hire_key(session_id, name), detail.id, ex=_KEY_TTL_SECONDS)
    await publish_task_event(user_id, "task_created", detail, task_id=detail.id)
    return detail


async def _reword(user_id: str, task: Task, line: str, actor: Actor) -> Task:
    """A re-proposed hire keeps its task; only a changed role rewords it."""
    if task.description == line:
        return task
    detail = await tasks_store().update_task(
        user_id, task.id, TaskUpdate(description=line), actor
    )
    await publish_task_event(user_id, "task_updated", detail, task_id=task.id)
    return detail
