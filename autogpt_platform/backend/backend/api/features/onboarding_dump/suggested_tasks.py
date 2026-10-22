"""The Home checklist built from the greeting's suggested prompts.

The greeting is read once and retires on the first message; the prompts
under it are worked through, so each keeps its own state on the row's
``suggestedPrompts`` JSON: open, sent (in progress), or dismissed.
"""

import logging
from datetime import UTC, datetime

from pydantic import ValidationError

from backend.api.features.onboarding_dump import db, intro
from backend.api.features.onboarding_dump.models import (
    SuggestedTask,
    SuggestedTasksResponse,
    SuggestedTaskState,
    SuggestedTasksUpdate,
)
from backend.util.exceptions import ConflictError, NotFoundError

logger = logging.getLogger(__name__)

# Which states a row may move to from which. A sent row never reopens: its
# chat exists, so hiding it and undoing that brings it back as sent.
_ALLOWED_FROM: dict[SuggestedTaskState, frozenset[SuggestedTaskState]] = {
    "open": frozenset({"dismissed"}),
    "dismissed": frozenset({"open", "sent"}),
    "sent": frozenset({"open", "dismissed"}),
}

# Each attempt is one read and one conditional write; losing three in a row
# would take three other writes to this user's row inside one request.
_WRITE_ATTEMPTS = 3


async def get_suggested_tasks(user_id: str) -> SuggestedTasksResponse:
    dump = await db.get_dump(user_id)
    stored = _stored_tasks(dump.suggestedPrompts if dump else None)
    if stored:
        return SuggestedTasksResponse(tasks=stored, stored=True)
    return SuggestedTasksResponse(tasks=_generic_tasks(), stored=False)


async def update_suggested_tasks(
    user_id: str, update: SuggestedTasksUpdate
) -> SuggestedTasksResponse:
    """Move the named rows to ``update.state`` and return the whole list.

    A row with nothing stored yet is showing the generic starters, which
    are stored here so the change has somewhere to live. A transition the
    row does not allow (reopening a sent task, say) is skipped rather than
    refused, so a stale tab gets the current list back instead of an error.
    """
    for _ in range(_WRITE_ATTEMPTS):
        dump = await db.get_dump(user_id) or await db.ensure_dump(user_id)
        stored = _stored_tasks(dump.suggestedPrompts)
        tasks = stored or _generic_tasks()
        if set(update.task_ids) - {task.id for task in tasks}:
            raise NotFoundError("Unknown suggestion")
        changed = [_apply(task, update) for task in tasks]
        if changed == tasks:
            return SuggestedTasksResponse(tasks=tasks, stored=bool(stored))
        if await db.replace_suggested_prompts(
            user_id,
            dump.updatedAt,
            [task.model_dump(mode="json") for task in changed],
        ):
            return SuggestedTasksResponse(tasks=changed, stored=True)
    logger.warning(
        f"Suggested tasks: gave up after {_WRITE_ATTEMPTS} conflicting writes"
    )
    raise ConflictError("Suggestions changed, try again")


def _apply(task: SuggestedTask, update: SuggestedTasksUpdate) -> SuggestedTask:
    if task.id not in update.task_ids:
        return task
    # Undo of a hidden row that had started a chat: back to sent, chat kept.
    state = "sent" if update.state == "open" and task.session_id else update.state
    if task.state not in _ALLOWED_FROM[state]:
        return task
    return task.model_copy(
        update={
            "state": state,
            "session_id": update.session_id or task.session_id,
            "updated_at": datetime.now(UTC),
        }
    )


def _stored_tasks(raw: object) -> list[SuggestedTask]:
    """Rehydrate the stored list, dropping anything malformed.

    The ``id`` a row was first given is its list index, kept on every write
    after, so dropping a malformed neighbour never renumbers it.
    """
    if not isinstance(raw, list):
        return []
    tasks = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        try:
            task = SuggestedTask.model_validate({"id": str(index), **item})
        except ValidationError:
            continue
        if task.icon not in intro.PROMPT_ICONS:
            task.icon = intro.DEFAULT_PROMPT_ICON
        tasks.append(task)
    return tasks


def _generic_tasks() -> list[SuggestedTask]:
    return [
        SuggestedTask(id=str(index), **prompt.model_dump())
        for index, prompt in enumerate(intro.fallback_prompts())
    ]
