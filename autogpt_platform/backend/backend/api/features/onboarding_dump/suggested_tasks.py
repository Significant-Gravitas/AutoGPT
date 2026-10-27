"""The Home checklist built from the greeting's suggested prompts.

The greeting is read once and retires on the first message; the prompts
under it are worked through, so each keeps its own state on the row's
``suggestedPrompts`` JSON: open, sent (in progress), or dismissed.

A task may have an owner (``expert_template_id``); every list this module
hands out says which of the user's hired experts that is (``expert_id``),
read fresh each time, so a fired expert's tasks go back to Otto.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from backend.api.features.experts import experts_db
from backend.api.features.experts.experts_db import list_templates
from backend.api.features.experts.models import Expert
from backend.api.features.onboarding_dump import db, intro, role_tasks, service
from backend.api.features.onboarding_dump.models import (
    SuggestedPrompt,
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
    stored = stored_tasks(dump.suggestedPrompts if dump else None)
    tasks = stored or await _default_tasks(user_id, storing=False)
    return SuggestedTasksResponse(
        tasks=await with_expert_ids(user_id, tasks), stored=bool(stored)
    )


async def update_suggested_tasks(
    user_id: str, update: SuggestedTasksUpdate
) -> SuggestedTasksResponse:
    """Move the named rows to ``update.state`` and return the whole list.

    A row with nothing stored yet is showing the default starters, which
    are stored here so the change has somewhere to live. A transition the
    row does not allow (reopening a sent task, say) is skipped rather than
    refused, so a stale tab gets the current list back instead of an error.
    """
    for _ in range(_WRITE_ATTEMPTS):
        dump = await db.get_dump(user_id) or await db.ensure_dump(user_id)
        stored = stored_tasks(dump.suggestedPrompts)
        tasks = stored or await _default_tasks(user_id, storing=True)
        if set(update.task_ids) - {task.id for task in tasks}:
            raise NotFoundError("Unknown suggestion")
        changed = [_apply(task, update) for task in tasks]
        if changed == tasks:
            return SuggestedTasksResponse(
                tasks=await with_expert_ids(user_id, tasks), stored=bool(stored)
            )
        if await db.replace_suggested_prompts(
            user_id, dump.updatedAt, [_storable(task) for task in changed]
        ):
            return SuggestedTasksResponse(
                tasks=await with_expert_ids(user_id, changed), stored=True
            )
    logger.warning(
        f"Suggested tasks: gave up after {_WRITE_ATTEMPTS} conflicting writes"
    )
    raise ConflictError("Suggestions changed, try again")


async def store_if_empty(
    user_id: str, prompts: list[SuggestedPrompt]
) -> list[SuggestedTask]:
    """The stored list, storing ``prompts`` first when there is none.

    So the ids a first read hands out are the ones every later read and
    write uses. A conditional write that loses means another request
    stored a list first, and that one stands.
    """
    for _ in range(_WRITE_ATTEMPTS):
        dump = await db.get_dump(user_id) or await db.ensure_dump(user_id)
        stored = stored_tasks(dump.suggestedPrompts)
        if stored:
            return stored
        tasks = as_tasks(prompts)
        if await db.replace_suggested_prompts(
            user_id, dump.updatedAt, [_storable(task) for task in tasks]
        ):
            return tasks
    raise ConflictError("Suggestions changed, try again")


async def with_expert_ids(
    user_id: str, tasks: list[SuggestedTask]
) -> list[SuggestedTask]:
    """``tasks`` with each owner's hired expert, or none for Otto's.

    A task whose expert is not hired, or is archived, reads as Otto's to
    the client.
    """
    if not any(task.expert_template_id for task in tasks):
        return tasks
    hired = await roster_by_template(user_id)
    return [
        task.model_copy(update={"expert_id": _hired_id(hired, task.expert_template_id)})
        for task in tasks
    ]


async def roster_by_template(user_id: str) -> dict[str, Expert]:
    """The user's active hires by the template each was hired from.

    A roster that cannot be read reads as empty, so every owner reads as
    unhired rather than failing the list.
    """
    try:
        experts = await experts_db.list_experts(user_id, with_metrics=False)
    except Exception as e:
        logger.warning(f"Suggested tasks: roster read failed: {e}")
        return {}
    return {
        expert.source_template_id: expert
        for expert in experts
        if expert.source_template_id
    }


def as_tasks(prompts: list[SuggestedPrompt]) -> list[SuggestedTask]:
    return [
        SuggestedTask(id=str(index), **prompt.model_dump())
        for index, prompt in enumerate(prompts)
    ]


def stored_tasks(raw: object) -> list[SuggestedTask]:
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
            task = SuggestedTask.model_validate(
                {"id": str(index), **item, "expert_id": None}
            )
        except ValidationError:
            continue
        if task.icon not in intro.PROMPT_ICONS:
            task.icon = intro.DEFAULT_PROMPT_ICON
        tasks.append(task)
    return tasks


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


def _storable(task: SuggestedTask) -> dict[str, Any]:
    # ``expert_id`` is read fresh each time, never stored.
    return task.model_dump(mode="json", exclude={"expert_id"})


async def _default_tasks(user_id: str, *, storing: bool) -> list[SuggestedTask]:
    """The role's starters for a user who picked one, the generic set otherwise.

    The role is read so that a failed read raises, on every path: a list
    built without it is a different list under the same ids. The roster
    read raises only when the list is about to be stored; for a read, a
    failed one only leaves the tasks without owners.
    """
    user_role, pain_points = await service.read_wizard_answers(user_id)
    if not user_role:
        return as_tasks(intro.fallback_prompts())
    templates = await list_templates() if storing else await service.safe_templates()
    prompts, _ = role_tasks.role_plan(user_role, pain_points, templates)
    return as_tasks(prompts)


def _hired_id(hired: dict[str, Expert], template_id: str | None) -> str | None:
    expert = hired.get(template_id or "")
    return expert.id if expert else None
