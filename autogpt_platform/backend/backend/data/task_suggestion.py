"""Tasks Otto drafts for the user to start or set aside.

A suggestion is an ordinary ``Task`` row with ``isSuggested`` set: it stays
off the board until the user starts it (the run route clears the flag) and
leaves the list when the user dismisses it (``dismissedAt``) or when a newer
draft replaces it (archived). Every query is scoped to the owning user.
"""

from datetime import UTC, datetime, timedelta

import prisma.models
from prisma.enums import TaskCreator, TaskStatus
from prisma.types import TaskCreateInput
from pydantic import BaseModel

from backend.data.task import (
    DESCRIPTION_MAX_CHARS,
    LINE_MAX_CHARS,
    TITLE_MAX_CHARS,
    Task,
    get_task,
)
from backend.util.exceptions import NotFoundError

# How many set-aside titles the drafting compares against.
DISMISSED_TITLES_LIMIT = 50
# How long a suggestion retired unstarted stays off limits for new drafts.
RETIRED_WINDOW = timedelta(days=14)


class SuggestedTaskInput(BaseModel):
    """One drafted task, cleaned and checked by the caller."""

    title: str
    goal: str
    done_when: str = ""
    brief: str
    reason: str
    owner_expert_id: str | None = None


async def create_suggested_task(user_id: str, draft: SuggestedTaskInput) -> Task:
    """Store ``draft`` as a to-do suggestion. An owner that is not one of the
    user's live experts is dropped, so Otto owns it instead."""
    title = " ".join(draft.title.split())[:TITLE_MAX_CHARS]
    if not title:
        raise ValueError("A suggested task needs a title.")
    owner = draft.owner_expert_id
    if owner and not await _is_live_expert(owner, user_id):
        owner = None
    data: TaskCreateInput = {
        "userId": user_id,
        "title": title,
        "description": draft.brief.strip()[:DESCRIPTION_MAX_CHARS] or None,
        "goal": _line(draft.goal),
        "doneWhen": _line(draft.done_when),
        "ownerExpertId": owner,
        "createdBy": TaskCreator.OTTO,
        "status": TaskStatus.TODO,
        "isSuggested": True,
        "suggestedReason": _line(draft.reason),
    }
    row = await prisma.models.Task.prisma().create(data=data)
    task = await get_task(row.id, user_id)
    if task is None:
        raise NotFoundError(f"Task {row.id} not found")
    return task


async def dismiss_task_suggestion(task_id: str, user_id: str) -> Task:
    """Set a live suggestion aside. Dismissing one twice is a no-op; a task
    that was never suggested is refused."""
    task = await get_task(task_id, user_id)
    if task is None:
        raise NotFoundError(f"Task {task_id} not found")
    if not task.is_suggested:
        raise ValueError("Only a suggested task can be dismissed.")
    await prisma.models.Task.prisma().update_many(
        where={
            "id": task_id,
            "userId": user_id,
            "isSuggested": True,
            "dismissedAt": None,
        },
        data={"dismissedAt": datetime.now(UTC)},
    )
    return task


async def dismissed_suggestion_titles(
    user_id: str, limit: int = DISMISSED_TITLES_LIMIT
) -> list[str]:
    """Titles of suggestions that must not be drafted again, newest first:
    every one the user dismissed, and the ones retired unstarted within
    ``RETIRED_WINDOW``, so a lapsed idea is not back the next morning."""
    retired_since = datetime.now(UTC) - RETIRED_WINDOW
    rows = await prisma.models.Task.prisma().find_many(
        where={
            "userId": user_id,
            "isSuggested": True,
            "OR": [
                {"dismissedAt": {"not": None}},
                {"isArchived": True, "updatedAt": {"gte": retired_since}},
            ],
        },
        order={"updatedAt": "desc"},
        take=max(1, limit),
    )
    return [row.title for row in rows]


async def retire_stale_suggestions(user_id: str, created_before: datetime) -> int:
    """Archive live suggestions drafted before ``created_before`` so newer
    drafts can take their place. Returns how many were retired."""
    return await prisma.models.Task.prisma().update_many(
        where={
            "userId": user_id,
            "isSuggested": True,
            "dismissedAt": None,
            "isArchived": False,
            "createdAt": {"lt": created_before},
        },
        # Stamped so the retired window counts from today, not from the
        # last edit.
        data={"isArchived": True, "updatedAt": datetime.now(UTC)},
    )


def _line(text: str) -> str | None:
    return " ".join(text.split())[:LINE_MAX_CHARS] or None


async def _is_live_expert(expert_id: str, user_id: str) -> bool:
    expert = await prisma.models.Expert.prisma().find_first(
        where={"id": expert_id, "ownerUserId": user_id, "isArchived": False}
    )
    return expert is not None
