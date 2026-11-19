"""The task list a user pages through: filters, a title search, and a page
that says whether more follow. Scoped to the user like every task read."""

import prisma.models
from prisma.enums import TaskStatus
from prisma.types import TaskWhereInput
from pydantic import BaseModel, Field

from backend.data.task import _WITH_SESSIONS, LIST_LIMIT_MAX, Task

SEARCH_MAX_CHARS = 200


class TaskFilter(BaseModel):
    """Which of the user's live tasks to list; unset fields filter nothing."""

    statuses: list[TaskStatus] = Field(default_factory=list)
    owner_expert_id: str | None = None
    owned_by_otto: bool = False
    session_id: str | None = None
    include_subtasks: bool = False
    scheduled: bool = False
    search: str | None = None


class TaskPage(BaseModel):
    tasks: list[Task]
    has_more: bool = Field(default=False, description="More tasks follow this page.")
    next_offset: int | None = Field(
        default=None, description="The offset of the next page, when there is one."
    )


async def list_task_page(
    user_id: str, filters: TaskFilter, limit: int, offset: int
) -> TaskPage:
    """One page of the user's tasks, most recently updated first (soonest
    start first with ``scheduled``). Reads one row past the page to know
    whether more follow."""
    size = max(1, min(limit, LIST_LIMIT_MAX))
    offset = max(0, offset)
    rows = await prisma.models.Task.prisma().find_many(
        where=_where(user_id, filters),
        include=_WITH_SESSIONS,
        order={"startAt": "asc"} if filters.scheduled else {"updatedAt": "desc"},
        take=size + 1,
        skip=offset,
    )
    has_more = len(rows) > size
    return TaskPage(
        tasks=[Task.from_db(row) for row in rows[:size]],
        has_more=has_more,
        next_offset=offset + size if has_more else None,
    )


def _where(user_id: str, filters: TaskFilter) -> TaskWhereInput:
    where: TaskWhereInput = {
        "userId": user_id,
        "isArchived": False,
        "isSuggested": False,
    }
    if not filters.include_subtasks:
        where["parentTaskId"] = None
    if filters.statuses:
        where["status"] = {"in": filters.statuses}
    if filters.owner_expert_id:
        where["ownerExpertId"] = filters.owner_expert_id
    elif filters.owned_by_otto:
        where["ownerExpertId"] = None
    if filters.session_id:
        where["Sessions"] = {"some": {"chatSessionId": filters.session_id}}
    if filters.scheduled:
        where["status"] = TaskStatus.TODO
        where["startAt"] = {"not": None}
    search = " ".join((filters.search or "").split())[:SEARCH_MAX_CHARS]
    if search:
        where["title"] = {"contains": search, "mode": "insensitive"}
    return where
