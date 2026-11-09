"""Tasks: one piece of work with its own record, apart from the chat it was
asked in.

A chat can carry several tasks and a task can span several chats, linked
through ``TaskSession``. Every read and write is scoped to the owning user;
a task, expert or chat another user owns reads as not found.
"""

from datetime import UTC, datetime

import prisma.models
from prisma.enums import TaskCreator, TaskStatus
from prisma.errors import UniqueViolationError
from prisma.types import (
    TaskCreateInput,
    TaskInclude,
    TaskSessionCreateInput,
    TaskUpdateInput,
    TaskWhereInput,
)
from pydantic import BaseModel, Field

from backend.data.db import transaction
from backend.util.exceptions import NotFoundError

TITLE_MAX_CHARS = 200
DESCRIPTION_MAX_CHARS = 10_000
LIST_LIMIT_MAX = 100

_WITH_SESSIONS: TaskInclude = {"Sessions": True}


class Task(BaseModel):
    """A task as callers see it, with the chats it was worked on in."""

    id: str
    user_id: str
    title: str
    description: str | None = None
    status: TaskStatus
    owner_expert_id: str | None = Field(
        default=None, description="The expert doing the work; None means Otto."
    )
    created_by: TaskCreator
    created_by_expert_id: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    is_archived: bool = False
    organization_id: str | None = None
    team_id: str | None = None
    session_ids: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_db(cls, row: prisma.models.Task) -> "Task":
        return cls(
            id=row.id,
            user_id=row.userId,
            title=row.title,
            description=row.description,
            status=TaskStatus(row.status),
            owner_expert_id=row.ownerExpertId,
            created_by=TaskCreator(row.createdBy),
            created_by_expert_id=row.createdByExpertId,
            started_at=row.startedAt,
            completed_at=row.completedAt,
            is_archived=row.isArchived,
            organization_id=row.organizationId,
            team_id=row.teamId,
            session_ids=[link.chatSessionId for link in row.Sessions or []],
            created_at=row.createdAt,
            updated_at=row.updatedAt,
        )


class TaskPatch(BaseModel):
    """A partial update. ``None`` leaves a field as it is, so the patch reads
    the same after an RPC hop that sends every field."""

    title: str | None = None
    description: str | None = Field(
        default=None, description="An empty string clears the description."
    )
    status: TaskStatus | None = None
    owner_expert_id: str | None = None
    assign_to_otto: bool = Field(
        default=False, description="Clear the owner expert so Otto owns the task."
    )


async def create_task(
    user_id: str,
    title: str,
    created_by: TaskCreator,
    description: str | None = None,
    owner_expert_id: str | None = None,
    created_by_expert_id: str | None = None,
    session_id: str | None = None,
    organization_id: str | None = None,
    team_id: str | None = None,
) -> Task:
    """Create a task, attached to ``session_id`` when one is given."""
    title = _clean_title(title)
    description = _clean_description(description)
    if owner_expert_id:
        await _require_expert(owner_expert_id, user_id)
    if session_id:
        await _require_session(session_id, user_id)

    data: TaskCreateInput = {
        "userId": user_id,
        "title": title,
        "description": description,
        "ownerExpertId": owner_expert_id,
        "createdBy": created_by,
        "createdByExpertId": created_by_expert_id,
        "organizationId": organization_id,
        "teamId": team_id,
    }
    async with transaction() as tx:
        row = await prisma.models.Task.prisma(tx).create(data=data)
        if session_id:
            await prisma.models.TaskSession.prisma(tx).create(
                data=_link(row.id, session_id)
            )
    return await _get_or_raise(row.id, user_id)


async def get_task(task_id: str, user_id: str) -> Task | None:
    """The user's task, or None when it is missing, archived or not theirs."""
    row = await prisma.models.Task.prisma().find_first(
        where={"id": task_id, "userId": user_id, "isArchived": False},
        include=_WITH_SESSIONS,
    )
    return Task.from_db(row) if row else None


async def list_tasks(
    user_id: str,
    status: TaskStatus | None = None,
    owner_expert_id: str | None = None,
    session_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Task]:
    """The user's live tasks, most recently updated first."""
    where: TaskWhereInput = {"userId": user_id, "isArchived": False}
    if status:
        where["status"] = status
    if owner_expert_id:
        where["ownerExpertId"] = owner_expert_id
    if session_id:
        where["Sessions"] = {"some": {"chatSessionId": session_id}}
    rows = await prisma.models.Task.prisma().find_many(
        where=where,
        include=_WITH_SESSIONS,
        order={"updatedAt": "desc"},
        take=max(1, min(limit, LIST_LIMIT_MAX)),
        skip=max(0, offset),
    )
    return [Task.from_db(row) for row in rows]


async def update_task(task_id: str, user_id: str, patch: TaskPatch) -> Task:
    """Apply ``patch``; a status change stamps or clears the task's times."""
    current = await _get_or_raise(task_id, user_id)
    data: TaskUpdateInput = {}
    if patch.title is not None:
        data["title"] = _clean_title(patch.title)
    if patch.description is not None:
        data["description"] = _clean_description(patch.description)
    if patch.assign_to_otto:
        data["OwnerExpert"] = {"disconnect": True}
    elif patch.owner_expert_id:
        await _require_expert(patch.owner_expert_id, user_id)
        data["OwnerExpert"] = {"connect": {"id": patch.owner_expert_id}}
    if patch.status is not None and patch.status != current.status:
        data["status"] = patch.status
        data.update(_status_times(current, patch.status, datetime.now(UTC)))
    if not data:
        return current

    await prisma.models.Task.prisma().update(where={"id": task_id}, data=data)
    return await _get_or_raise(task_id, user_id)


async def archive_task(task_id: str, user_id: str) -> bool:
    """Soft-delete the task. False when there was no live task to archive."""
    archived = await prisma.models.Task.prisma().update_many(
        where={"id": task_id, "userId": user_id, "isArchived": False},
        data={"isArchived": True},
    )
    return archived > 0


async def move_session_tasks(
    session_id: str,
    user_id: str,
    from_statuses: list[TaskStatus],
    to_status: TaskStatus,
) -> int:
    """Move the chat's live tasks in ``from_statuses`` to ``to_status``, e.g.
    to NEEDS_INPUT when Otto asks the user something. Returns how many moved."""
    return await prisma.models.Task.prisma().update_many(
        where={
            "userId": user_id,
            "isArchived": False,
            "status": {"in": from_statuses},
            "Sessions": {"some": {"chatSessionId": session_id}},
        },
        data={"status": to_status},
    )


async def attach_task_session(task_id: str, session_id: str, user_id: str) -> Task:
    """Link a chat to a task; linking it twice is a no-op."""
    await _get_or_raise(task_id, user_id)
    await _require_session(session_id, user_id)
    try:
        await prisma.models.TaskSession.prisma().create(data=_link(task_id, session_id))
    except UniqueViolationError:
        pass
    return await _get_or_raise(task_id, user_id)


async def detach_task_session(task_id: str, session_id: str, user_id: str) -> Task:
    """Unlink a chat from a task; unlinking one that is not linked is a no-op."""
    await _get_or_raise(task_id, user_id)
    await prisma.models.TaskSession.prisma().delete_many(
        where={"taskId": task_id, "chatSessionId": session_id}
    )
    return await _get_or_raise(task_id, user_id)


def _link(task_id: str, session_id: str) -> TaskSessionCreateInput:
    return {"taskId": task_id, "chatSessionId": session_id}


def _status_times(current: Task, status: TaskStatus, now: datetime) -> TaskUpdateInput:
    if status == TaskStatus.TODO:
        return {"startedAt": None, "completedAt": None}
    if status in (TaskStatus.IN_PROGRESS, TaskStatus.NEEDS_INPUT):
        return {"startedAt": current.started_at or now, "completedAt": None}
    return {"completedAt": now}


def _clean_title(title: str) -> str:
    title = " ".join(title.split())
    if not title:
        raise ValueError("A task needs a title.")
    if len(title) > TITLE_MAX_CHARS:
        raise ValueError(f"A task title is at most {TITLE_MAX_CHARS} characters.")
    return title


def _clean_description(description: str | None) -> str | None:
    description = (description or "").strip()
    if len(description) > DESCRIPTION_MAX_CHARS:
        raise ValueError(
            f"A task description is at most {DESCRIPTION_MAX_CHARS} characters."
        )
    return description or None


async def _get_or_raise(task_id: str, user_id: str) -> Task:
    task = await get_task(task_id, user_id)
    if task is None:
        raise NotFoundError(f"Task {task_id} not found")
    return task


async def _require_expert(expert_id: str, user_id: str) -> None:
    expert = await prisma.models.Expert.prisma().find_first(
        where={"id": expert_id, "ownerUserId": user_id, "isArchived": False}
    )
    if expert is None:
        raise NotFoundError(f"Expert {expert_id} not found")


async def _require_session(session_id: str, user_id: str) -> None:
    session = await prisma.models.ChatSession.prisma().find_first(
        where={"id": session_id, "userId": user_id}
    )
    if session is None:
        raise NotFoundError(f"Chat {session_id} not found")
