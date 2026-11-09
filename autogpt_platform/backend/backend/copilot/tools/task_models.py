"""Responses of the task tools (``tasks.py``)."""

from datetime import datetime

from prisma.enums import TaskCreator, TaskStatus
from pydantic import BaseModel, Field

from backend.data.task import Task

from .models import ResponseType, ToolResponseBase


class TaskInfo(BaseModel):
    """A task as the model reads it."""

    id: str
    title: str
    description: str | None = None
    status: TaskStatus
    owner_expert_id: str | None = Field(
        default=None, description="The expert doing the work; None means Otto."
    )
    created_by: TaskCreator
    session_ids: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime

    @classmethod
    def from_task(cls, task: Task) -> "TaskInfo":
        return cls(
            id=task.id,
            title=task.title,
            description=task.description,
            status=task.status,
            owner_expert_id=task.owner_expert_id,
            created_by=task.created_by,
            session_ids=task.session_ids,
            started_at=task.started_at,
            completed_at=task.completed_at,
            updated_at=task.updated_at,
        )


class TaskCreatedResponse(ToolResponseBase):
    type: ResponseType = ResponseType.TASK_CREATED
    task: TaskInfo


class TaskListResponse(ToolResponseBase):
    type: ResponseType = ResponseType.TASK_LIST
    tasks: list[TaskInfo] = Field(default_factory=list)
    count: int = 0


class TaskDetailsResponse(ToolResponseBase):
    type: ResponseType = ResponseType.TASK_DETAILS
    task: TaskInfo


class TaskUpdatedResponse(ToolResponseBase):
    type: ResponseType = ResponseType.TASK_UPDATED
    task: TaskInfo


class TaskArchivedResponse(ToolResponseBase):
    type: ResponseType = ResponseType.TASK_ARCHIVED
    task_id: str
