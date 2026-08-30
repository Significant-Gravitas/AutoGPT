from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

TaskStatus = Literal[
    "QUEUED",
    "WORKING",
    "WAITING_USER",
    "DONE",
    "FAILED",
    "CANCELLED",
]

TaskAcceptance = Literal["PENDING", "ACCEPTED", "REJECTED"]

TaskCreatedBy = Literal["USER", "EXPERT", "SCHEDULE", "DREAM"]

# A task is "open" while it can still change on its own. Cancel cascades to
# exactly these, and the Tasks tab splits active-vs-history on them.
OPEN_TASK_STATUSES: tuple[TaskStatus, ...] = ("QUEUED", "WORKING", "WAITING_USER")

TASK_TITLE_MAX_LENGTH = 200
TASK_SPEC_MAX_LENGTH = 20_000
TASK_OUTCOME_MAX_LENGTH = 2_000

# Bounds the list endpoint so one user's history can't turn into an unbounded
# scan; the Tasks tab pages by nothing else today.
MAX_TASKS_PER_PAGE = 50


class TaskExpertRef(BaseModel):
    """The expert a task is owned by. Null owner = Autopilot."""

    id: str
    name: str
    avatar_url: str | None
    role: str


class TaskRunRef(BaseModel):
    """One execution started to satisfy the task."""

    execution_id: str
    graph_id: str
    library_agent_id: str | None
    agent_name: str
    status: str
    started_at: datetime | None
    ended_at: datetime | None
    link: str | None


class TaskAmendment(BaseModel):
    """A scope change recorded against a live task."""

    at: datetime
    by: str
    note: str


class DelegatedTask(BaseModel):
    id: str
    title: str
    spec: str
    status: TaskStatus
    acceptance: TaskAcceptance
    created_by_type: TaskCreatedBy
    created_by_id: str | None
    owner: TaskExpertRef | None
    parent_task_id: str | None
    root_task_id: str | None
    origin_session_id: str | None
    ancestor_expert_ids: list[str]
    handoff_count: int
    revision_count: int
    # Credits (100 = $1) burned by this task's executions.
    spend_total: int
    outcome_summary: str | None
    amendments: list[TaskAmendment]
    created_at: datetime
    updated_at: datetime
    runs: list[TaskRunRef] = []


class DelegatedTaskDetail(BaseModel):
    """A task plus one level of children — enough for the detail drawer's
    timeline without recursing the whole delegation tree."""

    task: DelegatedTask
    children: list[DelegatedTask]


class CreateTaskRequest(BaseModel):
    """Internal shape used by the copilot wiring; not exposed as a route."""

    title: str = Field(min_length=1, max_length=TASK_TITLE_MAX_LENGTH)
    spec: str = Field(max_length=TASK_SPEC_MAX_LENGTH)
    owner_id: str | None = None
    origin_session_id: str | None = None
    created_by_type: TaskCreatedBy = "USER"
    created_by_id: str | None = None

    # "before" so the length bounds apply to the trimmed title and a blank
    # one fails with this message rather than the generic min_length error.
    @field_validator("title", mode="before")
    @classmethod
    def strip_title(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped:
            raise ValueError("Task title must not be blank")
        return stripped
