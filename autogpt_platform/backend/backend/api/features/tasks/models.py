from datetime import date, datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


class TaskStatus(str, Enum):
    BACKLOG = "backlog"
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    IN_REVIEW = "in_review"
    DONE = "done"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class AssigneeType(str, Enum):
    USER = "user"
    EXPERT = "expert"
    OTTO = "otto"


class ActorType(str, Enum):
    USER = "user"
    EXPERT = "expert"
    OTTO = "otto"
    SYSTEM = "system"


class ActivityKind(str, Enum):
    CREATED = "created"
    STATUS_CHANGED = "status_changed"
    ASSIGNED = "assigned"
    COMMENT = "comment"
    TURN = "turn"
    INPUT_REQUESTED = "input_requested"
    INPUT_ANSWERED = "input_answered"
    ATTACHMENT_ADDED = "attachment_added"
    SUBTASK_CREATED = "subtask_created"
    OUTCOME = "outcome"
    PLAN_CONFIRMED = "plan_confirmed"
    SYSTEM = "system"


class PlanStatus(str, Enum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


HIDDEN_STATUSES = (TaskStatus.BACKLOG, TaskStatus.CANCELLED)
DEFAULT_BUDGET_CAP_USD = 2.0


class Assignee(BaseModel):
    type: AssigneeType
    id: Optional[str] = None
    name: str
    avatar_url: Optional[str] = None
    color: Optional[str] = None


class Actor(BaseModel):
    type: ActorType
    id: Optional[str] = None
    name: str
    avatar_url: Optional[str] = None
    color: Optional[str] = None

    @classmethod
    def user(cls, user_id: str) -> "Actor":
        return cls(type=ActorType.USER, id=user_id, name="Me")

    @classmethod
    def system(cls) -> "Actor":
        return cls(type=ActorType.SYSTEM, id=None, name="System")

    @classmethod
    def otto(cls) -> "Actor":
        return cls(type=ActorType.OTTO, id=None, name="Otto")

    @classmethod
    def expert(
        cls,
        expert_id: str,
        name: str,
        avatar_url: Optional[str] = None,
        color: Optional[str] = None,
    ) -> "Actor":
        return cls(
            type=ActorType.EXPERT,
            id=expert_id,
            name=name,
            avatar_url=avatar_url,
            color=color,
        )


class TaskMetrics(BaseModel):
    turns: int = 0
    tool_calls: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0


class SubtaskCounts(BaseModel):
    total: int = 0
    done: int = 0


class PendingInput(BaseModel):
    id: str
    question: str
    options: list[str] = Field(default_factory=list)
    asked_by: Actor
    asked_at: datetime


class TaskLive(BaseModel):
    running: bool = False
    current_step: Optional[str] = None
    turn_started_at: Optional[datetime] = None


class Task(BaseModel):
    id: str
    number: int
    key: str
    title: str
    description: Optional[str] = None
    status: TaskStatus
    priority: TaskPriority
    assignee: Optional[Assignee] = None
    due_date: Optional[date] = None
    labels: list[str] = Field(default_factory=list)
    parent_id: Optional[str] = None
    parent_key: Optional[str] = None
    outcome_summary: Optional[str] = None
    work_session_id: Optional[str] = None
    budget_cap_usd: Optional[float] = None
    review_required: bool = False
    metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    subtask_counts: SubtaskCounts = Field(default_factory=SubtaskCounts)
    pending_input: Optional[PendingInput] = None
    live: TaskLive = Field(default_factory=TaskLive)
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    plan_id: Optional[str] = None


class TaskParentRef(BaseModel):
    id: str
    key: str
    title: str


class Attachment(BaseModel):
    id: str
    kind: str
    title: str
    ref_id: Optional[str] = None
    url: Optional[str] = None
    created_at: datetime
    added_by: Actor


class TaskDetail(Task):
    subtasks: list[Task] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)
    parent: Optional[TaskParentRef] = None


class ActivityEvent(BaseModel):
    id: str
    task_id: str
    created_at: datetime
    actor: Actor
    kind: ActivityKind
    body: Optional[str] = None
    data: dict[str, Any] = Field(default_factory=dict)


class ExpertOption(BaseModel):
    id: str
    name: str
    role: Optional[str] = None
    avatar_url: Optional[str] = None
    color: Optional[str] = None
    busy_task_id: Optional[str] = None
    busy_task_key: Optional[str] = None


class AssigneeRef(BaseModel):
    type: AssigneeType
    id: Optional[str] = None


class PlanStep(BaseModel):
    title: str
    description: Optional[str] = None
    assignee: Optional[Assignee] = None


class Plan(BaseModel):
    id: str
    status: PlanStatus
    goal: str
    budget_usd: Optional[float] = None
    steps: list[PlanStep] = Field(default_factory=list)
    session_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    task_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class _TaskFields(BaseModel):
    description: Optional[str] = None
    status: Optional[TaskStatus] = None
    priority: Optional[TaskPriority] = None
    assignee_type: Optional[AssigneeType] = None
    assignee_id: Optional[str] = None
    assignee: Optional[AssigneeRef] = None
    due_date: Optional[date] = None
    labels: Optional[list[str]] = None
    parent_id: Optional[str] = None
    budget_cap_usd: Optional[float] = Field(default=None, ge=0)
    budget_usd: Optional[float] = Field(default=None, ge=0)
    review_required: Optional[bool] = None

    @model_validator(mode="after")
    def apply_aliases(self):
        if self.assignee is not None:
            self.assignee_type = self.assignee.type
            self.assignee_id = self.assignee.id
            if "assignee_type" not in self.model_fields_set:
                self.model_fields_set.add("assignee_type")
            if "assignee_id" not in self.model_fields_set:
                self.model_fields_set.add("assignee_id")
        if self.budget_usd is not None and self.budget_cap_usd is None:
            self.budget_cap_usd = self.budget_usd
            self.model_fields_set.add("budget_cap_usd")
        return self


class TaskCreate(_TaskFields):
    title: str = Field(min_length=1, max_length=500)


class TaskUpdate(_TaskFields):
    title: Optional[str] = Field(default=None, min_length=1, max_length=500)
    outcome_summary: Optional[str] = None


class CommentCreate(BaseModel):
    body: str = Field(min_length=1)


class AnswerCreate(BaseModel):
    answer: str = Field(min_length=1)
    input_id: Optional[str] = None


class AttachmentCreate(BaseModel):
    kind: str = Field(pattern="^(file|chat|run|url|text)$")
    title: str = Field(min_length=1)
    ref_id: Optional[str] = None
    url: Optional[str] = None


class PlanConfirm(BaseModel):
    steps: Optional[list[PlanStep]] = None


class TaskFilters(BaseModel):
    status: Optional[list[TaskStatus]] = None
    assignee_type: Optional[AssigneeType] = None
    assignee_id: Optional[str] = None
    priority: Optional[list[TaskPriority]] = None
    due: Optional[str] = None
    parent_id: Optional[str] = None
    top_level: bool = False
    mine: bool = False
    include_hidden: bool = False
    q: Optional[str] = None
    limit: int = 200


class TaskListResponse(BaseModel):
    tasks: list[Task]


class ActivityListResponse(BaseModel):
    events: list[ActivityEvent]


class ExpertListResponse(BaseModel):
    experts: list[ExpertOption]


class TasksEnabledResponse(BaseModel):
    enabled: bool


class PlanConfirmResponse(BaseModel):
    plan: Plan
    parent_task: TaskDetail
