from datetime import date, datetime
from enum import Enum
from typing import Any

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
    CONTRACT_SET = "contract_set"
    GOAL_UPDATED = "goal_updated"
    STEP_UPDATED = "step_updated"
    EVIDENCE_ADDED = "evidence_added"
    CRITERION_MARKED = "criterion_marked"
    BLOCKER_REPORTED = "blocker_reported"
    ACTION_TAKEN = "action_taken"


class ContractState(str, Enum):
    WORKING = "working"
    NEEDS_YOU = "needs_you"
    DONE = "done"
    STUCK = "stuck"
    STOPPED = "stopped"


class CriterionStatus(str, Enum):
    PENDING = "pending"
    MET = "met"
    UNMET = "unmet"
    SKIPPED = "skipped"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


class Criterion(BaseModel):
    id: str
    text: str
    status: CriterionStatus = CriterionStatus.PENDING
    evidence_ids: list[str] = Field(default_factory=list)


class StepRow(BaseModel):
    """One of several runs of the same step going at once (five hires)."""

    id: str
    title: str
    status: StepStatus = StepStatus.PENDING


class Step(BaseModel):
    id: str
    title: str
    status: StepStatus = StepStatus.PENDING
    rows: list[StepRow] = Field(default_factory=list)


class Evidence(BaseModel):
    id: str
    kind: str
    title: str
    summary: str
    ref: str | None = None
    url: str | None = None
    criterion_ids: list[str] = Field(default_factory=list)
    attachment_id: str | None = None
    source_activity_id: str | None = None
    auto: bool = False


class Result(BaseModel):
    summary: str
    primary_evidence_id: str | None = None


class Ask(BaseModel):
    id: str
    question: str
    why: str | None = None
    options: list[str] = Field(default_factory=list)
    default_option: str | None = None
    blocking_step_id: str | None = None
    expires_at: datetime | None = None


class Blocker(BaseModel):
    done_so_far: str
    what_stopped_it: str
    # A whole sentence or a plain label, short enough for a card.
    short: str = ""
    cause: str = "unknown"
    next_actions: list[str] = Field(default_factory=list)


class CardApproval(BaseModel):
    """A hire or raise card proposed in the task's work chat, waiting on the
    user's Approve. Answered from the board through ``/approvals``."""

    confirmation_id: str
    kind: str = "hire"
    name: str
    role: str = ""
    tagline: str = ""


class CardDecision(BaseModel):
    confirmation_id: str = Field(min_length=1)
    approved: bool


class ApprovalsAnswer(BaseModel):
    decisions: list[CardDecision] = Field(min_length=1)


class Contract(BaseModel):
    goal: str
    # Set when the scope changed after work was planned, so the change shows.
    previous_goal: str | None = None
    done_when: list[Criterion] = Field(default_factory=list)
    plan: list[Step] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    result: Result | None = None
    ask: Ask | None = None
    blocker: Blocker | None = None
    approvals: list[CardApproval] = Field(default_factory=list)


class PlanStatus(str, Enum):
    PROPOSED = "proposed"
    CONFIRMED = "confirmed"
    CANCELLED = "cancelled"


HIDDEN_STATUSES = (TaskStatus.BACKLOG, TaskStatus.CANCELLED)
DEFAULT_BUDGET_CAP_USD = 2.0


class Assignee(BaseModel):
    type: AssigneeType
    id: str | None = None
    name: str
    avatar_url: str | None = None
    color: str | None = None


class Actor(BaseModel):
    type: ActorType
    id: str | None = None
    name: str
    avatar_url: str | None = None
    color: str | None = None

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
        avatar_url: str | None = None,
        color: str | None = None,
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
    current_step: str | None = None
    turn_started_at: datetime | None = None


class InboxState(str, Enum):
    RUNNING = "running"
    WAITING = "waiting"
    DONE = "done"
    FAILED = "failed"


class InboxStep(BaseModel):
    label: str
    time: datetime | None = None
    kind: str
    status: str | None = None


class InboxActivity(BaseModel):
    kind: str
    text: str
    actor_name: str | None = None
    at: datetime


class InboxTool(BaseModel):
    name: str
    title: str
    count: int = 0


class TaskInbox(BaseModel):
    state: InboxState
    needs_you: bool = False
    last_activity: InboxActivity | None = None
    steps: list[InboxStep] = Field(default_factory=list)
    tools_used: list[InboxTool] = Field(default_factory=list)


class TaskCounts(BaseModel):
    all: int = 0
    running: int = 0
    waiting: int = 0
    done: int = 0
    failed: int = 0
    needs_you: int = 0
    by_status: dict[str, int] = Field(default_factory=dict)
    by_state: dict[str, int] = Field(default_factory=dict)


class Task(BaseModel):
    id: str
    number: int
    key: str
    title: str
    description: str | None = None
    status: TaskStatus
    priority: TaskPriority
    assignee: Assignee | None = None
    due_date: date | None = None
    labels: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    parent_key: str | None = None
    outcome_summary: str | None = None
    work_session_id: str | None = None
    budget_cap_usd: float | None = None
    review_required: bool = False
    metrics: TaskMetrics = Field(default_factory=TaskMetrics)
    subtask_counts: SubtaskCounts = Field(default_factory=SubtaskCounts)
    pending_input: PendingInput | None = None
    live: TaskLive = Field(default_factory=TaskLive)
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    plan_id: str | None = None
    elapsed_ms: int | None = None
    inbox: TaskInbox | None = None
    state: ContractState = ContractState.WORKING
    raw_status: str = "queued"
    # What a card shows: the one line about now, and the open ask or blocker.
    now: str = ""
    ask: Ask | None = None
    blocker: Blocker | None = None
    approvals: list[CardApproval] = Field(default_factory=list)
    contract_version: int = 0


class TaskParentRef(BaseModel):
    id: str
    key: str
    title: str


class Attachment(BaseModel):
    id: str
    kind: str
    title: str
    ref_id: str | None = None
    url: str | None = None
    created_at: datetime
    added_by: Actor


class TaskDetail(Task):
    subtasks: list[Task] = Field(default_factory=list)
    attachments: list[Attachment] = Field(default_factory=list)
    parent: TaskParentRef | None = None


class ActivityEvent(BaseModel):
    id: str
    task_id: str
    created_at: datetime
    actor: Actor
    kind: ActivityKind
    body: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ExpertOption(BaseModel):
    id: str
    name: str
    role: str | None = None
    avatar_url: str | None = None
    color: str | None = None
    busy_task_id: str | None = None
    busy_task_key: str | None = None


class AssigneeRef(BaseModel):
    type: AssigneeType
    id: str | None = None


class PlanStep(BaseModel):
    title: str
    description: str | None = None
    assignee: Assignee | None = None


class Plan(BaseModel):
    id: str
    status: PlanStatus
    goal: str
    budget_usd: float | None = None
    steps: list[PlanStep] = Field(default_factory=list)
    split: bool = True
    session_id: str | None = None
    parent_task_id: str | None = None
    task_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class _TaskFields(BaseModel):
    description: str | None = None
    status: TaskStatus | None = None
    priority: TaskPriority | None = None
    assignee_type: AssigneeType | None = None
    assignee_id: str | None = None
    assignee: AssigneeRef | None = None
    due_date: date | None = None
    labels: list[str] | None = None
    parent_id: str | None = None
    budget_cap_usd: float | None = Field(default=None, ge=0)
    budget_usd: float | None = Field(default=None, ge=0)
    review_required: bool | None = None

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
    title: str | None = Field(default=None, min_length=1, max_length=500)
    outcome_summary: str | None = None


class CommentCreate(BaseModel):
    body: str = Field(min_length=1)


class AnswerCreate(BaseModel):
    answer: str = Field(min_length=1)
    input_id: str | None = None


class AttachmentCreate(BaseModel):
    kind: str = Field(pattern="^(file|chat|run|url|text)$")
    title: str = Field(min_length=1)
    ref_id: str | None = None
    url: str | None = None


class PlanConfirm(BaseModel):
    steps: list[PlanStep] | None = None


class TaskFilters(BaseModel):
    status: list[TaskStatus] | None = None
    assignee_type: AssigneeType | None = None
    assignee_id: str | None = None
    priority: list[TaskPriority] | None = None
    due: str | None = None
    parent_id: str | None = None
    top_level: bool = False
    mine: bool = False
    include_hidden: bool = False
    q: str | None = None
    limit: int = 200


class TaskListResponse(BaseModel):
    tasks: list[Task]
    counts: TaskCounts | None = None


class ActivityListResponse(BaseModel):
    events: list[ActivityEvent]
    next_cursor: str | None = None


class AskAnswer(BaseModel):
    answer: str = Field(min_length=1)
    ask_id: str | None = None


class TaskActionName(str, Enum):
    """The actions with their own handling; any other is passed on as the
    user's instruction."""

    RETRY_FAILED_PART = "retry_failed_part"
    SKIP = "skip"
    ALLOW = "allow"
    DO_IN_CHAT = "do_in_chat"
    CHANGE_GOAL = "change_goal"
    TAKE_OVER = "take_over"
    STOP = "stop"
    ACCEPT = "accept"


class TaskActionRequest(BaseModel):
    goal: str | None = Field(default=None, min_length=1, max_length=500)
    note: str | None = None


class ExpertListResponse(BaseModel):
    experts: list[ExpertOption]


class TasksEnabledResponse(BaseModel):
    enabled: bool


class PlanConfirmResponse(BaseModel):
    plan: Plan
    parent_task: TaskDetail
