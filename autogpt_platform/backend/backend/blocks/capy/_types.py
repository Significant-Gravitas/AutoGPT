"""Response models and enums for the Capy public API v1.

Capy's camelCase keys are snake-cased before validation, so responses validate
straight off the wire while block outputs (and their schemas) stay snake_case.
Models ignore unknown fields: Capy adds fields to its payloads, and a new field
must not break a running graph.
"""

from enum import Enum
from typing import Any, Optional

from pydantic import ConfigDict, FiniteFloat, model_validator
from pydantic.alias_generators import to_snake

from backend.sdk import BaseModel, Field


class _CapyModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _snake_case_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            return {
                to_snake(k) if isinstance(k, str) else k: v for k, v in data.items()
            }
        return data


class ThreadStatus(str, Enum):
    WORKING = "working"
    WAITING = "waiting"
    IDLE = "idle"
    FAILED = "failed"
    ARCHIVED = "archived"


# A thread in one of these states is still doing work; anything else means the
# agent has stopped and either delivered, asked a question, or failed.
ACTIVE_THREAD_STATUSES = {ThreadStatus.WORKING.value, ThreadStatus.WAITING.value}

# In these states the agent won't pick up a new message on its own.
STOPPED_THREAD_STATUSES = {ThreadStatus.FAILED.value, ThreadStatus.ARCHIVED.value}


class MachineSize(str, Enum):
    DEFAULT = ""
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    ULTRA = "ultra"
    HYPER = "hyper"
    BIGGUY = "bigguy"


class ReasoningEffort(str, Enum):
    DEFAULT = ""
    NONE = "none"
    INSTANT = "instant"
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    MAX = "max"


class MessageDelivery(str, Enum):
    INTERRUPT = "interrupt"
    STEER = "steer"
    QUEUE = "queue"


class ReviewTier(str, Enum):
    DEFAULT = ""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ProjectRepo(_CapyModel):
    repo_full_name: str
    base_branch: str = ""


class Project(_CapyModel):
    id: str
    name: str
    description: Optional[str] = None
    code: str = ""
    repos: list[ProjectRepo] = Field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


class Usage(_CapyModel):
    llm_credits: float = 0
    image_credits: float = 0
    vm_credits: float = 0
    total_credits: float = 0


class UsageTotals(_CapyModel):
    total_dollars: FiniteFloat


class UsageReport(_CapyModel):
    """The part of Capy's usage report the blocks read. A missing or
    non-finite total fails validation instead of reading as no spend."""

    totals: UsageTotals


class Thread(_CapyModel):
    id: str
    project_id: Optional[str] = None
    author_id: Optional[str] = None
    title: Optional[str] = None
    status: str
    archived: bool = False
    needs_you: bool = Field(
        default=False,
        description="True when the agent is waiting on an answer from a person",
    )
    last_model_id: Optional[str] = None
    usage: Usage = Field(default_factory=Usage)
    created_at: str = ""
    updated_at: str = ""
    last_activity_at: Optional[str] = None


class Message(_CapyModel):
    id: str
    source: str = Field(description="user, assistant or tool")
    text: str = ""
    author_name: Optional[str] = None
    model: Optional[str] = None
    created_at: str = ""


class MessagePage(_CapyModel):
    items: list[Message] = Field(default_factory=list)
    cursor: Optional[str] = None
    before_cursor: Optional[str] = None


class MessageReceipt(_CapyModel):
    id: str
    deduped: bool = False


class Task(_CapyModel):
    id: str
    thread_id: str = ""
    parent_id: str = ""
    task_path: str = ""
    title: Optional[str] = None
    status: str = ""
    archived: bool = False
    usage: Usage = Field(default_factory=Usage)
    created_at: str = ""
    updated_at: str = ""
    last_activity_at: Optional[str] = None


class ReviewStarted(_CapyModel):
    review_id: str
    request_id: str
    thread_id: str
    head_sha: str
    adopted: bool = False
    source_recorded: Optional[bool] = None


class ReviewFinding(_CapyModel):
    id: str
    kind: str = ""
    severity: Optional[str] = None
    confidence: Optional[str] = None
    category: Optional[str] = None
    summary: str = ""
    detail: Optional[str] = None
    suggestion: Optional[str] = None
    file: Optional[str] = None
    line: Optional[int] = None
    start_line: Optional[int] = None


class ReviewRound(_CapyModel):
    request_id: str
    review_id: str
    thread_id: Optional[str] = None
    repo: str = ""
    pr_number: int = 0
    head_sha: str = ""
    base_sha: str = ""
    status: str = Field(description="pending, running, completed, failed or stale")
    model: Optional[str] = None
    tier: Optional[str] = None
    title: Optional[str] = None
    failure_reason: Optional[str] = None
    created_at: str = ""
    settled_at: Optional[str] = None
    findings: list[ReviewFinding] = Field(default_factory=list)
