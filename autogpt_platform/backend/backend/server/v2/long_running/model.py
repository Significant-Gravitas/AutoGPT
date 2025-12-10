"""
Pydantic models for long-running agent session API.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """Status of a long-running agent session."""

    INITIALIZING = "initializing"
    READY = "ready"
    WORKING = "working"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class FeatureStatus(str, Enum):
    """Status of a feature."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSING = "passing"
    FAILING = "failing"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class FeatureCategory(str, Enum):
    """Category of a feature."""

    FUNCTIONAL = "functional"
    UI = "ui"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    INFRASTRUCTURE = "infrastructure"


class ProgressEntryType(str, Enum):
    """Type of progress entry."""

    SESSION_START = "session_start"
    SESSION_END = "session_end"
    FEATURE_START = "feature_start"
    FEATURE_COMPLETE = "feature_complete"
    FEATURE_FAILED = "feature_failed"
    CODE_CHANGE = "code_change"
    TEST_RUN = "test_run"
    BUG_FIX = "bug_fix"
    ENVIRONMENT_SETUP = "environment_setup"
    GIT_COMMIT = "git_commit"
    NOTE = "note"
    ERROR = "error"


# Request models
class CreateSessionRequest(BaseModel):
    """Request to create a new long-running session."""

    project_name: str = Field(description="Name of the project")
    project_description: str = Field(
        description="Detailed description of the project"
    )
    working_directory: str = Field(
        description="Directory where the project will be created"
    )
    features: list[dict[str, Any]] = Field(
        default=[],
        description="Optional list of features to initialize with",
    )


class UpdateSessionRequest(BaseModel):
    """Request to update a session."""

    status: Optional[SessionStatus] = None
    current_session_id: Optional[str] = None
    current_feature_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


class CreateFeatureRequest(BaseModel):
    """Request to create a new feature."""

    feature_id: str = Field(description="Unique identifier for the feature")
    category: FeatureCategory = Field(default=FeatureCategory.FUNCTIONAL)
    description: str = Field(description="Description of the feature")
    steps: list[str] = Field(default=[], description="Verification steps")
    priority: int = Field(default=5, ge=1, le=10)
    dependencies: list[str] = Field(default=[])


class UpdateFeatureRequest(BaseModel):
    """Request to update a feature."""

    status: Optional[FeatureStatus] = None
    notes: Optional[str] = None
    updated_by_session: Optional[str] = None


class CreateProgressEntryRequest(BaseModel):
    """Request to create a progress entry."""

    agent_session_id: str = Field(description="Session ID that created this entry")
    entry_type: ProgressEntryType
    title: str = Field(description="Short title")
    description: Optional[str] = None
    feature_id: Optional[str] = None
    git_commit_hash: Optional[str] = None
    files_changed: list[str] = Field(default=[])
    metadata: dict[str, Any] = Field(default={})


# Response models
class FeatureResponse(BaseModel):
    """Response model for a feature."""

    id: str
    feature_id: str
    category: FeatureCategory
    description: str
    steps: list[str]
    status: FeatureStatus
    priority: int
    dependencies: list[str]
    notes: Optional[str]
    updated_by_session: Optional[str]
    created_at: datetime
    updated_at: datetime


class ProgressEntryResponse(BaseModel):
    """Response model for a progress entry."""

    id: str
    agent_session_id: str
    entry_type: ProgressEntryType
    title: str
    description: Optional[str]
    feature_id: Optional[str]
    git_commit_hash: Optional[str]
    files_changed: list[str]
    metadata: dict[str, Any]
    created_at: datetime


class SessionResponse(BaseModel):
    """Response model for a session."""

    id: str
    project_name: str
    project_description: str
    status: SessionStatus
    current_session_id: Optional[str]
    session_count: int
    working_directory: str
    feature_list_path: Optional[str]
    progress_log_path: Optional[str]
    init_script_path: Optional[str]
    git_repo_initialized: bool
    current_feature_id: Optional[str]
    environment_variables: dict[str, Any]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class SessionDetailResponse(SessionResponse):
    """Detailed session response with features and progress."""

    features: list[FeatureResponse]
    recent_progress: list[ProgressEntryResponse]
    feature_summary: dict[str, int]


class SessionListResponse(BaseModel):
    """Response model for listing sessions."""

    sessions: list[SessionResponse]
    total: int
    page: int
    page_size: int


class FeatureListResponse(BaseModel):
    """Response model for listing features."""

    features: list[FeatureResponse]
    total: int
    summary: dict[str, int]
