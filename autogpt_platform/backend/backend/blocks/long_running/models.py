"""
Data models for long-running agent framework.

Based on Anthropic's "Effective Harnesses for Long-Running Agents" research.
https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents

The core concepts:
1. Feature List - A comprehensive list of features to be implemented, each with status
2. Progress Log - A log of what each agent session has accomplished
3. Session State - The overall state of a long-running agent project
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class FeatureCategory(str, Enum):
    """Category types for features."""

    FUNCTIONAL = "functional"
    UI = "ui"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    INFRASTRUCTURE = "infrastructure"


class FeatureStatus(str, Enum):
    """Status values for features."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSING = "passing"
    FAILING = "failing"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class FeatureListItem(BaseModel):
    """
    A single feature in the feature list.

    Features are end-to-end descriptions of functionality that should be testable.
    Each feature has verification steps that define how to test it.
    """

    id: str = Field(description="Unique identifier for the feature")
    category: FeatureCategory = Field(
        default=FeatureCategory.FUNCTIONAL, description="Category of the feature"
    )
    description: str = Field(description="Human-readable description of the feature")
    steps: list[str] = Field(
        default_factory=list, description="Verification steps to test the feature"
    )
    status: FeatureStatus = Field(
        default=FeatureStatus.PENDING, description="Current status of the feature"
    )
    priority: int = Field(
        default=5, ge=1, le=10, description="Priority level (1=highest, 10=lowest)"
    )
    dependencies: list[str] = Field(
        default_factory=list, description="IDs of features this depends on"
    )
    notes: str = Field(
        default="", description="Additional notes about the feature or blockers"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow, description="When the feature was last updated"
    )
    updated_by_session: Optional[str] = Field(
        default=None, description="Session ID that last updated this feature"
    )


class FeatureList(BaseModel):
    """
    The complete feature list for a long-running agent project.

    This file is written by the initializer agent and read by coding agents.
    Coding agents should ONLY modify the status field and notes - never remove features.
    """

    project_name: str = Field(description="Name of the project")
    project_description: str = Field(description="Description of the project goal")
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the feature list was created"
    )
    features: list[FeatureListItem] = Field(
        default_factory=list, description="List of all features"
    )

    def get_next_feature(self) -> Optional[FeatureListItem]:
        """Get the next feature to work on (highest priority pending/failing)."""
        candidates = [
            f
            for f in self.features
            if f.status in (FeatureStatus.PENDING, FeatureStatus.FAILING)
        ]
        if not candidates:
            return None
        # Sort by priority (ascending - 1 is highest) and status (failing first)
        candidates.sort(
            key=lambda f: (0 if f.status == FeatureStatus.FAILING else 1, f.priority)
        )
        return candidates[0]

    def get_feature_by_id(self, feature_id: str) -> Optional[FeatureListItem]:
        """Get a feature by its ID."""
        for f in self.features:
            if f.id == feature_id:
                return f
        return None

    def get_progress_summary(self) -> dict[str, int]:
        """Get a summary of feature statuses."""
        summary: dict[str, int] = {status.value: 0 for status in FeatureStatus}
        for f in self.features:
            summary[f.status.value] += 1
        return summary

    def is_complete(self) -> bool:
        """Check if all features are passing."""
        return all(f.status == FeatureStatus.PASSING for f in self.features)


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


class ProgressEntry(BaseModel):
    """
    A single entry in the progress log.

    Each agent session should add entries as it works, and write a summary at the end.
    This helps the next agent understand what was done and why.
    """

    id: str = Field(description="Unique identifier for the entry")
    session_id: str = Field(description="The session that created this entry")
    entry_type: ProgressEntryType = Field(description="Type of progress entry")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When this entry was created"
    )
    title: str = Field(description="Short title for the entry")
    description: str = Field(default="", description="Detailed description")
    feature_id: Optional[str] = Field(
        default=None, description="Related feature ID, if applicable"
    )
    git_commit_hash: Optional[str] = Field(
        default=None, description="Related git commit, if applicable"
    )
    files_changed: list[str] = Field(
        default_factory=list, description="List of files that were changed"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class ProgressLog(BaseModel):
    """
    The complete progress log for a long-running agent project.

    This file is continuously updated by agent sessions to track progress.
    """

    project_name: str = Field(description="Name of the project")
    entries: list[ProgressEntry] = Field(
        default_factory=list, description="All progress entries"
    )

    def get_recent_entries(self, limit: int = 20) -> list[ProgressEntry]:
        """Get the most recent entries."""
        return sorted(self.entries, key=lambda e: e.timestamp, reverse=True)[:limit]

    def get_session_entries(self, session_id: str) -> list[ProgressEntry]:
        """Get all entries for a specific session."""
        return [e for e in self.entries if e.session_id == session_id]

    def get_feature_entries(self, feature_id: str) -> list[ProgressEntry]:
        """Get all entries related to a specific feature."""
        return [e for e in self.entries if e.feature_id == feature_id]


class SessionStatus(str, Enum):
    """Status of a long-running agent session."""

    INITIALIZING = "initializing"
    READY = "ready"
    WORKING = "working"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class LongRunningSessionState(BaseModel):
    """
    The overall state of a long-running agent project.

    This is the main state object that tracks the project across sessions.
    """

    id: str = Field(description="Unique identifier for the session state")
    project_name: str = Field(description="Name of the project")
    project_description: str = Field(description="Description of the project goal")
    status: SessionStatus = Field(
        default=SessionStatus.INITIALIZING, description="Current status"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the project was created"
    )
    last_updated: datetime = Field(
        default_factory=datetime.utcnow, description="When the state was last updated"
    )
    current_session_id: Optional[str] = Field(
        default=None, description="ID of the currently active session"
    )
    session_count: int = Field(default=0, description="Number of sessions run")
    working_directory: str = Field(description="The working directory for the project")
    init_script_path: Optional[str] = Field(
        default=None, description="Path to the init.sh script"
    )
    feature_list_path: Optional[str] = Field(
        default=None, description="Path to the feature list JSON file"
    )
    progress_log_path: Optional[str] = Field(
        default=None, description="Path to the progress log file"
    )
    git_repo_initialized: bool = Field(
        default=False, description="Whether git repo is initialized"
    )
    current_feature_id: Optional[str] = Field(
        default=None, description="The feature currently being worked on"
    )
    environment_variables: dict[str, str] = Field(
        default_factory=dict, description="Environment variables for the project"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional project metadata"
    )


class InitializerConfig(BaseModel):
    """
    Configuration for the initializer agent.

    The initializer sets up the project environment on the first run.
    """

    project_name: str = Field(description="Name of the project to create")
    project_description: str = Field(
        description="Description of what the project should accomplish"
    )
    working_directory: str = Field(
        description="Directory where the project will be created"
    )
    generate_feature_list: bool = Field(
        default=True, description="Whether to generate a feature list from the spec"
    )
    initialize_git: bool = Field(
        default=True, description="Whether to initialize a git repository"
    )
    create_init_script: bool = Field(
        default=True, description="Whether to create an init.sh script"
    )
    custom_init_commands: list[str] = Field(
        default_factory=list, description="Additional commands to run during init"
    )
    feature_categories: list[FeatureCategory] = Field(
        default_factory=lambda: [
            FeatureCategory.FUNCTIONAL,
            FeatureCategory.UI,
            FeatureCategory.INTEGRATION,
        ],
        description="Categories of features to generate",
    )


class CodingAgentConfig(BaseModel):
    """
    Configuration for the coding agent.

    The coding agent works on implementing features incrementally.
    """

    session_state_path: str = Field(
        description="Path to the session state file to load"
    )
    max_features_per_session: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Maximum features to work on in a single session",
    )
    auto_commit: bool = Field(
        default=True, description="Whether to auto-commit changes after each feature"
    )
    run_tests_before_marking_complete: bool = Field(
        default=True, description="Whether to run verification tests before completion"
    )
    use_browser_testing: bool = Field(
        default=False, description="Whether to use browser automation for E2E testing"
    )
    verify_basic_functionality_first: bool = Field(
        default=True,
        description="Whether to verify basic functionality before starting new work",
    )
