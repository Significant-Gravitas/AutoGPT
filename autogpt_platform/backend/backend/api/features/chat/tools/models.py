"""Pydantic models for tool responses."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.data.model import CredentialsMetaInput


class ResponseType(str, Enum):
    """Types of tool responses."""

    AGENTS_FOUND = "agents_found"
    AGENT_DETAILS = "agent_details"
    SETUP_REQUIREMENTS = "setup_requirements"
    EXECUTION_STARTED = "execution_started"
    NEED_LOGIN = "need_login"
    ERROR = "error"
    NO_RESULTS = "no_results"
    AGENT_OUTPUT = "agent_output"
    UNDERSTANDING_UPDATED = "understanding_updated"


# Base response model
class ToolResponseBase(BaseModel):
    """Base model for all tool responses."""

    type: ResponseType
    message: str
    session_id: str | None = None


# Agent discovery models
class AgentInfo(BaseModel):
    """Information about an agent."""

    id: str
    name: str
    description: str
    source: str = Field(description="marketplace or library")
    in_library: bool = False
    creator: str | None = None
    category: str | None = None
    rating: float | None = None
    runs: int | None = None
    is_featured: bool | None = None
    status: str | None = None
    can_access_graph: bool | None = None
    has_external_trigger: bool | None = None
    new_output: bool | None = None
    graph_id: str | None = None


class AgentsFoundResponse(ToolResponseBase):
    """Response for find_agent tool."""

    type: ResponseType = ResponseType.AGENTS_FOUND
    title: str = "Available Agents"
    agents: list[AgentInfo]
    count: int
    name: str = "agents_found"


class NoResultsResponse(ToolResponseBase):
    """Response when no agents found."""

    type: ResponseType = ResponseType.NO_RESULTS
    suggestions: list[str] = []
    name: str = "no_results"


# Agent details models
class InputField(BaseModel):
    """Input field specification."""

    name: str
    type: str = "string"
    description: str = ""
    required: bool = False
    default: Any | None = None
    options: list[Any] | None = None
    format: str | None = None


class ExecutionOptions(BaseModel):
    """Available execution options for an agent."""

    manual: bool = True
    scheduled: bool = True
    webhook: bool = False


class AgentDetails(BaseModel):
    """Detailed agent information."""

    id: str
    name: str
    description: str
    in_library: bool = False
    inputs: dict[str, Any] = {}
    credentials: list[CredentialsMetaInput] = []
    execution_options: ExecutionOptions = Field(default_factory=ExecutionOptions)
    trigger_info: dict[str, Any] | None = None


class AgentDetailsResponse(ToolResponseBase):
    """Response for get_details action."""

    type: ResponseType = ResponseType.AGENT_DETAILS
    agent: AgentDetails
    user_authenticated: bool = False
    graph_id: str | None = None
    graph_version: int | None = None


# Setup info models
class UserReadiness(BaseModel):
    """User readiness status."""

    has_all_credentials: bool = False
    missing_credentials: dict[str, Any] = {}
    ready_to_run: bool = False


class SetupInfo(BaseModel):
    """Complete setup information."""

    agent_id: str
    agent_name: str
    requirements: dict[str, list[Any]] = Field(
        default_factory=lambda: {
            "credentials": [],
            "inputs": [],
            "execution_modes": [],
        },
    )
    user_readiness: UserReadiness = Field(default_factory=UserReadiness)


class SetupRequirementsResponse(ToolResponseBase):
    """Response for validate action."""

    type: ResponseType = ResponseType.SETUP_REQUIREMENTS
    setup_info: SetupInfo
    graph_id: str | None = None
    graph_version: int | None = None


# Execution models
class ExecutionStartedResponse(ToolResponseBase):
    """Response for run/schedule actions."""

    type: ResponseType = ResponseType.EXECUTION_STARTED
    execution_id: str
    graph_id: str
    graph_name: str
    library_agent_id: str | None = None
    library_agent_link: str | None = None
    status: str = "QUEUED"


# Auth/error models
class NeedLoginResponse(ToolResponseBase):
    """Response when login is needed."""

    type: ResponseType = ResponseType.NEED_LOGIN
    agent_info: dict[str, Any] | None = None


class ErrorResponse(ToolResponseBase):
    """Response for errors."""

    type: ResponseType = ResponseType.ERROR
    error: str | None = None
    details: dict[str, Any] | None = None


# Agent output models
class ExecutionOutputInfo(BaseModel):
    """Summary of a single execution's outputs."""

    execution_id: str
    status: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    outputs: dict[str, list[Any]]
    inputs_summary: dict[str, Any] | None = None


class AgentOutputResponse(ToolResponseBase):
    """Response for agent_output tool."""

    type: ResponseType = ResponseType.AGENT_OUTPUT
    agent_name: str
    agent_id: str
    library_agent_id: str | None = None
    library_agent_link: str | None = None
    execution: ExecutionOutputInfo | None = None
    available_executions: list[dict[str, Any]] | None = None
    total_executions: int = 0


# Business understanding models
class UnderstandingUpdatedResponse(ToolResponseBase):
    """Response for add_understanding tool."""

    type: ResponseType = ResponseType.UNDERSTANDING_UPDATED
    updated_fields: list[str] = Field(default_factory=list)
    current_understanding: dict[str, Any] = Field(default_factory=dict)
