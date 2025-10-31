"""Pydantic models for tool responses."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from backend.data.model import CredentialsMetaInput


class ResponseType(str, Enum):
    """Types of tool responses."""

    AGENT_CAROUSEL = "agent_carousel"
    AGENT_DETAILS = "agent_details"
    AGENT_DETAILS_NEED_LOGIN = "agent_details_need_login"
    AGENT_DETAILS_NEED_CREDENTIALS = "agent_details_need_credentials"
    SETUP_REQUIREMENTS = "setup_requirements"
    SCHEDULE_CREATED = "schedule_created"
    WEBHOOK_CREATED = "webhook_created"
    PRESET_CREATED = "preset_created"
    EXECUTION_STARTED = "execution_started"
    NEED_LOGIN = "need_login"
    NEED_CREDENTIALS = "need_credentials"
    INSUFFICIENT_CREDITS = "insufficient_credits"
    VALIDATION_ERROR = "validation_error"
    ERROR = "error"
    NO_RESULTS = "no_results"
    SUCCESS = "success"


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


class AgentCarouselResponse(ToolResponseBase):
    """Response for find_agent tool."""

    type: ResponseType = ResponseType.AGENT_CAROUSEL
    title: str = "Available Agents"
    agents: list[AgentInfo]
    count: int


class NoResultsResponse(ToolResponseBase):
    """Response when no agents found."""

    type: ResponseType = ResponseType.NO_RESULTS
    suggestions: list[str] = []


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
    """Response for get_agent_details tool."""

    type: ResponseType = ResponseType.AGENT_DETAILS
    agent: AgentDetails
    user_authenticated: bool = False
    graph_id: str | None = None
    graph_version: int | None = None


class AgentDetailsNeedLoginResponse(ToolResponseBase):
    """Response when agent details need login."""

    type: ResponseType = ResponseType.AGENT_DETAILS_NEED_LOGIN
    agent: AgentDetails
    agent_info: dict[str, Any] | None = None
    graph_id: str | None = None
    graph_version: int | None = None


class AgentDetailsNeedCredentialsResponse(ToolResponseBase):
    """Response when agent needs credentials to be configured."""

    type: ResponseType = ResponseType.NEED_CREDENTIALS
    agent: AgentDetails
    credentials_schema: dict[str, Any]
    agent_info: dict[str, Any] | None = None
    graph_id: str | None = None
    graph_version: int | None = None


# Setup info models
class SetupRequirementInfo(BaseModel):
    """Setup requirement information."""

    key: str
    provider: str
    required: bool = True
    user_has: bool = False
    credential_id: str | None = None
    type: str | None = None
    scopes: list[str] | None = None
    description: str | None = None


class ExecutionModeInfo(BaseModel):
    """Execution mode information."""

    type: str  # manual, scheduled, webhook
    description: str
    supported: bool
    config_required: dict[str, str] | None = None
    trigger_info: dict[str, Any] | None = None


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
    setup_instructions: list[str] = []


class SetupRequirementsResponse(ToolResponseBase):
    """Response for get_required_setup_info tool."""

    type: ResponseType = ResponseType.SETUP_REQUIREMENTS
    setup_info: SetupInfo
    graph_id: str | None = None
    graph_version: int | None = None


# Setup agent models
class ScheduleCreatedResponse(ToolResponseBase):
    """Response for scheduled agent setup."""

    type: ResponseType = ResponseType.SCHEDULE_CREATED
    schedule_id: str
    name: str
    cron: str
    timezone: str = "UTC"
    next_run: str | None = None
    graph_id: str
    graph_name: str


class WebhookCreatedResponse(ToolResponseBase):
    """Response for webhook agent setup."""

    type: ResponseType = ResponseType.WEBHOOK_CREATED
    webhook_id: str
    webhook_url: str
    preset_id: str | None = None
    name: str
    graph_id: str
    graph_name: str


class PresetCreatedResponse(ToolResponseBase):
    """Response for preset agent setup."""

    type: ResponseType = ResponseType.PRESET_CREATED
    preset_id: str
    name: str
    graph_id: str
    graph_name: str


# Run agent models
class ExecutionStartedResponse(ToolResponseBase):
    """Response for agent execution started."""

    type: ResponseType = ResponseType.EXECUTION_STARTED
    execution_id: str
    graph_id: str
    graph_name: str
    status: str = "QUEUED"
    ended_at: str | None = None
    outputs: dict[str, Any] | None = None
    error: str | None = None
    timeout_reached: bool | None = None


class InsufficientCreditsResponse(ToolResponseBase):
    """Response for insufficient credits."""

    type: ResponseType = ResponseType.INSUFFICIENT_CREDITS
    balance: float


class ValidationErrorResponse(ToolResponseBase):
    """Response for validation errors."""

    type: ResponseType = ResponseType.VALIDATION_ERROR
    error: str
    details: dict[str, Any] | None = None


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
