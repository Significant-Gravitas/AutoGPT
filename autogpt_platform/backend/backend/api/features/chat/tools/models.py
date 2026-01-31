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
    AGENT_PREVIEW = "agent_preview"
    AGENT_SAVED = "agent_saved"
    CLARIFICATION_NEEDED = "clarification_needed"
    BLOCK_LIST = "block_list"
    BLOCK_OUTPUT = "block_output"
    DOC_SEARCH_RESULTS = "doc_search_results"
    DOC_PAGE = "doc_page"
    # Long-running operation types
    OPERATION_STARTED = "operation_started"
    OPERATION_PENDING = "operation_pending"
    OPERATION_IN_PROGRESS = "operation_in_progress"
    # Input validation
    INPUT_VALIDATION_ERROR = "input_validation_error"


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
    inputs: dict[str, Any] | None = Field(
        default=None,
        description="Input schema for the agent, including field names, types, and defaults",
    )


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


class InputValidationErrorResponse(ToolResponseBase):
    """Response when run_agent receives unknown input fields."""

    type: ResponseType = ResponseType.INPUT_VALIDATION_ERROR
    unrecognized_fields: list[str] = Field(
        description="List of input field names that were not recognized"
    )
    inputs: dict[str, Any] = Field(
        description="The agent's valid input schema for reference"
    )
    graph_id: str | None = None
    graph_version: int | None = None


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


# Agent generation models
class ClarifyingQuestion(BaseModel):
    """A question that needs user clarification."""

    question: str
    keyword: str
    example: str | None = None


class AgentPreviewResponse(ToolResponseBase):
    """Response for previewing a generated agent before saving."""

    type: ResponseType = ResponseType.AGENT_PREVIEW
    agent_json: dict[str, Any]
    agent_name: str
    description: str
    node_count: int
    link_count: int = 0


class AgentSavedResponse(ToolResponseBase):
    """Response when an agent is saved to the library."""

    type: ResponseType = ResponseType.AGENT_SAVED
    agent_id: str
    agent_name: str
    library_agent_id: str
    library_agent_link: str
    agent_page_link: str  # Link to the agent builder/editor page


class ClarificationNeededResponse(ToolResponseBase):
    """Response when the LLM needs more information from the user."""

    type: ResponseType = ResponseType.CLARIFICATION_NEEDED
    questions: list[ClarifyingQuestion] = Field(default_factory=list)


# Documentation search models
class DocSearchResult(BaseModel):
    """A single documentation search result."""

    title: str
    path: str
    section: str
    snippet: str  # Short excerpt for UI display
    score: float
    doc_url: str | None = None


class DocSearchResultsResponse(ToolResponseBase):
    """Response for search_docs tool."""

    type: ResponseType = ResponseType.DOC_SEARCH_RESULTS
    results: list[DocSearchResult]
    count: int
    query: str


class DocPageResponse(ToolResponseBase):
    """Response for get_doc_page tool."""

    type: ResponseType = ResponseType.DOC_PAGE
    title: str
    path: str
    content: str  # Full document content
    doc_url: str | None = None


# Block models
class BlockInputFieldInfo(BaseModel):
    """Information about a block input field."""

    name: str
    type: str
    description: str = ""
    required: bool = False
    default: Any | None = None


class BlockInfoSummary(BaseModel):
    """Summary of a block for search results."""

    id: str
    name: str
    description: str
    categories: list[str]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    required_inputs: list[BlockInputFieldInfo] = Field(
        default_factory=list,
        description="List of required input fields for this block",
    )


class BlockListResponse(ToolResponseBase):
    """Response for find_block tool."""

    type: ResponseType = ResponseType.BLOCK_LIST
    blocks: list[BlockInfoSummary]
    count: int
    query: str
    usage_hint: str = Field(
        default="To execute a block, call run_block with block_id set to the block's "
        "'id' field and input_data containing the required fields from input_schema."
    )


class BlockOutputResponse(ToolResponseBase):
    """Response for run_block tool."""

    type: ResponseType = ResponseType.BLOCK_OUTPUT
    block_id: str
    block_name: str
    outputs: dict[str, list[Any]]
    success: bool = True


# Long-running operation models
class OperationStartedResponse(ToolResponseBase):
    """Response when a long-running operation has been started in the background.

    This is returned immediately to the client while the operation continues
    to execute. The user can close the tab and check back later.
    """

    type: ResponseType = ResponseType.OPERATION_STARTED
    operation_id: str
    tool_name: str


class OperationPendingResponse(ToolResponseBase):
    """Response stored in chat history while a long-running operation is executing.

    This is persisted to the database so users see a pending state when they
    refresh before the operation completes.
    """

    type: ResponseType = ResponseType.OPERATION_PENDING
    operation_id: str
    tool_name: str


class OperationInProgressResponse(ToolResponseBase):
    """Response when an operation is already in progress.

    Returned for idempotency when the same tool_call_id is requested again
    while the background task is still running.
    """

    type: ResponseType = ResponseType.OPERATION_IN_PROGRESS
    tool_call_id: str
