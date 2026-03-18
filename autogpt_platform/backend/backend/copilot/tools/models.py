"""Pydantic models for tool responses."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.data.model import CredentialsMetaInput


class ResponseType(str, Enum):
    """Types of tool responses."""

    # General
    ERROR = "error"
    NO_RESULTS = "no_results"
    NEED_LOGIN = "need_login"

    # Agent discovery & execution
    AGENTS_FOUND = "agents_found"
    AGENT_DETAILS = "agent_details"
    SETUP_REQUIREMENTS = "setup_requirements"
    INPUT_VALIDATION_ERROR = "input_validation_error"
    EXECUTION_STARTED = "execution_started"
    AGENT_OUTPUT = "agent_output"
    UNDERSTANDING_UPDATED = "understanding_updated"
    SUGGESTED_GOAL = "suggested_goal"

    # Agent builder (create / edit / validate / fix)
    AGENT_BUILDER_GUIDE = "agent_builder_guide"
    AGENT_BUILDER_PREVIEW = "agent_builder_preview"
    AGENT_BUILDER_SAVED = "agent_builder_saved"
    AGENT_BUILDER_CLARIFICATION_NEEDED = "agent_builder_clarification_needed"
    AGENT_BUILDER_VALIDATION_RESULT = "agent_builder_validation_result"
    AGENT_BUILDER_FIX_RESULT = "agent_builder_fix_result"

    # Block
    BLOCK_LIST = "block_list"
    BLOCK_DETAILS = "block_details"
    BLOCK_OUTPUT = "block_output"
    REVIEW_REQUIRED = "review_required"
    BLOCK_JOB_STARTED = "block_job_started"
    BLOCK_JOB_RESULT = "block_job_result"

    # MCP
    MCP_GUIDE = "mcp_guide"
    MCP_TOOLS_DISCOVERED = "mcp_tools_discovered"
    MCP_TOOL_OUTPUT = "mcp_tool_output"

    # Docs
    DOC_SEARCH_RESULTS = "doc_search_results"
    DOC_PAGE = "doc_page"

    # Workspace files
    WORKSPACE_FILE_LIST = "workspace_file_list"
    WORKSPACE_FILE_CONTENT = "workspace_file_content"
    WORKSPACE_FILE_METADATA = "workspace_file_metadata"
    WORKSPACE_FILE_WRITTEN = "workspace_file_written"
    WORKSPACE_FILE_DELETED = "workspace_file_deleted"

    # Folder management
    FOLDER_CREATED = "folder_created"
    FOLDER_LIST = "folder_list"
    FOLDER_UPDATED = "folder_updated"
    FOLDER_MOVED = "folder_moved"
    FOLDER_DELETED = "folder_deleted"
    AGENTS_MOVED_TO_FOLDER = "agents_moved_to_folder"

    # Browser automation
    BROWSER_NAVIGATE = "browser_navigate"
    BROWSER_ACT = "browser_act"
    BROWSER_SCREENSHOT = "browser_screenshot"

    # Code execution
    BASH_EXEC = "bash_exec"

    # Web
    WEB_FETCH = "web_fetch"

    # Feature requests
    FEATURE_REQUEST_SEARCH = "feature_request_search"
    FEATURE_REQUEST_CREATED = "feature_request_created"


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
    graph_version: int | None = None
    input_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema for the agent's inputs (for AgentExecutorBlock)",
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema for the agent's outputs (for AgentExecutorBlock)",
    )
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

    type: ResponseType = ResponseType.AGENT_BUILDER_PREVIEW
    agent_json: dict[str, Any]
    agent_name: str
    description: str
    node_count: int
    link_count: int = 0


class AgentSavedResponse(ToolResponseBase):
    """Response when an agent is saved to the library."""

    type: ResponseType = ResponseType.AGENT_BUILDER_SAVED
    agent_id: str
    agent_name: str
    library_agent_id: str
    library_agent_link: str
    agent_page_link: str  # Link to the agent builder/editor page


class ClarificationNeededResponse(ToolResponseBase):
    """Response when the LLM needs more information from the user."""

    type: ResponseType = ResponseType.AGENT_BUILDER_CLARIFICATION_NEEDED
    questions: list[ClarifyingQuestion] = Field(default_factory=list)


class SuggestedGoalResponse(ToolResponseBase):
    """Response when the goal needs refinement with a suggested alternative."""

    type: ResponseType = ResponseType.SUGGESTED_GOAL
    suggested_goal: str = Field(description="The suggested alternative goal")
    reason: str = Field(
        default="", description="Why the original goal needs refinement"
    )
    original_goal: str = Field(
        default="", description="The user's original goal for context"
    )
    goal_type: Literal["vague", "unachievable"] = Field(
        default="vague", description="Type: 'vague' or 'unachievable'"
    )


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
    input_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="Full JSON schema for block inputs",
    )
    output_schema: dict[str, Any] = Field(
        default_factory=dict,
        description="Full JSON schema for block outputs",
    )
    static_output: bool = Field(
        default=False,
        description="Whether the block produces output without needing input",
    )
    required_inputs: list[BlockInputFieldInfo] = Field(
        default_factory=list,
        description="List of input fields for this block",
    )


class BlockListResponse(ToolResponseBase):
    """Response for find_block tool."""

    type: ResponseType = ResponseType.BLOCK_LIST
    blocks: list[BlockInfoSummary]
    count: int
    query: str
    usage_hint: str = Field(
        default="To execute a block, call run_block with block_id set to the block's "
        "'id' field and input_data containing the fields listed in required_inputs."
    )


class BlockDetails(BaseModel):
    """Detailed block information."""

    id: str
    name: str
    description: str
    inputs: dict[str, Any] = {}
    outputs: dict[str, Any] = {}
    credentials: list[CredentialsMetaInput] = []


class BlockDetailsResponse(ToolResponseBase):
    """Response for block details (first run_block attempt)."""

    type: ResponseType = ResponseType.BLOCK_DETAILS
    block: BlockDetails
    user_authenticated: bool = False


class BlockOutputResponse(ToolResponseBase):
    """Response for run_block tool."""

    type: ResponseType = ResponseType.BLOCK_OUTPUT
    block_id: str
    block_name: str
    outputs: dict[str, list[Any]]
    success: bool = True


class ReviewRequiredResponse(ToolResponseBase):
    """Response when a block requires human review before execution."""

    type: ResponseType = ResponseType.REVIEW_REQUIRED
    block_id: str
    block_name: str
    review_id: str = Field(description="The review ID for tracking approval status")
    graph_exec_id: str = Field(
        description="The graph execution ID for fetching review status"
    )
    input_data: dict[str, Any] = Field(
        description="The input data that requires review"
    )


class BlockJobStartedResponse(ToolResponseBase):
    """Response for run_block_async tool — returned immediately before execution."""

    type: ResponseType = ResponseType.BLOCK_JOB_STARTED
    job_id: str
    block_id: str
    block_name: str


class BlockJobResultResponse(ToolResponseBase):
    """Response for get_block_result tool."""

    type: ResponseType = ResponseType.BLOCK_JOB_RESULT
    job_id: str
    block_id: str
    block_name: str
    outputs: dict[str, list[Any]] | None = None
    success: bool = True
    error: str | None = None


class WebFetchResponse(ToolResponseBase):
    """Response for web_fetch tool."""

    type: ResponseType = ResponseType.WEB_FETCH
    url: str
    status_code: int
    content_type: str
    content: str
    truncated: bool = False


class BashExecResponse(ToolResponseBase):
    """Response for bash_exec tool."""

    type: ResponseType = ResponseType.BASH_EXEC
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


# Feature request models
class FeatureRequestInfo(BaseModel):
    """Information about a feature request issue."""

    id: str
    identifier: str
    title: str


class FeatureRequestSearchResponse(ToolResponseBase):
    """Response for search_feature_requests tool."""

    type: ResponseType = ResponseType.FEATURE_REQUEST_SEARCH
    results: list[FeatureRequestInfo]
    count: int
    query: str


class FeatureRequestCreatedResponse(ToolResponseBase):
    """Response for create_feature_request tool."""

    type: ResponseType = ResponseType.FEATURE_REQUEST_CREATED
    issue_id: str
    issue_identifier: str
    issue_title: str
    issue_url: str
    is_new_issue: bool  # False if added to existing
    customer_name: str


# MCP tool models
class MCPToolInfo(BaseModel):
    """Information about a single MCP tool discovered from a server."""

    name: str
    description: str
    input_schema: dict[str, Any]


class MCPToolsDiscoveredResponse(ToolResponseBase):
    """Response when MCP tools are discovered from a server (agent-internal)."""

    type: ResponseType = ResponseType.MCP_TOOLS_DISCOVERED
    server_url: str
    tools: list[MCPToolInfo]


class MCPToolOutputResponse(ToolResponseBase):
    """Response after executing an MCP tool."""

    type: ResponseType = ResponseType.MCP_TOOL_OUTPUT
    server_url: str
    tool_name: str
    result: Any = None
    success: bool = True


# Agent-browser multi-step automation models


class BrowserNavigateResponse(ToolResponseBase):
    """Response for browser_navigate tool."""

    type: ResponseType = ResponseType.BROWSER_NAVIGATE
    url: str
    title: str
    snapshot: str  # Interactive accessibility tree with @ref IDs


class BrowserActResponse(ToolResponseBase):
    """Response for browser_act tool."""

    type: ResponseType = ResponseType.BROWSER_ACT
    action: str
    current_url: str = ""
    snapshot: str  # Updated accessibility tree after the action


class BrowserScreenshotResponse(ToolResponseBase):
    """Response for browser_screenshot tool."""

    type: ResponseType = ResponseType.BROWSER_SCREENSHOT
    file_id: str  # Workspace file ID — use read_workspace_file to retrieve
    filename: str


# Agent generation tool response models


class ValidationResultResponse(ToolResponseBase):
    """Response for validate_agent_graph tool."""

    type: ResponseType = ResponseType.AGENT_BUILDER_VALIDATION_RESULT
    valid: bool
    errors: list[str] = Field(default_factory=list)
    error_count: int = 0


class FixResultResponse(ToolResponseBase):
    """Response for fix_agent_graph tool."""

    type: ResponseType = ResponseType.AGENT_BUILDER_FIX_RESULT
    fixed_agent_json: dict[str, Any]
    fixes_applied: list[str] = Field(default_factory=list)
    fix_count: int = 0
    valid_after_fix: bool = False
    remaining_errors: list[str] = Field(default_factory=list)


# Folder management models


class FolderAgentSummary(BaseModel):
    """Lightweight agent info for folder listings."""

    id: str
    name: str
    description: str = ""


class FolderInfo(BaseModel):
    """Information about a folder."""

    id: str
    name: str
    parent_id: str | None = None
    icon: str | None = None
    color: str | None = None
    agent_count: int = 0
    subfolder_count: int = 0
    agents: list[FolderAgentSummary] | None = None


class FolderTreeInfo(FolderInfo):
    """Folder with nested children for tree display."""

    children: list["FolderTreeInfo"] = []


class FolderCreatedResponse(ToolResponseBase):
    """Response when a folder is created."""

    type: ResponseType = ResponseType.FOLDER_CREATED
    folder: FolderInfo


class FolderListResponse(ToolResponseBase):
    """Response for listing folders."""

    type: ResponseType = ResponseType.FOLDER_LIST
    folders: list[FolderInfo] = Field(default_factory=list)
    tree: list[FolderTreeInfo] | None = None
    root_agents: list[FolderAgentSummary] | None = None
    count: int = 0


class FolderUpdatedResponse(ToolResponseBase):
    """Response when a folder is updated."""

    type: ResponseType = ResponseType.FOLDER_UPDATED
    folder: FolderInfo


class FolderMovedResponse(ToolResponseBase):
    """Response when a folder is moved."""

    type: ResponseType = ResponseType.FOLDER_MOVED
    folder: FolderInfo
    target_parent_id: str | None = None


class FolderDeletedResponse(ToolResponseBase):
    """Response when a folder is deleted."""

    type: ResponseType = ResponseType.FOLDER_DELETED
    folder_id: str


class AgentsMovedToFolderResponse(ToolResponseBase):
    """Response when agents are moved to a folder."""

    type: ResponseType = ResponseType.AGENTS_MOVED_TO_FOLDER
    agent_ids: list[str]
    agent_names: list[str] = []
    folder_id: str | None = None
    count: int = 0
