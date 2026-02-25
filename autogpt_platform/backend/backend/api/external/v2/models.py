"""
V2 External API - Request and Response Models

This module defines all request and response models for the v2 external API.
All models are self-contained and specific to the external API contract.

Route files should import models from here rather than defining them locally.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, JsonValue

from backend.blocks._base import BlockCostType

# ============================================================================
# Common/Shared Models
# ============================================================================


class PaginatedResponse(BaseModel):
    """Base class for paginated responses."""

    total_count: int = Field(description="Total number of items across all pages")
    page: int = Field(description="Current page number (1-indexed)")
    page_size: int = Field(description="Number of items per page")
    total_pages: int = Field(description="Total number of pages")


# ============================================================================
# Graph Models
# ============================================================================


class GraphLink(BaseModel):
    """A link between two nodes in a graph."""

    id: str
    source_id: str = Field(description="ID of the source node")
    sink_id: str = Field(description="ID of the target node")
    source_name: str = Field(description="Output pin name on source node")
    sink_name: str = Field(description="Input pin name on target node")
    is_static: bool = Field(
        default=False, description="Whether this link provides static data"
    )


class GraphNode(BaseModel):
    """A node in an agent graph."""

    id: str
    block_id: str = Field(description="ID of the block type")
    input_default: dict[str, Any] = Field(
        default_factory=dict, description="Default input values"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Node metadata (e.g., position)"
    )


class Graph(BaseModel):
    """Graph definition for creating or updating an agent."""

    id: Optional[str] = Field(default=None, description="Graph ID (assigned by server)")
    version: int = Field(default=1, description="Graph version")
    is_active: bool = Field(default=True, description="Whether this version is active")
    name: str = Field(description="Graph name")
    description: str = Field(default="", description="Graph description")
    nodes: list[GraphNode] = Field(default_factory=list, description="List of nodes")
    links: list[GraphLink] = Field(
        default_factory=list, description="Links between nodes"
    )


class GraphMeta(BaseModel):
    """Graph metadata (summary information)."""

    id: str
    version: int
    is_active: bool
    name: str
    description: str
    created_at: datetime


class GraphDetails(GraphMeta):
    """Full graph details including nodes and links."""

    nodes: list[GraphNode]
    links: list[GraphLink]
    input_schema: dict[str, Any] = Field(description="Input schema for the graph")
    output_schema: dict[str, Any] = Field(description="Output schema for the graph")
    credentials_input_schema: dict[str, Any] = Field(
        description="Schema for required credentials"
    )


class GraphSettings(BaseModel):
    """Settings for a graph."""

    human_in_the_loop_safe_mode: Optional[bool] = Field(
        default=None, description="Enable safe mode for human-in-the-loop blocks"
    )


class CreateGraphRequest(BaseModel):
    """Request to create a new graph."""

    graph: Graph = Field(description="The graph definition")


class SetActiveVersionRequest(BaseModel):
    """Request to set the active graph version."""

    active_graph_version: int = Field(description="Version number to set as active")


class GraphsListResponse(PaginatedResponse):
    """Response for listing graphs."""

    graphs: list[GraphMeta]


class DeleteGraphResponse(BaseModel):
    """Response for deleting a graph."""

    version_count: int = Field(description="Number of versions deleted")


# ============================================================================
# Block Models
# ============================================================================


class BlockCost(BaseModel):
    """Cost information for a block."""

    cost_type: BlockCostType = Field(
        description="Type of cost (e.g., 'per_call', 'per_token')"
    )
    cost_filter: dict[str, Any] = Field(
        default_factory=dict, description="Conditions for this cost"
    )
    cost_amount: int = Field(description="Cost amount in credits")


class Block(BaseModel):
    """A building block that can be used in graphs."""

    id: str
    name: str
    description: str
    categories: list[str] = Field(default_factory=list)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    costs: list[BlockCost] = Field(default_factory=list)
    disabled: bool = Field(default=False)


class BlocksListResponse(BaseModel):
    """Response for listing blocks."""

    blocks: list[Block]
    total_count: int


# ============================================================================
# Schedule Models
# ============================================================================


class Schedule(BaseModel):
    """An execution schedule for a graph."""

    id: str
    name: str
    graph_id: str
    graph_version: int
    cron: str = Field(description="Cron expression for the schedule")
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Input data for scheduled executions"
    )
    next_run_time: Optional[datetime] = Field(
        default=None, description="Next scheduled run time"
    )
    is_enabled: bool = Field(default=True, description="Whether schedule is enabled")


class CreateScheduleRequest(BaseModel):
    """Request to create a schedule."""

    name: str = Field(description="Display name for the schedule")
    cron: str = Field(description="Cron expression (e.g., '0 9 * * *' for 9am daily)")
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Input data for scheduled executions"
    )
    credentials_inputs: dict[str, Any] = Field(
        default_factory=dict, description="Credentials for the schedule"
    )
    graph_version: Optional[int] = Field(
        default=None, description="Graph version (default: active version)"
    )
    timezone: Optional[str] = Field(
        default=None,
        description=(
            "Timezone for schedule (e.g., 'America/New_York'). "
            "Defaults to user's timezone."
        ),
    )


class SchedulesListResponse(PaginatedResponse):
    """Response for listing schedules."""

    schedules: list[Schedule]


# ============================================================================
# Library Models
# ============================================================================


class LibraryAgent(BaseModel):
    """An agent in the user's library."""

    id: str
    graph_id: str
    graph_version: int
    name: str
    description: str
    is_favorite: bool = False
    can_access_graph: bool = False
    is_latest_version: bool = False
    image_url: Optional[str] = None
    creator_name: str
    input_schema: dict[str, Any] = Field(description="Input schema for the agent")
    output_schema: dict[str, Any] = Field(description="Output schema for the agent")
    created_at: datetime
    updated_at: datetime


class LibraryAgentsResponse(PaginatedResponse):
    """Response for listing library agents."""

    agents: list[LibraryAgent]


class ExecuteAgentRequest(BaseModel):
    """Request to execute an agent."""

    inputs: dict[str, Any] = Field(
        default_factory=dict, description="Input values for the agent"
    )
    credentials_inputs: dict[str, Any] = Field(
        default_factory=dict, description="Credentials for the agent"
    )


# ============================================================================
# Run Models
# ============================================================================


class Run(BaseModel):
    """An execution run."""

    id: str
    graph_id: str
    graph_version: int
    status: str = Field(
        description="One of: INCOMPLETE, QUEUED, RUNNING, COMPLETED, TERMINATED, FAILED, REVIEW"
    )
    started_at: datetime | None
    ended_at: Optional[datetime] = None
    inputs: Optional[dict[str, Any]] = None
    cost: int = Field(default=0, description="Cost in credits")
    duration: float = Field(default=0, description="Duration in seconds")
    node_count: int = Field(default=0, description="Number of nodes executed")


class RunDetails(Run):
    """Detailed information about a run including outputs and node executions."""

    outputs: Optional[dict[str, list[Any]]] = None
    node_executions: list[dict[str, Any]] = Field(
        default_factory=list, description="Individual node execution results"
    )


class RunsListResponse(PaginatedResponse):
    """Response for listing runs."""

    runs: list[Run]


# ============================================================================
# Run Review Models (Human-in-the-loop)
# ============================================================================


class PendingReview(BaseModel):
    """A pending human-in-the-loop review."""

    id: str  # node_exec_id
    run_id: str
    graph_id: str
    graph_version: int
    payload: JsonValue = Field(description="Data to be reviewed")
    instructions: Optional[str] = Field(
        default=None, description="Instructions for the reviewer"
    )
    editable: bool = Field(
        default=True, description="Whether the reviewer can edit the data"
    )
    status: str = Field(description="One of: WAITING, APPROVED, REJECTED")
    created_at: datetime


class PendingReviewsResponse(PaginatedResponse):
    """Response for listing pending reviews."""

    reviews: list[PendingReview]


class ReviewDecision(BaseModel):
    """Decision for a single review item."""

    node_exec_id: str = Field(description="Node execution ID (review ID)")
    approved: bool = Field(description="Whether to approve the data")
    edited_payload: Optional[JsonValue] = Field(
        default=None, description="Modified payload data (if editing)"
    )
    message: Optional[str] = Field(
        default=None, description="Optional message from reviewer", max_length=2000
    )


class SubmitReviewsRequest(BaseModel):
    """Request to submit review responses for all pending reviews of an execution."""

    reviews: list[ReviewDecision] = Field(
        description="All review decisions for the execution"
    )


class SubmitReviewsResponse(BaseModel):
    """Response after submitting reviews."""

    run_id: str
    approved_count: int = Field(description="Number of reviews approved")
    rejected_count: int = Field(description="Number of reviews rejected")


# ============================================================================
# Credit Models
# ============================================================================


class CreditBalance(BaseModel):
    """User's credit balance."""

    balance: int = Field(description="Current credit balance")


class CreditTransaction(BaseModel):
    """A credit transaction."""

    transaction_key: str
    amount: int = Field(description="Transaction amount (positive or negative)")
    type: str = Field(description="One of: TOP_UP, USAGE, GRANT, REFUND")
    transaction_time: datetime
    running_balance: Optional[int] = Field(
        default=None, description="Balance after this transaction"
    )
    description: Optional[str] = None


class CreditTransactionsResponse(PaginatedResponse):
    """Response for listing credit transactions."""

    transactions: list[CreditTransaction]


# ============================================================================
# Integration Models
# ============================================================================


class Credential(BaseModel):
    """A user's credential for an integration."""

    id: str
    provider: str = Field(description="Integration provider name")
    title: Optional[str] = Field(
        default=None, description="User-assigned title for this credential"
    )
    scopes: list[str] = Field(default_factory=list, description="Granted scopes")


class CredentialsListResponse(BaseModel):
    """Response for listing credentials."""

    credentials: list[Credential]


class CredentialRequirement(BaseModel):
    """A credential requirement for a graph or agent."""

    provider: str = Field(description="Required provider name")
    required_scopes: list[str] = Field(
        default_factory=list, description="Required scopes"
    )
    matching_credentials: list[Credential] = Field(
        default_factory=list,
        description="User's credentials that match this requirement",
    )


class CredentialRequirementsResponse(BaseModel):
    """Response for listing credential requirements."""

    requirements: list[CredentialRequirement]


# ============================================================================
# File Models
# ============================================================================


class UploadFileResponse(BaseModel):
    """Response after uploading a file."""

    file_uri: str = Field(description="URI to reference the uploaded file in agents")
    file_name: str
    size: int = Field(description="File size in bytes")
    content_type: str
    expires_in_hours: int


# ============================================================================
# Marketplace Models
# ============================================================================


class MarketplaceAgent(BaseModel):
    """An agent available in the marketplace."""

    slug: str
    name: str
    description: str
    sub_heading: str
    creator: str
    creator_avatar: str
    runs: int = Field(default=0, description="Number of times this agent has been run")
    rating: float = Field(default=0.0, description="Average rating")
    image_url: str = Field(default="")


class MarketplaceAgentDetails(BaseModel):
    """Detailed information about a marketplace agent."""

    store_listing_version_id: str
    slug: str
    name: str
    description: str
    sub_heading: str
    instructions: Optional[str] = None
    creator: str
    creator_avatar: str
    categories: list[str] = Field(default_factory=list)
    runs: int = Field(default=0)
    rating: float = Field(default=0.0)
    image_urls: list[str] = Field(default_factory=list)
    video_url: str = Field(default="")
    versions: list[str] = Field(default_factory=list, description="Available versions")
    agent_graph_versions: list[str] = Field(default_factory=list)
    agent_graph_id: str
    last_updated: datetime


class MarketplaceAgentsResponse(PaginatedResponse):
    """Response for listing marketplace agents."""

    agents: list[MarketplaceAgent]


class MarketplaceCreator(BaseModel):
    """A creator on the marketplace."""

    name: str
    username: str
    description: str
    avatar_url: str
    num_agents: int
    agent_rating: float
    agent_runs: int
    is_featured: bool = False


class MarketplaceCreatorDetails(BaseModel):
    """Detailed information about a marketplace creator."""

    name: str
    username: str
    description: str
    avatar_url: str
    agent_rating: float
    agent_runs: int
    top_categories: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)


class MarketplaceCreatorsResponse(PaginatedResponse):
    """Response for listing marketplace creators."""

    creators: list[MarketplaceCreator]


class MarketplaceSubmission(BaseModel):
    """A marketplace submission."""

    graph_id: str
    graph_version: int
    name: str
    sub_heading: str
    slug: str
    description: str
    instructions: Optional[str] = None
    image_urls: list[str] = Field(default_factory=list)
    date_submitted: datetime
    status: str = Field(description="One of: DRAFT, PENDING, APPROVED, REJECTED")
    runs: int = Field(default=0)
    rating: float = Field(default=0.0)
    store_listing_version_id: Optional[str] = None
    version: Optional[int] = None
    review_comments: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    video_url: Optional[str] = None
    categories: list[str] = Field(default_factory=list)


class CreateSubmissionRequest(BaseModel):
    """Request to create a marketplace submission."""

    graph_id: str = Field(description="ID of the graph to submit")
    graph_version: int = Field(description="Version of the graph to submit")
    name: str = Field(description="Display name for the agent")
    slug: str = Field(description="URL-friendly identifier")
    description: str = Field(description="Full description")
    sub_heading: str = Field(description="Short tagline")
    image_urls: list[str] = Field(default_factory=list)
    video_url: Optional[str] = None
    categories: list[str] = Field(default_factory=list)


class SubmissionsListResponse(PaginatedResponse):
    """Response for listing submissions."""

    submissions: list[MarketplaceSubmission]
