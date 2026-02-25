"""
V2 External API - Request and Response Models

This module defines all request and response models for the v2 external API.
All models are self-contained and specific to the external API contract.

Route files should import models from here rather than defining them locally.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional, Self, TypeAlias

from pydantic import BaseModel, Field, JsonValue

import backend.blocks._base as block_types

if TYPE_CHECKING:
    from backend.api.features.executions.review.model import PendingHumanReviewModel
    from backend.api.features.library.model import LibraryAgent as _LibraryAgent
    from backend.api.features.store.model import (
        Creator,
        CreatorDetails,
        StoreAgent,
        StoreAgentDetails,
        StoreSubmission,
    )
    from backend.data.execution import GraphExecutionMeta, GraphExecutionWithNodes
    from backend.data.graph import Graph as _Graph
    from backend.data.graph import GraphMeta as _GraphMeta
    from backend.data.graph import GraphModel as _GraphModel
    from backend.data.graph import GraphSettings as _GraphSettings
    from backend.data.model import Credentials, UserTransaction
    from backend.executor.scheduler import GraphExecutionJobInfo

# ============================================================================
# Common/Shared Models
# ============================================================================


class PaginatedResponse(BaseModel):
    """Base class for paginated responses."""

    page: int = Field(description="Current page number (1-indexed)")
    page_size: int = Field(description="Number of items per page")
    total_count: int = Field(description="Total number of items across all pages")
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


class CreatableGraph(BaseModel):
    """Graph model for creating or updating an agent graph."""

    name: str = Field(description="Graph name")
    description: str = Field(default="", description="Graph description")
    nodes: list[GraphNode] = Field(description="List of nodes")
    links: list[GraphLink] = Field(description="Links between nodes")
    is_active: bool = Field(default=True, description="Whether this version is active")

    def to_internal(
        self,
        *,
        id: str,
        version: int,
    ) -> "_Graph":
        from backend.data.graph import Graph as _Graph
        from backend.data.graph import Link as _Link
        from backend.data.graph import Node as _Node

        return _Graph(
            id=id,  # "" triggers UUID generation
            version=version,
            is_active=self.is_active,
            name=self.name,
            description=self.description,
            nodes=[
                _Node(
                    id=node.id,
                    block_id=node.block_id,
                    input_default=node.input_default,
                    metadata=node.metadata,
                )
                for node in self.nodes
            ],
            links=[
                _Link(
                    id=link.id,
                    source_id=link.source_id,
                    sink_id=link.sink_id,
                    source_name=link.source_name,
                    sink_name=link.sink_name,
                    is_static=link.is_static,
                )
                for link in self.links
            ],
        )


class GraphMeta(BaseModel):
    """Graph metadata (summary information)."""

    id: str
    version: int
    is_active: bool
    name: str
    description: str
    created_at: datetime

    @classmethod
    def from_internal(cls, graph: _GraphMeta) -> Self:
        return cls(
            id=graph.id,
            version=graph.version,
            is_active=graph.is_active,
            name=graph.name,
            description=graph.description,
            created_at=graph.created_at,
        )


class Graph(GraphMeta):
    """Full graph details including nodes and links."""

    nodes: list[GraphNode]
    links: list[GraphLink]
    input_schema: dict[str, Any] = Field(description="Input schema for the graph")
    output_schema: dict[str, Any] = Field(description="Output schema for the graph")
    credentials_input_schema: dict[str, Any] = Field(
        description="Schema for required credentials"
    )

    @classmethod
    def from_internal(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, graph: _GraphModel
    ) -> Self:
        return cls(
            id=graph.id,
            version=graph.version,
            is_active=graph.is_active,
            name=graph.name,
            description=graph.description,
            created_at=graph.created_at,
            input_schema=graph.input_schema,
            output_schema=graph.output_schema,
            nodes=[
                GraphNode(
                    id=node.id,
                    block_id=node.block_id,
                    input_default=node.input_default,
                    metadata=node.metadata,
                )
                for node in graph.nodes
            ],
            links=[
                GraphLink(
                    id=link.id,
                    source_id=link.source_id,
                    sink_id=link.sink_id,
                    source_name=link.source_name,
                    sink_name=link.sink_name,
                    is_static=link.is_static,
                )
                for link in graph.links
            ],
            credentials_input_schema=graph.credentials_input_schema,
        )


class GraphSettings(BaseModel):
    """Settings for a graph."""

    human_in_the_loop_safe_mode: Optional[bool] = Field(
        default=None, description="Enable safe mode for human-in-the-loop blocks"
    )

    def to_internal(self) -> "_GraphSettings":
        from backend.data.graph import GraphSettings as _GraphSettings

        settings = _GraphSettings()
        if self.human_in_the_loop_safe_mode is not None:
            settings.human_in_the_loop_safe_mode = self.human_in_the_loop_safe_mode
        return settings


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


class BlockInfo(BaseModel):
    """A building block that can be used in graphs."""

    id: str
    name: str
    description: str
    categories: list["BlockCategoryInfo"] = Field(default_factory=list)
    contributors: list["BlockContributorInfo"] = Field(default_factory=list)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    static_output: bool
    ui_type: block_types.BlockType
    costs: list["BlockCostInfo"] = Field(default_factory=list)

    @classmethod
    def from_internal(cls, b: block_types.AnyBlockSchema) -> "BlockInfo":
        from backend.data.credit import get_block_cost

        return cls(
            id=b.id,
            name=b.name,
            description=b.description,
            categories=[
                BlockCategoryInfo(category=c.name, description=c.value)
                for c in b.categories
            ],
            contributors=[BlockContributorInfo(name=c.name) for c in b.contributors],
            input_schema=b.input_schema.jsonschema(),
            output_schema=b.output_schema.jsonschema(),
            static_output=b.static_output,
            ui_type=b.block_type,
            costs=[
                BlockCostInfo(
                    cost_type=c.cost_type,
                    cost_filter=c.cost_filter,
                    cost_amount=c.cost_amount,
                )
                for c in get_block_cost(b)
            ],
        )


class BlockCategoryInfo(BaseModel):
    """A block's category."""

    category: str = Field(description="Category identifier")
    description: str = Field(description="Category description")


class BlockCostInfo(BaseModel):
    """Cost information for a block."""

    cost_type: block_types.BlockCostType = Field(
        description="Type of cost (e.g., 'run', 'byte', 'second')"
    )
    cost_filter: dict[str, Any] = Field(
        default_factory=dict, description="Conditions for this cost"
    )
    cost_amount: int = Field(description="Cost amount in credits")


class BlockContributorInfo(BaseModel):
    name: str


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

    @classmethod
    def from_internal(cls, job: GraphExecutionJobInfo) -> Self:
        next_run = (
            datetime.fromisoformat(job.next_run_time) if job.next_run_time else None
        )
        return cls(
            id=job.id,
            name=job.name or "",
            graph_id=job.graph_id,
            graph_version=job.graph_version,
            cron=job.cron,
            input_data=job.input_data,
            next_run_time=next_run,
            is_enabled=True,
        )


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

    @classmethod
    def from_internal(cls, agent: _LibraryAgent) -> Self:
        return cls(
            id=agent.id,
            graph_id=agent.graph_id,
            graph_version=agent.graph_version,
            name=agent.name,
            description=agent.description,
            is_favorite=agent.is_favorite,
            can_access_graph=agent.can_access_graph,
            is_latest_version=agent.is_latest_version,
            image_url=agent.image_url,
            creator_name=agent.creator_name,
            input_schema=agent.input_schema,
            output_schema=agent.output_schema,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
        )


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


RunStatus: TypeAlias = Literal[
    "INCOMPLETE", "QUEUED", "RUNNING", "COMPLETED", "TERMINATED", "FAILED", "REVIEW"
]


class Run(BaseModel):
    """An execution run."""

    id: str
    graph_id: str
    graph_version: int
    status: RunStatus
    started_at: datetime | None
    ended_at: datetime | None = None
    inputs: Optional[dict[str, Any]] = None
    cost: int = Field(default=0, description="Cost in cents ($)")
    duration: float = Field(default=0, description="Duration in seconds")
    node_exec_count: int = Field(default=0, description="Number of nodes executed")

    @classmethod
    def from_internal(cls, exec: GraphExecutionMeta) -> Self:
        """Convert internal execution to v2 API Run model."""
        return cls(
            id=exec.id,
            graph_id=exec.graph_id,
            graph_version=exec.graph_version,
            status=exec.status.value,
            started_at=exec.started_at,
            ended_at=exec.ended_at,
            inputs=exec.inputs,
            cost=exec.stats.cost if exec.stats else 0,
            duration=exec.stats.duration if exec.stats else 0,
            node_exec_count=exec.stats.node_exec_count if exec.stats else 0,
        )


class NodeExecution(BaseModel):
    """Result of a single node execution within a run."""

    node_id: str
    status: RunStatus
    input_data: dict[str, Any] = Field(
        default_factory=dict, description="Input values keyed by pin name"
    )
    output_data: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Output values keyed by pin name, each with a list of results",
    )
    started_at: datetime | None = None
    ended_at: datetime | None = None


class RunDetails(Run):
    """Detailed information about a run including outputs and node executions."""

    outputs: Optional[dict[str, list[Any]]] = None
    node_executions: Optional[list[NodeExecution]] = Field(
        description="Individual node execution results; "
        "may be omitted in case of permission restrictions"
    )

    @classmethod
    def from_internal(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, exec: GraphExecutionWithNodes
    ) -> Self:
        """Convert internal execution with nodes to v2 API RunDetails model."""
        return cls(
            id=exec.id,
            graph_id=exec.graph_id,
            graph_version=exec.graph_version,
            status=exec.status.value,
            started_at=exec.started_at,
            ended_at=exec.ended_at,
            inputs=exec.inputs,
            outputs=exec.outputs,
            cost=exec.stats.cost if exec.stats else 0,
            duration=exec.stats.duration if exec.stats else 0,
            node_exec_count=exec.stats.node_exec_count if exec.stats else 0,
            node_executions=[
                NodeExecution(
                    node_id=node.node_id,
                    status=node.status.value,
                    input_data=node.input_data,
                    output_data=node.output_data,
                    started_at=node.start_time,
                    ended_at=node.end_time,
                )
                for node in exec.node_executions
            ],
        )


class RunsListResponse(PaginatedResponse):
    """Response for listing runs."""

    runs: list[Run]


# ============================================================================
# Run Review Models (Human-in-the-loop)
# ============================================================================


class PendingReview(BaseModel):
    """A pending human-in-the-loop review."""

    Status = Literal["WAITING", "APPROVED", "REJECTED"]

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
    status: Status
    created_at: datetime

    @classmethod
    def from_internal(cls, review: PendingHumanReviewModel) -> Self:
        return cls(
            id=review.node_exec_id,
            run_id=review.graph_exec_id,
            graph_id=review.graph_id,
            graph_version=review.graph_version,
            payload=review.payload,
            instructions=review.instructions,
            editable=review.editable,
            status=review.status.value,
            created_at=review.created_at,
        )


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

    Type = Literal["TOP_UP", "USAGE", "GRANT", "REFUND", "CARD_CHECK"]

    transaction_key: str
    amount: int = Field(description="Transaction amount (positive or negative)")
    type: Type
    transaction_time: datetime
    running_balance: Optional[int] = Field(
        default=None, description="Balance after this transaction"
    )
    description: Optional[str] = None

    @classmethod
    def from_internal(cls, t: UserTransaction) -> Self:
        return cls(
            transaction_key=t.transaction_key,
            amount=t.amount,
            type=t.transaction_type.value,
            transaction_time=t.transaction_time,
            running_balance=t.running_balance,
            description=t.description,
        )


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

    @classmethod
    def from_internal(cls, cred: Credentials) -> Self:
        from backend.data.model import OAuth2Credentials

        scopes: list[str] = []
        if isinstance(cred, OAuth2Credentials):
            scopes = cred.scopes or []
        return cls(
            id=cred.id,
            provider=cred.provider,
            title=cred.title,
            scopes=scopes,
        )


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

    @classmethod
    def from_internal(cls, agent: StoreAgent) -> Self:
        return cls(
            slug=agent.slug,
            name=agent.agent_name,
            description=agent.description,
            sub_heading=agent.sub_heading,
            creator=agent.creator,
            creator_avatar=agent.creator_avatar,
            runs=agent.runs,
            rating=agent.rating,
            image_url=agent.agent_image,
        )


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

    @classmethod
    def from_internal(cls, agent: StoreAgentDetails) -> Self:
        return cls(
            store_listing_version_id=agent.store_listing_version_id,
            slug=agent.slug,
            name=agent.agent_name,
            description=agent.description,
            sub_heading=agent.sub_heading,
            instructions=agent.instructions,
            creator=agent.creator,
            creator_avatar=agent.creator_avatar,
            categories=agent.categories,
            runs=agent.runs,
            rating=agent.rating,
            image_urls=agent.agent_image,
            video_url=agent.agent_video,
            versions=agent.versions,
            agent_graph_versions=agent.agentGraphVersions,
            agent_graph_id=agent.agentGraphId,
            last_updated=agent.last_updated,
        )


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

    @classmethod
    def from_internal(cls, creator: Creator) -> Self:
        return cls(
            name=creator.name,
            username=creator.username,
            description=creator.description,
            avatar_url=creator.avatar_url,
            num_agents=creator.num_agents,
            agent_rating=creator.agent_rating,
            agent_runs=creator.agent_runs,
            is_featured=creator.is_featured,
        )


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

    @classmethod
    def from_internal(cls, creator: CreatorDetails) -> Self:
        return cls(
            name=creator.name,
            username=creator.username,
            description=creator.description,
            avatar_url=creator.avatar_url,
            agent_rating=creator.agent_rating,
            agent_runs=creator.agent_runs,
            top_categories=creator.top_categories,
            links=creator.links,
        )


class MarketplaceCreatorsResponse(PaginatedResponse):
    """Response for listing marketplace creators."""

    creators: list[MarketplaceCreator]


class MarketplaceSubmission(BaseModel):
    """A marketplace submission."""

    Status = Literal["DRAFT", "PENDING", "APPROVED", "REJECTED"]

    graph_id: str
    graph_version: int
    name: str
    sub_heading: str
    slug: str
    description: str
    instructions: Optional[str] = None
    image_urls: list[str] = Field(default_factory=list)
    date_submitted: datetime
    status: Status
    runs: int = Field(default=0)
    rating: float = Field(default=0.0)
    store_listing_version_id: Optional[str] = None
    version: Optional[int] = None
    review_comments: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    video_url: Optional[str] = None
    categories: list[str] = Field(default_factory=list)

    @classmethod
    def from_internal(cls, sub: StoreSubmission) -> Self:
        return cls(
            graph_id=sub.agent_id,
            graph_version=sub.agent_version,
            name=sub.name,
            sub_heading=sub.sub_heading,
            slug=sub.slug,
            description=sub.description,
            instructions=sub.instructions,
            image_urls=sub.image_urls,
            date_submitted=sub.date_submitted,
            status=sub.status.value,
            runs=sub.runs,
            rating=sub.rating,
            store_listing_version_id=sub.store_listing_version_id,
            version=sub.version,
            review_comments=sub.review_comments,
            reviewed_at=sub.reviewed_at,
            video_url=sub.video_url,
            categories=sub.categories,
        )


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
