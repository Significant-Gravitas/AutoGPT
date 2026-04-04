"""
V2 External API - Request and Response Models

This module defines all request and response models for the v2 external API.
All models are self-contained and specific to the external API contract.

Route files should import models from here rather than defining them locally.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Optional, Self, TypeAlias

from pydantic import BaseModel, Field, JsonValue, field_validator

import backend.blocks._base as block_types

if TYPE_CHECKING:
    from backend.api.features.executions.review.model import PendingHumanReviewModel
    from backend.api.features.library.model import LibraryAgent as _LibraryAgent
    from backend.api.features.library.model import (
        LibraryAgentPreset as _LibraryAgentPreset,
    )
    from backend.api.features.library.model import LibraryFolder as _LibraryFolder
    from backend.api.features.library.model import (
        LibraryFolderTree as _LibraryFolderTree,
    )
    from backend.api.features.store.model import CreatorDetails
    from backend.api.features.store.model import ProfileDetails as _ProfileDetails
    from backend.api.features.store.model import (
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


class GraphCreateRequest(BaseModel):
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
    """Existing agent graph metadata (summary information)."""

    id: str
    version: int
    is_active: bool
    name: str
    description: str
    instructions: str | None
    recommended_schedule_cron: str | None
    created_at: datetime

    forked_from_id: str | None
    forked_from_version: int | None

    @classmethod
    def from_internal(cls, graph: _GraphMeta) -> Self:
        return cls(
            id=graph.id,
            version=graph.version,
            is_active=graph.is_active,
            name=graph.name,
            description=graph.description,
            instructions=graph.instructions,
            recommended_schedule_cron=graph.recommended_schedule_cron,
            created_at=graph.created_at,
            forked_from_id=graph.forked_from_id,
            forked_from_version=graph.forked_from_version,
        )


class Graph(GraphMeta):
    """Existing agent graph details, including nodes and links."""

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
            instructions=graph.instructions,
            recommended_schedule_cron=graph.recommended_schedule_cron,
            created_at=graph.created_at,
            forked_from_id=graph.forked_from_id,
            forked_from_version=graph.forked_from_version,
            input_schema=graph.input_schema,
            output_schema=graph.output_schema,
            credentials_input_schema=graph.credentials_input_schema,
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


class GraphSetActiveVersionRequest(BaseModel):
    """Request to set the active graph version."""

    active_graph_version: int = Field(description="Version number to set as active")


class GraphListResponse(PaginatedResponse):
    """Response for listing graphs."""

    graphs: list[GraphMeta]


# ============================================================================
# Block Models
# ============================================================================


class BlockInfo(BaseModel):
    """A building block that can be used in graphs."""

    id: str
    name: str
    description: str
    categories: list["BlockCategoryInfo"]
    contributors: list["BlockContributorInfo"]
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    static_output: bool
    block_type: block_types.BlockType
    costs: list["BlockCostInfo"]

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
            block_type=b.block_type,
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
        description="Partial node input that, if it matches the input "
        "for an execution of this block, applies this cost to it"
    )
    cost_amount: int = Field(description="Cost (× $0.01) per {cost_type}")


class BlockContributorInfo(BaseModel):
    name: str


# ============================================================================
# Schedule Models
# ============================================================================


class AgentRunSchedule(BaseModel):
    """An execution schedule for an agent."""

    id: str
    name: str
    graph_id: str
    graph_version: int
    cron: str = Field(description="Cron expression for the schedule")
    input_data: dict[str, Any] = Field(
        description="Input data for scheduled executions"
    )
    next_run_time: Optional[datetime]
    is_enabled: bool

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


class AgentRunScheduleCreateRequest(BaseModel):
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


class AgentRunScheduleListResponse(PaginatedResponse):
    """Response for listing agent run schedules."""

    schedules: list[AgentRunSchedule]


# ============================================================================
# Library Models
# ============================================================================


class TriggerSetupInfo(BaseModel):
    """
    Trigger configuration requirements for agents that support webhook triggers.

    Use `config_schema` and `credentials_input_name` to populate the
    `trigger_config` and `agent_credentials` fields when calling
    ``POST /library/presets/setup-trigger``.
    """

    provider: str = Field(description="Trigger provider (e.g. 'github')")
    config_schema: dict[str, Any] = Field(
        description="JSON Schema for the trigger block's input"
    )
    credentials_input_name: Optional[str] = Field(
        description=(
            "Name of the credentials input field, if the trigger requires credentials"
        )
    )


class LibraryAgent(BaseModel):
    """An agent in the user's library."""

    id: str
    graph_id: str
    graph_version: int
    name: str
    description: str
    is_favorite: bool
    can_access_graph: bool
    is_latest_version: bool
    image_url: Optional[str]
    creator_name: str
    input_schema: dict[str, Any] = Field(description="Input schema for the agent")
    output_schema: dict[str, Any] = Field(description="Output schema for the agent")
    trigger_setup_info: Optional[TriggerSetupInfo] = Field(
        default=None,
        description="Trigger configuration requirements; "
        "present if the agent has a webhook trigger input",
    )
    recommended_schedule_cron: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_internal(cls, agent: _LibraryAgent) -> Self:
        trigger_info = None
        if agent.trigger_setup_info:
            trigger_info = TriggerSetupInfo(
                provider=agent.trigger_setup_info.provider,
                config_schema=agent.trigger_setup_info.config_schema,
                credentials_input_name=agent.trigger_setup_info.credentials_input_name,
            )

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
            trigger_setup_info=trigger_info,
            recommended_schedule_cron=agent.recommended_schedule_cron,
            created_at=agent.created_at,
            updated_at=agent.updated_at,
        )


class LibraryAgentListResponse(PaginatedResponse):
    """Response for listing library agents."""

    agents: list[LibraryAgent]


class LibraryAgentUpdateRequest(BaseModel):
    """Request to update a library agent."""

    auto_update_version: Optional[bool] = None
    graph_version: Optional[int] = None
    is_favorite: Optional[bool] = None
    is_archived: Optional[bool] = None
    folder_id: Optional[str] = None


class AgentRunRequest(BaseModel):
    """Request to execute an agent."""

    inputs: dict[str, Any] = Field(
        default_factory=dict, description="Input values for the agent"
    )
    credentials_inputs: dict[str, Any] = Field(
        default_factory=dict, description="Credentials for the agent"
    )


# ============================================================================
# Library Folder Models
# ============================================================================


class LibraryFolder(BaseModel):
    """A folder for organizing library agents."""

    id: str
    name: str
    icon: Optional[str]
    color: Optional[str]
    parent_id: Optional[str]
    agent_count: int
    subfolder_count: int
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_internal(cls, f: _LibraryFolder) -> Self:
        return cls(
            id=f.id,
            name=f.name,
            icon=f.icon,
            color=f.color,
            parent_id=f.parent_id,
            agent_count=f.agent_count,
            subfolder_count=f.subfolder_count,
            created_at=f.created_at,
            updated_at=f.updated_at,
        )


class LibraryFolderTree(BaseModel):
    """Recursive folder tree node."""

    id: str
    name: str
    icon: Optional[str]
    color: Optional[str]
    agent_count: int
    subfolder_count: int
    children: list["LibraryFolderTree"]

    @classmethod
    def from_internal(cls, f: _LibraryFolderTree) -> Self:
        return cls(
            id=f.id,
            name=f.name,
            icon=f.icon,
            color=f.color,
            agent_count=f.agent_count,
            subfolder_count=f.subfolder_count,
            children=[LibraryFolderTree.from_internal(c) for c in f.children],
        )


class LibraryFolderListResponse(BaseModel):
    """Response for listing folders."""

    folders: list[LibraryFolder]


class LibraryFolderTreeResponse(BaseModel):
    """Response for folder tree."""

    tree: list[LibraryFolderTree]


class LibraryFolderCreateRequest(BaseModel):
    """Request to create a folder."""

    name: str = Field(min_length=1, max_length=100)
    icon: Optional[str] = None
    color: Optional[str] = Field(
        default=None, pattern=r"^#[0-9A-Fa-f]{6}$", description="Hex color (#RRGGBB)"
    )
    parent_id: Optional[str] = None


class LibraryFolderUpdateRequest(BaseModel):
    """Request to update a folder."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    icon: Optional[str] = None
    color: Optional[str] = None


class LibraryFolderMoveRequest(BaseModel):
    """Request to move a folder."""

    target_parent_id: Optional[str] = Field(
        default=None, description="Target parent folder ID (null = root)"
    )


# ============================================================================
# Preset Models
# ============================================================================


class AgentPreset(BaseModel):
    """A saved preset configuration for running an agent."""

    id: str
    graph_id: str
    graph_version: int
    name: str
    description: str
    is_active: bool
    inputs: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_internal(cls, p: _LibraryAgentPreset) -> Self:
        return cls(
            id=p.id,
            graph_id=p.graph_id,
            graph_version=p.graph_version,
            name=p.name,
            description=p.description,
            is_active=p.is_active,
            inputs=p.inputs,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )


class AgentPresetListResponse(PaginatedResponse):
    """Response for listing presets."""

    presets: list[AgentPreset]


class AgentPresetCreateRequest(BaseModel):
    """Request to create a preset."""

    graph_id: str = Field(description="Graph ID")
    graph_version: int = Field(description="Graph version")
    name: str = Field(description="Preset name")
    description: str = Field(default="", description="Preset description")
    inputs: dict[str, Any] = Field(default_factory=dict, description="Input values")
    credentials: dict[str, Any] = Field(
        default_factory=dict, description="Credential references"
    )
    is_active: bool = Field(default=True, description="Whether the preset is active")


class AgentPresetUpdateRequest(BaseModel):
    """Request to update a preset."""

    name: Optional[str] = None
    description: Optional[str] = None
    inputs: Optional[dict[str, Any]] = None
    credentials: Optional[dict[str, Any]] = None
    is_active: Optional[bool] = None


class AgentTriggerSetupRequest(BaseModel):
    """Request to set up a webhook-triggered preset."""

    name: str = Field(description="Preset name")
    description: str = Field(default="", description="Preset description")
    graph_id: str = Field(description="Graph ID")
    graph_version: int = Field(description="Graph version")
    trigger_config: dict[str, Any] = Field(description="Trigger block configuration")
    agent_credentials: dict[str, Any] = Field(
        default_factory=dict, description="Credential references"
    )


class AgentPresetRunRequest(BaseModel):
    """Request to run an agent preset with optional overrides."""

    inputs: dict[str, Any] = Field(default_factory=dict, description="Input overrides")
    credentials_inputs: dict[str, Any] = Field(
        default_factory=dict, description="Credential overrides"
    )


# ============================================================================
# Run Models
# ============================================================================


RunStatus: TypeAlias = Literal[
    "INCOMPLETE", "QUEUED", "RUNNING", "COMPLETED", "TERMINATED", "FAILED", "REVIEW"
]


class AgentGraphRun(BaseModel):
    id: str
    graph_id: str
    graph_version: int
    status: RunStatus
    started_at: datetime | None
    ended_at: datetime | None
    inputs: Optional[dict[str, Any]]
    cost: int = Field(description="Cost in cents ($)")
    duration: float = Field(description="Duration in seconds")
    node_exec_count: int = Field(description="Number of nodes executed")
    correctness_score: float | None = Field(
        description=(
            "AI-generated score (0.0-1.0) indicating how well "
            "the execution achieved its intended purpose"
        ),
    )

    @classmethod
    def from_internal(cls, exec: GraphExecutionMeta) -> Self:
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
            correctness_score=exec.stats.correctness_score if exec.stats else None,
        )


class AgentGraphRunDetails(AgentGraphRun):
    """Detailed information about a run including outputs and node executions."""

    outputs: Optional[dict[str, list[Any]]]
    node_executions: Optional[list[AgentNodeExecution]] = Field(
        description="Individual node execution results; "
        "may be omitted in case of permission restrictions"
    )

    @classmethod
    def from_internal(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, exec: GraphExecutionWithNodes
    ) -> Self:
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
            correctness_score=exec.stats.correctness_score if exec.stats else None,
            node_executions=[
                AgentNodeExecution(
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


class AgentNodeExecution(BaseModel):
    """Result of a single node execution within an agent run."""

    node_id: str
    status: RunStatus
    input_data: dict[str, Any] = Field(description="Input values keyed by pin name")
    output_data: dict[str, list[Any]] = Field(
        description="Output values keyed by pin name, each with a list of results",
    )
    started_at: datetime | None
    ended_at: datetime | None


class AgentRunListResponse(PaginatedResponse):
    """Response for listing agent runs."""

    runs: list[AgentGraphRun]


class AgentRunShareResponse(BaseModel):
    """Response after enabling sharing for an agent run."""

    share_url: str = Field(description="Public URL for the shared run")
    share_token: str = Field(description="Unique share token")


# ============================================================================
# Run Review Models (Human-in-the-loop)
# ============================================================================


AgentRunReviewStatus: TypeAlias = Literal["WAITING", "APPROVED", "REJECTED"]


class AgentRunReview(BaseModel):
    """A human-in-the-loop review for an agent run."""

    node_exec_id: str  # primary key for reviews
    run_id: str = Field(description="Graph Execution ID")
    graph_id: str
    graph_version: int
    payload: JsonValue = Field(description="Data to be reviewed")
    instructions: Optional[str] = Field(description="Instructions for the reviewer")
    editable: bool = Field(description="Whether the reviewer can edit the data")
    status: AgentRunReviewStatus
    requested_at: datetime
    reviewed_at: datetime | None
    processed: bool = Field(
        description="Whether the review has been consumed by the execution engine"
    )
    reviewer_comment: str | None

    @classmethod
    def from_internal(cls, review: PendingHumanReviewModel) -> Self:
        return cls(
            node_exec_id=review.node_exec_id,
            run_id=review.graph_exec_id,
            graph_id=review.graph_id,
            graph_version=review.graph_version,
            payload=review.payload,
            instructions=review.instructions,
            editable=review.editable,
            status=review.status.value,
            requested_at=review.created_at,
            reviewed_at=review.reviewed_at,
            processed=review.processed,
            reviewer_comment=review.review_message,
        )


class AgentRunReviewsResponse(PaginatedResponse):
    """Response for listing run reviews."""

    reviews: list[AgentRunReview]


class AgentRunReviewDecision(BaseModel):
    """Decision for a single review item."""

    node_exec_id: str = Field(description="Node execution ID (review ID)")
    approved: bool = Field(description="Whether to approve the data")
    edited_payload: Optional[JsonValue] = Field(
        default=None, description="Modified payload data (if editing)"
    )
    message: Optional[str] = Field(
        default=None, description="Optional message from reviewer", max_length=2000
    )


class AgentRunReviewsSubmitRequest(BaseModel):
    """Request to submit review responses for all pending reviews of an execution."""

    reviews: list[AgentRunReviewDecision] = Field(
        description="All review decisions for the execution"
    )


class AgentRunReviewsSubmitResponse(BaseModel):
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


TransactionType: TypeAlias = Literal["TOP_UP", "USAGE", "GRANT", "REFUND", "CARD_CHECK"]


class CreditTransactionOrigin(BaseModel):
    """Origin context for a credit transaction."""

    graph_id: Optional[str] = Field(
        description="ID of the graph that caused this transaction, if applicable"
    )
    run_id: Optional[str] = Field(
        description="ID of the agent run that caused this transaction, if applicable"
    )


class CreditTransaction(BaseModel):
    """A credit transaction."""

    transaction_key: str
    amount: int = Field(description="Transaction amount (positive or negative)")
    type: TransactionType
    transaction_time: datetime
    running_balance: Optional[int] = Field(description="Balance after this transaction")
    description: Optional[str]
    reason: Optional[str] = Field(
        description=(
            "Contextual explanation for this transaction "
            "(e.g. admin grant reason, refund justification)"
        ),
    )
    origin: Optional[CreditTransactionOrigin] = Field(
        description="Origin context linking this transaction to a graph (execution)",
    )

    @classmethod
    def from_internal(cls, t: UserTransaction) -> Self:
        origin = None
        if t.usage_graph_id or t.usage_execution_id:
            origin = CreditTransactionOrigin(
                graph_id=t.usage_graph_id,
                run_id=t.usage_execution_id,
            )
        return cls(
            transaction_key=t.transaction_key,
            amount=t.amount,
            type=t.transaction_type.value,
            transaction_time=t.transaction_time,
            running_balance=t.running_balance,
            description=t.description,
            reason=t.reason,
            origin=origin,
        )


class CreditTransactionsResponse(PaginatedResponse):
    """Response for listing credit transactions."""

    transactions: list[CreditTransaction]


# ============================================================================
# Integration Models
# ============================================================================


CredentialType: TypeAlias = Literal["api_key", "oauth2", "user_password", "host_scoped"]


class CredentialInfo(BaseModel):
    """A user's credential for an integration."""

    id: str
    type: CredentialType
    provider: str = Field(description="Integration provider name")
    title: Optional[str] = Field(description="User-assigned title for this credential")
    scopes: list[str] = Field(
        description="Permission scopes granted to this credential"
    )
    expires_at: Optional[datetime]

    @classmethod
    def from_internal(cls, cred: Credentials) -> Self:
        scopes: list[str] = []
        expires_at: int | None = None
        if cred.type == "oauth2":
            scopes = cred.scopes
            expires_at = cred.refresh_token_expires_at
        elif cred.type == "api_key":
            expires_at = cred.expires_at

        return cls(
            id=cred.id,
            type=cred.type,
            provider=cred.provider,
            title=cred.title,
            scopes=scopes,
            expires_at=(
                datetime.fromtimestamp(expires_at, tz=timezone.utc)
                if expires_at
                else None
            ),
        )


class CredentialListResponse(BaseModel):
    """Response for listing credentials."""

    credentials: list[CredentialInfo]


class _CredentialCreateBase(BaseModel):
    provider: str = Field(description="Provider name (e.g., 'github', 'openai')")
    title: Optional[str] = Field(
        default=None, description="User-friendly name for this credential"
    )


class APIKeyCredentialCreateRequest(_CredentialCreateBase):
    """Request to create an API key credential."""

    type: Literal["api_key"]
    api_key: str


class UserPasswordCredentialCreateRequest(_CredentialCreateBase):
    """Request to create a username/password credential."""

    type: Literal["user_password"]
    username: str
    password: str


class HostScopedCredentialCreateRequest(_CredentialCreateBase):
    """Request to create a host-scoped credential with custom headers."""

    type: Literal["host_scoped"]
    host: str = Field(
        description=(
            "Host pattern to match against request URLs. "
            "Supports exact hosts (api.example.com), wildcard subdomains "
            "(*.example.com), and optional port (api.example.com:8080)"
        ),
    )
    headers: dict[str, str] = Field(
        description="Key-value header map to add to matching requests"
    )

    @field_validator("host")
    @classmethod
    def validate_host(cls, v: str) -> str:
        """
        Validates that `host` is a pattern compatible with
        `HostScopedCredentials.matches_url()`, which supports exact hosts,
        `*.wildcard` subdomains, and optional ports.
        """
        import ipaddress
        import re

        from urllib3.util import parse_url

        v = v.strip()
        if not v:
            raise ValueError("host must not be empty")

        try:
            parsed = parse_url(v)
        except Exception:
            # parse_url can't handle bare IPv6 like "::1";
            # check if it's a valid IP before rejecting
            try:
                ipaddress.ip_address(v.strip("[]"))
                return v
            except ValueError:
                pass
            raise ValueError(f"Invalid host pattern: {v}")

        # If a full URL was given, extract just the host part
        if parsed.scheme:
            raise ValueError(
                f"host must be a host pattern, not a URL: "
                f"omit the scheme ({parsed.scheme}://)"
            )
        if parsed.path and parsed.path != "/":
            raise ValueError("host must be a host pattern without a path component")

        # Validate the hostname portion (with optional *. prefix)
        hostname = parsed.hostname or v.split(":")[0]

        # Allow IPv4 and IPv6 addresses (matches_url handles them via exact match)
        bare = hostname.strip("[]")  # strip brackets from [::1]-style IPv6
        try:
            ipaddress.ip_address(bare)
            return v  # valid IP, skip domain validation
        except ValueError:
            pass

        if hostname.startswith("*."):
            domain = hostname[2:]
        else:
            domain = hostname

        # Domain validation: labels separated by dots, no empty labels
        if not re.match(
            r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*$",
            domain,
        ):
            raise ValueError(
                f"Invalid hostname: {hostname}. "
                "Expected a domain like api.example.com, *.example.com, "
                "or an IP address"
            )

        return v


CredentialCreateRequest = (
    APIKeyCredentialCreateRequest
    | UserPasswordCredentialCreateRequest
    | HostScopedCredentialCreateRequest
)


class CredentialRequirement(BaseModel):
    """A credential requirement for an agent (graph)."""

    provider: str = Field(description="Required provider name")
    required_scopes: list[str] = Field(description="Required scopes")
    matching_credentials: list[CredentialInfo] = Field(
        description="User's credentials that match this requirement",
    )


class CredentialRequirementsResponse(BaseModel):
    """Response for listing credential requirements for an agent (graph)."""

    requirements: list[CredentialRequirement]


# ============================================================================
# File Workspace Models
# ============================================================================


class UploadWorkspaceFileResponse(BaseModel):
    """Response after uploading a file to the user's workspace."""

    file_uri: str = Field(description="URI to reference the uploaded file in agents")
    file_name: str
    size: int = Field(description="File size in bytes")
    content_type: str
    expires_in_hours: int


class WorkspaceFileInfo(BaseModel):
    """Metadata for a file in the user's workspace."""

    id: str
    name: str
    path: str
    mime_type: str
    size_bytes: int = Field(description="File size in bytes")
    created_at: datetime
    updated_at: datetime


class WorkspaceFileListResponse(PaginatedResponse):
    """Response for listing workspace files."""

    files: list[WorkspaceFileInfo]


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
    runs: int = Field(description="Number of times this agent has been run")
    rating: float = Field(description="Average rating")
    image_url: str

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


class MarketplaceAgentDetails(MarketplaceAgent):
    """Detailed information about a marketplace agent."""

    store_listing_version_id: str
    versions: list[int] = Field(
        description="Available store listing versions (sequential; != graph version)",
    )
    instructions: Optional[str]
    categories: list[str]
    image_urls: list[str]
    video_url: Optional[str]
    agent_output_demo_url: str
    recommended_schedule_cron: Optional[str]
    graph_id: str
    graph_versions: list[int]
    last_updated: datetime

    @classmethod
    def from_internal(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, agent: StoreAgentDetails
    ) -> Self:
        return cls(
            store_listing_version_id=agent.store_listing_version_id,
            slug=agent.slug,
            versions=[int(v) for v in agent.versions],
            name=agent.agent_name,
            description=agent.description,
            sub_heading=agent.sub_heading,
            instructions=agent.instructions,
            creator=agent.creator,
            creator_avatar=agent.creator_avatar,
            categories=agent.categories,
            runs=agent.runs,
            rating=agent.rating,
            image_url=agent.agent_image[0] if agent.agent_image else "",
            image_urls=agent.agent_image,
            video_url=agent.agent_video or None,
            agent_output_demo_url=agent.agent_output_demo,
            recommended_schedule_cron=agent.recommended_schedule_cron or None,
            graph_id=agent.graph_id,
            graph_versions=[int(v) for v in agent.graph_versions],
            last_updated=agent.last_updated,
        )


class MarketplaceAgentListResponse(PaginatedResponse):
    """Response for listing marketplace agents."""

    agents: list[MarketplaceAgent]


class MarketplaceUserProfile(BaseModel):
    """User's marketplace profile."""

    name: str
    username: str
    description: str
    links: list[str]
    avatar_url: Optional[str]
    is_featured: bool

    @classmethod
    def from_internal(cls, profile: _ProfileDetails) -> Self:
        return cls(
            name=profile.name,
            username=profile.username,
            description=profile.description,
            links=profile.links,
            avatar_url=profile.avatar_url,
            is_featured=profile.is_featured,
        )


class MarketplaceUserProfileUpdateRequest(BaseModel):
    """Request to partially update marketplace profile."""

    name: Optional[str] = Field(default=None, description="Display name")
    username: Optional[str] = Field(default=None, description="Unique username")
    description: Optional[str] = Field(default=None, description="Bio/description")
    links: Optional[list[str]] = Field(default=None, description="Profile links")
    avatar_url: Optional[str] = Field(default=None, description="Avatar image URL")


class MarketplaceCreatorDetails(MarketplaceUserProfile):
    """Profile + stats for a creator on the marketplace."""

    num_agents: int
    agent_runs: int
    agent_rating: float
    top_categories: list[str]

    @classmethod
    def from_internal(  # pyright: ignore[reportIncompatibleMethodOverride]
        cls, creator: CreatorDetails
    ) -> Self:
        return cls(
            name=creator.name,
            username=creator.username,
            avatar_url=creator.avatar_url,
            description=creator.description,
            links=creator.links,
            is_featured=creator.is_featured,
            num_agents=creator.num_agents,
            agent_runs=creator.agent_runs,
            agent_rating=creator.agent_rating,
            top_categories=creator.top_categories,
        )


class MarketplaceCreatorsResponse(PaginatedResponse):
    """Response for listing marketplace creators."""

    creators: list[MarketplaceCreatorDetails]


SubmissionStatus: TypeAlias = Literal["DRAFT", "PENDING", "APPROVED", "REJECTED"]


class MarketplaceAgentSubmission(BaseModel):
    """A marketplace submission."""

    listing_id: str
    listing_version_id: str
    listing_version: int
    graph_id: str
    graph_version: int
    slug: str
    name: str
    sub_heading: str
    description: str
    instructions: Optional[str]
    categories: list[str]
    image_urls: list[str]
    video_url: Optional[str]
    agent_output_demo_url: Optional[str]
    changes_summary: Optional[str] = Field(
        description="Summary of changes in this version"
    )

    submitted_at: Optional[datetime]
    submission_status: SubmissionStatus
    submission_reviewed_at: datetime | None
    submission_review_comments: Optional[str] = Field(
        description="Comments by the admin who reviewed the submission"
    )

    # Aggregated stats
    run_count: int
    user_review_count: int
    user_review_avg_rating: float

    @classmethod
    def from_internal(cls, sub: StoreSubmission) -> Self:
        return cls(
            listing_id=sub.listing_id,
            listing_version_id=sub.listing_version_id,
            listing_version=sub.listing_version,
            graph_id=sub.graph_id,
            graph_version=sub.graph_version,
            slug=sub.slug,
            name=sub.name,
            sub_heading=sub.sub_heading,
            description=sub.description,
            instructions=sub.instructions,
            categories=sub.categories,
            image_urls=sub.image_urls,
            video_url=sub.video_url,
            agent_output_demo_url=sub.agent_output_demo_url,
            changes_summary=sub.changes_summary,
            submitted_at=sub.submitted_at,
            submission_status=sub.status.value,
            submission_reviewed_at=sub.reviewed_at,
            submission_review_comments=sub.review_comments,
            run_count=sub.run_count,
            user_review_count=sub.review_count,
            user_review_avg_rating=sub.review_avg_rating,
        )


class MarketplaceAgentSubmissionCreateRequest(BaseModel):
    """Request to create a marketplace submission."""

    graph_id: str = Field(description="ID of the graph to submit")
    graph_version: int = Field(description="Version of the graph to submit")
    name: str = Field(description="Display name for the agent")
    slug: str = Field(description="URL-friendly identifier")
    description: str = Field(description="Full description")
    sub_heading: str = Field(description="Short tagline")
    image_urls: list[str] = Field(default_factory=list)
    video_url: Optional[str] = None
    agent_output_demo_url: Optional[str] = None
    categories: list[str] = Field(default_factory=list)
    instructions: Optional[str] = Field(default=None, description="Usage instructions")
    changes_summary: Optional[str] = Field(
        default="Initial Submission", description="Summary of changes"
    )
    recommended_schedule_cron: Optional[str] = Field(
        default=None, description="Recommended cron schedule"
    )


class MarketplaceAgentSubmissionsListResponse(PaginatedResponse):
    """Response for listing submissions."""

    submissions: list[MarketplaceAgentSubmission]


class MarketplaceAgentSubmissionEditRequest(BaseModel):
    """Request to edit a marketplace submission."""

    name: str = Field(description="Agent display name")
    sub_heading: str = Field(default="", description="Short tagline")
    description: str = Field(default="", description="Full description")
    image_urls: list[str] = Field(default_factory=list, description="Image URLs")
    video_url: Optional[str] = Field(default=None, description="Demo video URL")
    agent_output_demo_url: Optional[str] = Field(
        default=None, description="Agent output demo URL"
    )
    categories: list[str] = Field(default_factory=list, description="Categories")
    changes_summary: Optional[str] = Field(
        default="Update submission", description="Summary of changes"
    )
    recommended_schedule_cron: Optional[str] = Field(
        default=None, description="Recommended cron schedule"
    )
    instructions: Optional[str] = Field(default=None, description="Usage instructions")


class MarketplaceMediaUploadResponse(BaseModel):
    """Response after uploading media."""

    url: str = Field(description="Public URL of the uploaded media")


SearchContentType: TypeAlias = Literal[
    "STORE_AGENT", "BLOCK", "INTEGRATION", "DOCUMENTATION", "LIBRARY_AGENT"
]


class MarketplaceSearchResult(BaseModel):
    """A single search result from marketplace search."""

    content_type: SearchContentType
    content_id: str
    searchable_text: str
    metadata: Optional[dict]
    updated_at: Optional[datetime]
    combined_score: Optional[float]


class MarketplaceSearchResponse(PaginatedResponse):
    """Response for marketplace search."""

    results: list[MarketplaceSearchResult]
