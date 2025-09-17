import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import prisma.enums
import prisma.models
import pydantic

import backend.data.block as block_model
import backend.data.graph as graph_model
from backend.data.model import CredentialsMetaInput, is_credentials_field_name
from backend.util.models import Pagination

if TYPE_CHECKING:
    from backend.data.integrations import Webhook


class LibraryAgentStatus(str, Enum):
    COMPLETED = "COMPLETED"  # All runs completed
    HEALTHY = "HEALTHY"  # Agent is running (not all runs have completed)
    WAITING = "WAITING"  # Agent is queued or waiting to start
    ERROR = "ERROR"  # Agent is in an error state


class LibraryAgent(pydantic.BaseModel):
    """
    Represents an agent in the library, including metadata for display and
    user interaction within the system.
    """

    id: str
    graph_id: str
    graph_version: int

    image_url: str | None

    creator_name: str
    creator_image_url: str

    status: LibraryAgentStatus

    updated_at: datetime.datetime

    name: str
    description: str
    instructions: str | None = None

    input_schema: dict[str, Any]  # Should be BlockIOObjectSubSchema in frontend
    output_schema: dict[str, Any]
    credentials_input_schema: dict[str, Any] | None = pydantic.Field(
        description="Input schema for credentials required by the agent",
    )

    has_external_trigger: bool = pydantic.Field(
        description="Whether the agent has an external trigger (e.g. webhook) node"
    )
    trigger_setup_info: Optional[graph_model.GraphTriggerInfo] = None

    # Indicates whether there's a new output (based on recent runs)
    new_output: bool

    # Whether the user can access the underlying graph
    can_access_graph: bool

    # Indicates if this agent is the latest version
    is_latest_version: bool

    # Whether the agent is marked as favorite by the user
    is_favorite: bool

    # Recommended schedule cron (from marketplace agents)
    recommended_schedule_cron: str | None = None

    @staticmethod
    def from_db(
        agent: prisma.models.LibraryAgent,
        sub_graphs: Optional[list[prisma.models.AgentGraph]] = None,
    ) -> "LibraryAgent":
        """
        Factory method that constructs a LibraryAgent from a Prisma LibraryAgent
        model instance.
        """
        if not agent.AgentGraph:
            raise ValueError("Associated Agent record is required.")

        graph = graph_model.GraphModel.from_db(agent.AgentGraph, sub_graphs=sub_graphs)

        agent_updated_at = agent.AgentGraph.updatedAt
        lib_agent_updated_at = agent.updatedAt

        # Compute updated_at as the latest between library agent and graph
        updated_at = (
            max(agent_updated_at, lib_agent_updated_at)
            if agent_updated_at
            else lib_agent_updated_at
        )

        creator_name = "Unknown"
        creator_image_url = ""
        if agent.Creator:
            creator_name = agent.Creator.name or "Unknown"
            creator_image_url = agent.Creator.avatarUrl or ""

        # Logic to calculate status and new_output
        week_ago = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=7
        )
        executions = agent.AgentGraph.Executions or []
        status_result = _calculate_agent_status(executions, week_ago)
        status = status_result.status
        new_output = status_result.new_output

        # Check if user can access the graph
        can_access_graph = agent.AgentGraph.userId == agent.userId

        # Hard-coded to True until a method to check is implemented
        is_latest_version = True

        return LibraryAgent(
            id=agent.id,
            graph_id=agent.agentGraphId,
            graph_version=agent.agentGraphVersion,
            image_url=agent.imageUrl,
            creator_name=creator_name,
            creator_image_url=creator_image_url,
            status=status,
            updated_at=updated_at,
            name=graph.name,
            description=graph.description,
            instructions=graph.instructions,
            input_schema=graph.input_schema,
            output_schema=graph.output_schema,
            credentials_input_schema=(
                graph.credentials_input_schema if sub_graphs is not None else None
            ),
            has_external_trigger=graph.has_external_trigger,
            trigger_setup_info=graph.trigger_setup_info,
            new_output=new_output,
            can_access_graph=can_access_graph,
            is_latest_version=is_latest_version,
            is_favorite=agent.isFavorite,
            recommended_schedule_cron=agent.AgentGraph.recommendedScheduleCron,
        )


class AgentStatusResult(pydantic.BaseModel):
    status: LibraryAgentStatus
    new_output: bool


def _calculate_agent_status(
    executions: list[prisma.models.AgentGraphExecution],
    recent_threshold: datetime.datetime,
) -> AgentStatusResult:
    """
    Helper function to determine the overall agent status and whether there
    is new output (i.e., completed runs within the recent threshold).

    :param executions: A list of AgentGraphExecution objects.
    :param recent_threshold: A datetime; any execution after this indicates new output.
    :return: (AgentStatus, new_output_flag)
    """

    if not executions:
        return AgentStatusResult(status=LibraryAgentStatus.COMPLETED, new_output=False)

    # Track how many times each execution status appears
    status_counts = {status: 0 for status in prisma.enums.AgentExecutionStatus}
    new_output = False

    for execution in executions:
        # Check if there's a completed run more recent than `recent_threshold`
        if execution.createdAt >= recent_threshold:
            if execution.executionStatus == prisma.enums.AgentExecutionStatus.COMPLETED:
                new_output = True
        status_counts[execution.executionStatus] += 1

    # Determine the final status based on counts
    if status_counts[prisma.enums.AgentExecutionStatus.FAILED] > 0:
        return AgentStatusResult(status=LibraryAgentStatus.ERROR, new_output=new_output)
    elif status_counts[prisma.enums.AgentExecutionStatus.QUEUED] > 0:
        return AgentStatusResult(
            status=LibraryAgentStatus.WAITING, new_output=new_output
        )
    elif status_counts[prisma.enums.AgentExecutionStatus.RUNNING] > 0:
        return AgentStatusResult(
            status=LibraryAgentStatus.HEALTHY, new_output=new_output
        )
    else:
        return AgentStatusResult(
            status=LibraryAgentStatus.COMPLETED, new_output=new_output
        )


class LibraryAgentResponse(pydantic.BaseModel):
    """Response schema for a list of library agents and pagination info."""

    agents: list[LibraryAgent]
    pagination: Pagination


class LibraryAgentPresetCreatable(pydantic.BaseModel):
    """
    Request model used when creating a new preset for a library agent.
    """

    graph_id: str
    graph_version: int

    inputs: block_model.BlockInput
    credentials: dict[str, CredentialsMetaInput]

    name: str
    description: str

    is_active: bool = True

    webhook_id: Optional[str] = None


class LibraryAgentPresetCreatableFromGraphExecution(pydantic.BaseModel):
    """
    Request model used when creating a new preset for a library agent.
    """

    graph_execution_id: str

    name: str
    description: str

    is_active: bool = True


class LibraryAgentPresetUpdatable(pydantic.BaseModel):
    """
    Request model used when updating a preset for a library agent.
    """

    inputs: Optional[block_model.BlockInput] = None
    credentials: Optional[dict[str, CredentialsMetaInput]] = None

    name: Optional[str] = None
    description: Optional[str] = None

    is_active: Optional[bool] = None


class TriggeredPresetSetupRequest(pydantic.BaseModel):
    name: str
    description: str = ""

    graph_id: str
    graph_version: int

    trigger_config: dict[str, Any]
    agent_credentials: dict[str, CredentialsMetaInput] = pydantic.Field(
        default_factory=dict
    )


class LibraryAgentPreset(LibraryAgentPresetCreatable):
    """Represents a preset configuration for a library agent."""

    id: str
    user_id: str
    created_at: datetime.datetime
    updated_at: datetime.datetime

    webhook: "Webhook | None"

    @classmethod
    def from_db(cls, preset: prisma.models.AgentPreset) -> "LibraryAgentPreset":
        from backend.data.integrations import Webhook

        if preset.InputPresets is None:
            raise ValueError("InputPresets must be included in AgentPreset query")
        if preset.webhookId and preset.Webhook is None:
            raise ValueError(
                "Webhook must be included in AgentPreset query when webhookId is set"
            )

        input_data: block_model.BlockInput = {}
        input_credentials: dict[str, CredentialsMetaInput] = {}

        for preset_input in preset.InputPresets:
            if not is_credentials_field_name(preset_input.name):
                input_data[preset_input.name] = preset_input.data
            else:
                input_credentials[preset_input.name] = (
                    CredentialsMetaInput.model_validate(preset_input.data)
                )

        return cls(
            id=preset.id,
            user_id=preset.userId,
            created_at=preset.createdAt,
            updated_at=preset.updatedAt,
            graph_id=preset.agentGraphId,
            graph_version=preset.agentGraphVersion,
            name=preset.name,
            description=preset.description,
            is_active=preset.isActive,
            inputs=input_data,
            credentials=input_credentials,
            webhook_id=preset.webhookId,
            webhook=Webhook.from_db(preset.Webhook) if preset.Webhook else None,
        )


class LibraryAgentPresetResponse(pydantic.BaseModel):
    """Response schema for a list of agent presets and pagination info."""

    presets: list[LibraryAgentPreset]
    pagination: Pagination


class LibraryAgentFilter(str, Enum):
    """Possible filters for searching library agents."""

    IS_FAVOURITE = "isFavourite"
    IS_CREATED_BY_USER = "isCreatedByUser"


class LibraryAgentSort(str, Enum):
    """Possible sort options for sorting library agents."""

    CREATED_AT = "createdAt"
    UPDATED_AT = "updatedAt"


class LibraryAgentUpdateRequest(pydantic.BaseModel):
    """
    Schema for updating a library agent via PUT.

    Includes flags for auto-updating version, marking as favorite,
    archiving, or deleting.
    """

    auto_update_version: Optional[bool] = pydantic.Field(
        default=None, description="Auto-update the agent version"
    )
    is_favorite: Optional[bool] = pydantic.Field(
        default=None, description="Mark the agent as a favorite"
    )
    is_archived: Optional[bool] = pydantic.Field(
        default=None, description="Archive the agent"
    )
