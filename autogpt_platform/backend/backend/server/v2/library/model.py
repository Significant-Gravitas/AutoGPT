import datetime
import enum
import json
from typing import Any, Dict, List, Tuple, Union

import prisma.enums
import prisma.models
import pydantic

import backend.data.block
import backend.data.graph
import backend.server.model


class AgentStatus(str, enum.Enum):
    """Enumeration for various statuses an agent can have."""

    COMPLETED = "COMPLETED"  # All runs completed
    HEALTHY = "HEALTHY"  # Agent is running (not all runs have completed)
    WAITING = "WAITING"  # Agent is queued or waiting to start
    ERROR = "ERROR"  # Agent is in an error state


def _calculate_agent_status(
    executions: List[prisma.models.AgentGraphExecution],
    recent_threshold: datetime.datetime,
) -> Tuple[AgentStatus, bool]:
    """
    Helper function to determine the overall agent status and whether there
    is new output (i.e., completed runs within the recent threshold).

    :param executions: A list of AgentGraphExecution objects.
    :param recent_threshold: A datetime; any execution after this indicates new output.
    :return: (AgentStatus, new_output_flag)
    """

    if not executions:
        return AgentStatus.COMPLETED, False

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
        return AgentStatus.ERROR, new_output
    elif status_counts[prisma.enums.AgentExecutionStatus.QUEUED] > 0:
        return AgentStatus.WAITING, new_output
    elif status_counts[prisma.enums.AgentExecutionStatus.RUNNING] > 0:
        return AgentStatus.HEALTHY, new_output
    else:
        return AgentStatus.COMPLETED, new_output


class LibraryAgent(pydantic.BaseModel):
    """
    Represents an agent in the library, including metadata for display and
    user interaction within the system.
    """

    id: str
    agent_id: str
    agent_version: int

    image_url: str

    creator_name: str
    creator_image_url: str

    status: AgentStatus

    updated_at: datetime.datetime

    name: str
    description: str

    # The schema of the input for this agent (matches GraphMeta input schema)
    input_schema: Dict[str, Any]

    # Indicates whether there's a new output (based on recent runs)
    new_output: bool

    # Whether the user can access the underlying graph
    can_access_graph: bool

    # Indicates if this agent is the latest version
    is_latest_version: bool

    class Config:
        """Pydantic model configuration."""

        orm_mode = True
        allow_population_by_field_name = True
        # Any additional config options you need

    @staticmethod
    def from_db(agent: prisma.models.LibraryAgent) -> "LibraryAgent":
        """
        Factory method that constructs a LibraryAgent from a Prisma LibraryAgent
        model instance.
        """
        if not agent.Agent:
            raise ValueError("Associated Agent record is required.")

        graph = backend.data.graph.GraphModel.from_db(agent.Agent)
        agent_updated_at = agent.Agent.updatedAt
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
        executions = agent.Agent.AgentGraphExecution or []
        status, new_output = _calculate_agent_status(executions, week_ago)

        # Check if user can access the graph
        can_access_graph = agent.Agent.userId == agent.userId

        # Hard-coded to True until a method to check is implemented
        is_latest_version = True

        return LibraryAgent(
            id=agent.id,
            agent_id=agent.agentId,
            agent_version=agent.agentVersion,
            image_url=agent.image_url or "",
            creator_name=creator_name,
            creator_image_url=creator_image_url,
            status=status,
            updated_at=updated_at,
            name=graph.name,
            description=graph.description,
            input_schema=graph.input_schema,
            new_output=new_output,
            can_access_graph=can_access_graph,
            is_latest_version=is_latest_version,
        )


class LibraryAgentResponse(pydantic.BaseModel):
    """Response schema for a list of library agents and pagination info."""

    agents: List[LibraryAgent]
    pagination: backend.server.model.Pagination


class LibraryAgentPreset(pydantic.BaseModel):
    """Represents a preset configuration for a library agent."""

    id: str
    updated_at: datetime.datetime

    agent_id: str
    agent_version: int

    name: str
    description: str

    is_active: bool
    inputs: Dict[str, Union[backend.data.block.BlockInput, Any]]

    @classmethod
    def from_db(cls, preset: prisma.models.AgentPreset) -> "LibraryAgentPreset":
        """Constructs a LibraryAgentPreset from a Prisma AgentPreset model."""
        input_data: Dict[str, Any] = {}
        for data in preset.InputPresets or []:
            input_data[data.name] = json.loads(data.data)

        return cls(
            id=preset.id,
            updated_at=preset.updatedAt,
            agent_id=preset.agentId,
            agent_version=preset.agentVersion,
            name=preset.name,
            description=preset.description,
            is_active=preset.isActive,
            inputs=input_data,
        )


class LibraryAgentPresetResponse(pydantic.BaseModel):
    """Response schema for a list of agent presets and pagination info."""

    presets: List[LibraryAgentPreset]
    pagination: backend.server.model.Pagination


class CreateLibraryAgentPresetRequest(pydantic.BaseModel):
    """
    Request model used when creating a new preset for a library agent.
    """

    name: str
    description: str
    inputs: Dict[str, Union[backend.data.block.BlockInput, Any]]
    agent_id: str
    agent_version: int
    is_active: bool


class LibraryAgentFilter(str, enum.Enum):
    """Possible filters for searching library agents."""

    IS_FAVOURITE = "isFavourite"
    IS_CREATED_BY_USER = "isCreatedByUser"


class LibraryAgentSort(str, enum.Enum):
    """Possible sort options for sorting library agents."""

    CREATED_AT = "createdAt"
    UPDATED_AT = "updatedAt"


class LibraryAgentUpdateRequest(pydantic.BaseModel):
    """
    Schema for updating a library agent via PUT.

    Includes flags for auto-updating version, marking as favorite,
    archiving, or deleting.
    """

    auto_update_version: bool = pydantic.Field(
        False, description="Auto-update the agent version"
    )
    is_favorite: bool = pydantic.Field(
        False, description="Mark the agent as a favorite"
    )
    is_archived: bool = pydantic.Field(False, description="Archive the agent")
    is_deleted: bool = pydantic.Field(False, description="Delete the agent")
