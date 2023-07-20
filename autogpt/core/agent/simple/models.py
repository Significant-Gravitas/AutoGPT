
from pydantic import BaseModel

from autogpt.core.ability import (
    AbilityRegistrySettings,
)
from autogpt.core.configuration import SystemConfiguration, SystemSettings
from autogpt.core.memory import MemorySettings
from autogpt.core.planning import PlannerSettings
from autogpt.core.plugin.simple import (
    PluginLocation,
)
from autogpt.core.resource.model_providers import OpenAISettings
from autogpt.core.workspace.simple import WorkspaceSettings


class AgentSystems(SystemConfiguration):
    ability_registry: PluginLocation
    memory: PluginLocation
    openai_provider: PluginLocation
    planning: PluginLocation
    workspace: PluginLocation


class AgentConfiguration(SystemConfiguration):
    cycle_count: int
    max_task_cycle_count: int
    creation_time: str
    name: str
    role: str
    goals: list[str]
    systems: AgentSystems


class AgentSystemSettings(SystemSettings):
    configuration: AgentConfiguration


class AgentSettings(BaseModel):
    agent: AgentSystemSettings
    ability_registry: AbilityRegistrySettings
    memory: MemorySettings
    openai_provider: OpenAISettings
    planning: PlannerSettings
    workspace: WorkspaceSettings

    def update_agent_name_and_goals(self, agent_goals: dict) -> None:
        self.agent.configuration.name = agent_goals["agent_name"]
        self.agent.configuration.role = agent_goals["agent_role"]
        self.agent.configuration.goals = agent_goals["agent_goals"]
