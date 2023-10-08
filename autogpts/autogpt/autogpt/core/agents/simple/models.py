from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional
from pydantic import Field

from autogpt.core.tools import ToolsRegistrySettings
from autogpt.core.agents.base.models import (
    BaseAgentConfiguration,
    BaseAgentSettings,
    BaseAgentSystems,
    BaseAgentSystemSettings,
)
from autogpt.core.agents.simple.lib import SimplePlannerSettings
from autogpt.core.plugin.simple import PluginLocation
from autogpt.core.resource.model_providers import OpenAISettings

if TYPE_CHECKING:
    pass


class PlannerAgentSystems(BaseAgentSystems):
    tool_registry : str ="autogpt.core.tools.SimpleToolRegistry"
    chat_model_provider : str ="autogpt.core.resource.model_providers.OpenAIProvider"
    planning : str ="autogpt.core.agents.simple.lib.SimplePlanner"

    class Config(BaseAgentSystems.Config):
        pass



class PlannerAgentConfiguration(BaseAgentConfiguration):
    systems: PlannerAgentSystems
    agent_name: str = Field(default="New Agent")
    agent_role: Optional[str] = Field(default=None)
    agent_goals: Optional[list[str]] = Field(default=None)
    agent_goal_sentence: Optional[str] = Field(default=None)
    cycle_count : int =0 
    max_task_cycle_count : int =3
    creation_time : str=""
    systems=PlannerAgentSystems()

    class Config(BaseAgentConfiguration.Config):
        pass


class PlannerAgentSystemSettings(BaseAgentSystemSettings):
    name: str ="simple_agent"
    description: str ="A simple agent."
    configuration : PlannerAgentConfiguration = PlannerAgentConfiguration(
        # agent_name="Entrepreneur-GPT",
        # agent_role=(
        #     "An AI designed to autonomously develop and run businesses with "
        #     "the sole goal of increasing your net worth."
        # ),
        # agent_goals=[
        #     "Increase net worth",
        #     "Grow Twitter Account",
        #     "Develop and manage multiple businesses autonomously",
        # ],
        # agent_goal_sentence="""Increase net worth
        #     and Grow Twitter Account
        #     and Develop and manage multiple businesses autonomously""",
        # cycle_count=0,
        # max_task_cycle_count=3,
        # creation_time="",
        # systems=PlannerAgentSystems(
        #     # tool_registry=PluginLocation(
        #     #     storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
        #     #     storage_route="autogpt.core.tools.SimpleToolRegistry",
        #     # ),
        #     # memory=PluginLocation(
        #     #     storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
        #     #     storage_route="autogpt.core.memory.base.Memory",
        #     # ),
        #     # chat_model_provider=PluginLocation(
        #     #     storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
        #     #     storage_route="autogpt.core.resource.model_providers.OpenAIProvider",
        #     # ),
        #     # planning=PluginLocation(
        #     #     storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
        #     #     storage_route="autogpt.core.agents.simple.lib.SimplePlanner",
        #     # ),
        #     # workspace=PluginLocation(
        #     #     storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
        #     #     storage_route="autogpt.core.workspace.SimpleWorkspace",
        #     # ),
        # ),
    )


    class Config(BaseAgentSystemSettings.Config):
        pass


class PlannerAgentSettings(BaseAgentSettings):
    agent: PlannerAgentSystemSettings
    chat_model_provider: OpenAISettings
    tool_registry: ToolsRegistrySettings
    planning: SimplePlannerSettings
    user_id: Optional[uuid.UUID] = Field(default=None)
    agent_id: Optional[uuid.UUID] = Field(default=None)
    agent_name: str = Field(default="New Agent")
    agent_role: Optional[str] = Field(default=None)
    agent_goals: Optional[list] = Field(default=None)
    agent_goal_sentence: Optional[list] = Field(default=None)
    agent_class: str = Field(default="autogpt.core.agents.simple.main.PlannerAgent")

    class Config(BaseAgentSettings.Config):
        pass
