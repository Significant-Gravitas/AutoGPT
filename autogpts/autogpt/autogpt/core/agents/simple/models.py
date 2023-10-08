from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional
from pydantic import Field

from autogpt.core.tools import SimpleToolRegistry
from autogpt.core.agents.base.models import (
    BaseAgentConfiguration,
    BaseAgentSettings,
    BaseAgentSystems,
    BaseAgentSystemSettings,
)
from autogpt.core.agents.simple.lib import SimplePlannerSettings
from autogpt.core.plugin.simple import PluginLocation
from autogpt.core.resource.model_providers import OpenAISettings


if TYPE_CHECKING :
    from autogpt.core.agents.simple import PlannerAgent

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


# class PlannerAgentSystemSettings(BaseAgentSystemSettings):
#     name: str ="simple_agent"
#     description: str ="A simple agent."
#     configuration : PlannerAgentConfiguration = PlannerAgentConfiguration()


#     class Config(BaseAgentSystemSettings.Config):
#         pass


class PlannerAgentSettings(BaseAgentSettings):

    #agent: PlannerAgent.SystemSettings =  PlannerAgent.SystemSettings()
    agent_class: str = Field(default="autogpt.core.agents.simple.main.PlannerAgent")
    
    chat_model_provider: OpenAISettings = OpenAISettings()
    tool_registry: SimpleToolRegistry.SystemSettings = SimpleToolRegistry.SystemSettings()
    planning: SimplePlannerSettings = SimplePlannerSettings()
    user_id: Optional[uuid.UUID] = Field(default=None)
    agent_id: Optional[uuid.UUID] = Field(default=None)
    agent_name: str = Field(default="New Agent")
    agent_role: Optional[str] = Field(default=None)
    agent_goals: Optional[list] = Field(default=None)
    agent_goal_sentence: Optional[list] = Field(default=None)

    class Config(BaseAgentSettings.Config):
        pass
