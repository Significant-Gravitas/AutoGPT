from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional
from pydantic import Field


if TYPE_CHECKING :
    from . import PlannerAgent

from ..base import PromptManager

from autogpts.autogpt.autogpt.core.tools import SimpleToolRegistry
from autogpts.autogpt.autogpt.core.agents.base.models import (
    BaseAgentConfiguration,
    BaseAgentSystems,
)
from autogpts.autogpt.autogpt.core.plugin.simple import PluginLocation
from autogpts.autogpt.autogpt.core.resource.model_providers import OpenAISettings


class PlannerAgentSystems(BaseAgentSystems):
    tool_registry : str ="autogpt.core.tools.SimpleToolRegistry"
    chat_model_provider : str ="autogpt.core.resource.model_providers.OpenAIProvider"
    prompt_manager : str ="autogpt.core.agents.base.PromptManager"

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

    systems=PlannerAgentSystems()

    class Config(BaseAgentConfiguration.Config):
        pass


# class PlannerAgent.SystemSettings(BaseAgent.SystemSettings):

#     chat_model_provider: OpenAISettings = OpenAISettings()
#     tool_registry: SimpleToolRegistry.SystemSettings = SimpleToolRegistry.SystemSettings()
#     prompt_manager: PromptManager.SystemSettings = PromptManager.SystemSettings()

#     user_id: Optional[str]

#     agent_name: str = Field(default="New Agent")
#     agent_role: Optional[str] = Field(default=None)
#     agent_goals: Optional[list] 
#     agent_goal_sentence: Optional[list] 
#     agent_class: str = Field(default="autogpt.core.agents.simple.main.PlannerAgent")

#     class Config(BaseAgent.SystemSettings.Config):
#         pass
