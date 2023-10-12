from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import Field

from autogpt.core.agents.base.models import (
    BaseAgentConfiguration,
    BaseAgentSystems,
)
from autogpt.core.agents.simple.lib import PromptManager
from autogpt.core.plugin.simple import PluginLocation
from autogpt.core.resource.model_providers import OpenAISettings
from autogpt.core.agents.base import BaseAgent

if TYPE_CHECKING:
    pass


class UserContextAgentSystems(BaseAgentSystems):

    ability_registry : str  ="autogpt.core.ability.SimpleToolRegistry"
    chat_model_provider : str  = "autogpt.core.resource.model_providers.OpenAIProvider"
    prompt_manager : str ="autogpt.core.agents.simple.lib.PromptManager"
    
    class Config(BaseAgentSystems.Config):
        pass


class UserContextAgentConfiguration(BaseAgentConfiguration):
    systems: UserContextAgentSystems = UserContextAgentSystems()
    agent_name="UCC (User Context Checker)"

    class Config(BaseAgentConfiguration.Config):
        pass


# class UserContextAgentSystemSettings(BaseAgentSystemSettings):
#     configuration: UserContextAgentConfiguration = UserContextAgentConfiguration()
#     name="usercontext_agent"
#     description="An agent that improve the quality of input provided by users."
#     user_id: Optional[uuid.UUID] = Field(default=None)
#     agent_id: Optional[uuid.UUID] = Field(default=None)

#     class Config(BaseAgentSystemSettings.Config):
#         pass


class UserContextAgentSettings(BaseAgent.SystemSettings):

    chat_model_provider: OpenAISettings
    prompt_manager: PromptManager.SystemSettings

    user_id: uuid.UUID 
    parent_agent_id: str 

    agent_name: str = Field(default="UserHelperAgent")
    agent_class: str = Field(default="UserContextAgent")
    _type_: str = "autogpt.core.agents.usercontext.main.UserContextAgent"

    class Config(BaseAgent.SystemSettings.Config):
        pass
