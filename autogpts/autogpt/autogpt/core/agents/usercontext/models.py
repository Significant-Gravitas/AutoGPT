from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import Field

from autogpt.core.agents.base.models import (
    BaseAgentConfiguration,
    BaseAgentSettings,
    BaseAgentSystems,
)
from autogpt.core.agents.simple.lib import PromptManagerSettings
from autogpt.core.plugin.simple import PluginLocation
from autogpt.core.resource.model_providers import OpenAISettings

if TYPE_CHECKING:
    pass


class UserContextAgentSystems(BaseAgentSystems):

    ability_registry : str  ="autogpt.core.ability.SimpleToolRegistry"
    chat_model_provider : str  = "autogpt.core.resource.model_providers.OpenAIProvider"
    planning : str ="autogpt.core.agents.simple.lib.PromptManager"
    
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


class UserContextAgentSettings(BaseAgentSettings):
    # agent: UserContextAgentSystemSettings = UserContextAgentSystemSettings()
    chat_model_provider: OpenAISettings = OpenAISettings()
    planning: PromptManagerSettings =PromptManagerSettings()
    user_id: Optional[uuid.UUID] = Field(default=None)
    agent_id: Optional[uuid.UUID] = Field(default=None)
    agent_name: str = Field(default="New Agent")
    parent_agent_id: uuid.UUID = Field(default=None)
    agent_class: str = Field(default="UserContextAgent")
    _type_: str = "autogpt.core.agents.usercontext.agent.UserContextAgent"

    class Config(BaseAgentSettings.Config):
        pass
