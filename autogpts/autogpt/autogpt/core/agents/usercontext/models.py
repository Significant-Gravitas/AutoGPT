from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import Field

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


class UserContextAgentSystems(BaseAgentSystems):
    ability_registry: PluginLocation
    chat_model_provider: PluginLocation
    planning: PluginLocation

    class Config(BaseAgentSystems.Config):
        pass


class UserContextAgentConfiguration(BaseAgentConfiguration):
    systems: UserContextAgentSystems
    agent_name: str = Field(default="New Agent")

    class Config(BaseAgentConfiguration.Config):
        pass


class UserContextAgentSystemSettings(BaseAgentSystemSettings):
    configuration: UserContextAgentConfiguration
    user_id: Optional[uuid.UUID] = Field(default=None)
    agent_id: Optional[uuid.UUID] = Field(default=None)

    class Config(BaseAgentSystemSettings.Config):
        pass


class UserContextAgentSettings(BaseAgentSettings):
    agent: UserContextAgentSystemSettings
    chat_model_provider: OpenAISettings
    planning: SimplePlannerSettings
    user_id: Optional[uuid.UUID] = Field(default=None)
    agent_id: Optional[uuid.UUID] = Field(default=None)
    agent_name: str = Field(default="New Agent")
    parent_agent_id: uuid.UUID = Field(default=None)
    agent_class: str = Field(default="UserContextAgent")
    _type_: str = "autogpt.core.agents.usercontext.agent.UserContextAgent"

    class Config(BaseAgentSettings.Config):
        pass
