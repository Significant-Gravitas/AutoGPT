# Import necessary libraries and classes
from __future__ import annotations

import uuid
from typing import Optional

from AFAAS.interfaces.configuration import SystemConfiguration, SystemSettings
from AFAAS.interfaces.db import MemorySettings
from AFAAS.core.plugin.simple import PluginLocation
from AFAAS.interfaces.adapters import OpenAISettings
from pydantic import BaseModel, Field


# Define the Agent Systems Class
# This class defines the systems that will be used by your custom agent.
class MyCustomAgentSystems(SystemConfiguration):
    # Define the plugin locations for each system your agent will use.
    memory: PluginLocation  # Plugin location for the memory system
    openai_provider: PluginLocation  # Plugin location for the OpenAI provider system
    workspace: PluginLocation
    # ... add other systems as needed

    class Config(SystemConfiguration.Config):
        # Specify any additional configuration options for this class.
        extra = "allow"


# Define the Agent Configuration Class
# This class holds configurations specific to your custom agent.
class MyCustomAgentConfiguration(SystemConfiguration):
    # Define configurations like cycle count, max task cycle count, etc.
    cycle_count: int
    max_task_cycle_count: int
    creation_time: str
    systems: MyCustomAgentSystems  # Reference to the Agent Systems class
    # ... add other configurations as needed

    class Config(SystemConfiguration.Config):
        # Specify any additional configuration options for this class.
        extra = "allow"


# Define the Agent System Settings Class
# This class holds settings for your custom agent's systems.
class MyCustomAgentSystemSettings(SystemSettings):
    # Hold a reference to the Agent Configuration class.
    configuration: MyCustomAgentConfiguration

    class Config(SystemSettings.Config):
        # Specify any additional configuration options for this class.
        extra = "allow"


# Define the Agent Settings Class
# This class aggregates all settings required for initializing your custom agent.
class MyCustomAgentSettings(BaseModel):
    # Hold references to the Agent System Settings class,
    # and settings for memory, OpenAI provider, etc.
    agent: MyCustomAgentSystemSettings
    memory: MemorySettings
    openai_provider: OpenAISettings  # Settings for the OpenAI provider
    # ... add other settings as needed

    class Config(BaseModel.Config):
        # Define any additional configuration options for this class.
        json_encoders = {uuid.UUID: lambda v: str(v)}  # Custom encoder for UUID4
        extra = "allow"
        default_exclude = {
            "agent",
            # ... add other fields to exclude during serialization
        }

    # Define methods for serialization, loading values, etc., as needed.
    # ...
