import logging
from pathlib import Path

from pydantic import BaseModel

from autogpt.core.agent.base import Agent
from autogpt.core.command import CommandRegistrySettings, SimpleCommandRegistry
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory import MemorySettings, SimpleMemory
from autogpt.core.model import (
    EmbeddingModelSettings,
    LanguageModelSettings,
    SimpleEmbeddingModel,
    SimpleLanguageModel,
)
from autogpt.core.planning.simple import PlannerSettings, SimplePlanner
from autogpt.core.plugin.simple import (
    PluginLocation,
    PluginStorageFormat,
    SimplePluginService,
)
from autogpt.core.resource import ResourceManagerSettings, SimpleResourceManager
from autogpt.core.resource.model_providers import OpenAIProvider, OpenAISettings
from autogpt.core.workspace.simple import SimpleWorkspace, WorkspaceSettings


class AgentSystems(SystemConfiguration):
    command_registry: PluginLocation
    memory: PluginLocation
    embedding_model: PluginLocation
    language_model: PluginLocation
    openai_provider: PluginLocation
    resource_manager: PluginLocation
    planning: PluginLocation
    workspace: PluginLocation


class AgentConfiguration(SystemConfiguration):
    systems: AgentSystems


class AgentSystemSettings(SystemSettings):
    configuration: AgentConfiguration


class AgentSettings(BaseModel):
    agent: AgentSystemSettings
    command_registry: CommandRegistrySettings
    memory: MemorySettings
    embedding_model: EmbeddingModelSettings
    language_model: LanguageModelSettings
    openai_provider: OpenAISettings
    resource_manager: ResourceManagerSettings
    planning: PlannerSettings
    workspace: WorkspaceSettings


class SimpleAgent(Agent, Configurable):
    defaults = AgentSystemSettings(
        name="simple_agent",
        description="A simple agent.",
        configuration=AgentConfiguration(
            systems=AgentSystems(
                command_registry=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.command.SimpleCommandRegistry",
                ),
                memory=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.memory.SimpleMemory",
                ),
                embedding_model=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.model.SimpleEmbeddingModel",
                ),
                language_model=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.model.SimpleLanguageModel",
                ),
                openai_provider=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.resource.model_providers.OpenAIProvider",
                ),
                resource_manager=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.resource.SimpleResourceManager",
                ),
                planning=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.planning.SimplePlanner",
                ),
                workspace=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.workspace.SimpleWorkspace",
                ),
            ),
        ),
    )

    def __init__(
        self,
        configuration: AgentConfiguration,
        logger: logging.Logger,
        command_registry: SimpleCommandRegistry,
        memory: SimpleMemory,
        embedding_model: SimpleEmbeddingModel,
        language_model: SimpleLanguageModel,
        openai_provider: OpenAIProvider,
        resource_manager: SimpleResourceManager,
        planning: SimplePlanner,
        workspace: SimpleWorkspace,
    ):
        self._configuration = configuration
        self._logger = logger
        self._command_registry = command_registry
        self._memory = memory
        self._embedding_model = embedding_model
        self._language_model = language_model
        # FIXME: Need some work to make this work as a dict of providers
        #  Getting the construction of the config to work is a bit tricky
        self._openai_provider = openai_provider
        self._resource_manager = resource_manager
        self._planning = planning
        self._workspace = workspace

    def step(self, *args, **kwargs):
        pass

    def run(self):
        pass

    @staticmethod
    def load_system(
        system_name: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
        **kwargs,
    ):
        logger.debug(f"Loading system {system_name}")
        system_load_config = agent_settings.system[system_name]
        system_class = SimplePluginService.get_plugin(system_load_config)

        system_logger = logger.getChild(system_name)
        system_config = getattr(agent_settings, system_name)
        return system_class(system_config, system_logger, **kwargs)

    def __repr__(self):
        return "SimpleAgent()"

    ###############################################################
    # Factory interface for agent boostrapping and initialization #
    ###############################################################

    @classmethod
    def compile_settings(
        cls, logger: logging.Logger, user_configuration: dict
    ) -> AgentSettings:
        """Compile the user's configuration with the defaults."""
        logger.debug("Compiling agent settings")
        logger.debug("Processing agent system configuration.")
        configuration_dict = {
            "agent": cls.process_user_configuration(
                user_configuration.get("agent", {})
            ).dict(),
        }

        system_locations = configuration_dict["agent"]["configuration"]["systems"]

        # Build up default configuration
        for system_name, system_location in system_locations.items():
            system_class = SimplePluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.process_user_configuration(
                user_configuration.get(system_name, {})
            ).dict()

        return AgentSettings.parse_obj(configuration_dict)

    @classmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        logger: logging.Logger,
    ) -> "SimpleAgent":
        agent_settings: AgentSettings = SimpleWorkspace.load_agent_settings(
            workspace_path,
        )

        resource_manager: SimpleResourceManager = SimpleAgent.load_system(
            "budget_manager",
            agent_settings,
            logger,
        )
        # command_registry: SimpleCommandRegistry = SimpleAgent.load_system(
        #     "command_registry",
        #     configuration,
        #     logger,
        #     credentials=credentials,
        # )
        # embedding_model: OpenAIEmbeddingModel = SimpleAgent.load_system(
        #     "embedding_model",
        #     configuration,
        #     logger,
        #     credentials=credentials,
        # )
        # language_model: OpenAILanguageModel = SimpleAgent.load_system(
        #     "language_model",
        #     configuration,
        #     logger,
        # )
        # memory_backend: SimpleMemoryBackend = SimpleAgent.load_system(
        #     "memory_backend",
        #     configuration,
        #     logger,
        # )
        planner: SimplePlanner = SimpleAgent.load_system(
            "planner",
            agent_settings,
            logger,
        )
        workspace: SimpleWorkspace = SimpleAgent.load_system(
            "workspace",
            agent_settings,
            logger,
        )
        return SimpleAgent(
            agent_settings=agent_settings,
            logger=logger,
            resource_manager=resource_manager,
            # language_model=language_model,
            planner=planner,
            workspace=workspace,
            # command_registry=command_registry,
            # embedding_model=embedding_model,
            # memory_backend=memory_backend,
        )
