import logging
from pathlib import Path

from autogpt.core.agent.base import Agent
from autogpt.core.configuration import AgentSettings

# from autogpt.core.model.embedding.openai import OpenAIEmbeddingModel
from autogpt.core.planning.simple import SimplePlanner
from autogpt.core.plugin.simple import SimplePluginService
from autogpt.core.resource import SimpleResourceManager
from autogpt.core.workspace.simple import SimpleWorkspace

# from autogpt.core.memory.simple import SimpleMemoryBackend


# from autogpt.core.command.simple import SimpleCommandRegistry


class SimpleAgent(Agent):
    def __init__(self, *args, **kwargs):
        pass

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
