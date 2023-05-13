import logging
from pathlib import Path

from autogpt.core.agent.base import Agent
from autogpt.core.budget.simple import SimpleBudgetManager
from autogpt.core.configuration import Configuration

# from autogpt.core.command.simple import SimpleCommandRegistry
from autogpt.core.credentials.simple import SimpleCredentialsManager

# from autogpt.core.memory.simple import SimpleMemoryBackend
from autogpt.core.model.language.openai import OpenAILanguageModel

# from autogpt.core.model.embedding.openai import OpenAIEmbeddingModel
from autogpt.core.planning.simple import SimplePlanner
from autogpt.core.plugin.simple import SimplePluginService
from autogpt.core.workspace.simple import SimpleWorkspace


class SimpleAgent(Agent):
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        logger: logging.Logger,
    ) -> "SimpleAgent":
        configuration: Configuration = SimpleWorkspace.load_configuration(
            workspace_path,
        )

        credentials: SimpleCredentialsManager = SimpleAgent.load_system(
            "credentials",
            configuration,
            logger,
        )

        budget_manager: SimpleBudgetManager = SimpleAgent.load_system(
            "budget_manager",
            configuration,
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
        language_model: OpenAILanguageModel = SimpleAgent.load_system(
            "language_model",
            configuration,
            logger,
            credentials=credentials,
        )
        # memory_backend: SimpleMemoryBackend = SimpleAgent.load_system(
        #     "memory_backend",
        #     configuration,
        #     logger,
        #     credentials=credentials,
        # )
        planner: SimplePlanner = SimpleAgent.load_system(
            "planner",
            configuration,
            logger,
        )
        workspace: SimpleWorkspace = SimpleAgent.load_system(
            "workspace",
            configuration,
            logger,
        )
        return SimpleAgent(
            configuration=configuration,
            logger=logger,
            credentials=credentials,
            budget_manager=budget_manager,
            language_model=language_model,
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
        configuration: Configuration,
        logger: logging.Logger,
        **kwargs,
    ):
        logger.debug(f"Loading system {system_name}")
        system_load_config = configuration.system[system_name]
        system_class = SimplePluginService.get_plugin(system_load_config)

        system_logger = logger.getChild(system_name)
        system_config = getattr(configuration, system_name)
        return system_class(system_config, system_logger, **kwargs)

    def __repr__(self):
        return "SimpleAgent()"
