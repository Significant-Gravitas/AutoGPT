from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from autogpt.agents.agent import Agent, AgentSettings
from autogpt.app.config import ConfigBuilder
from autogpt_server.data.block import Block, BlockFieldSecret, BlockOutput, BlockSchema
from forge.agent.components import AgentComponent
from forge.agent.protocols import (
    CommandProvider,
)
from forge.command import command
from forge.command.command import Command
from forge.file_storage import FileStorageBackendName, get_storage
from forge.file_storage.base import FileStorage
from forge.llm.providers import (
    MultiProvider,
)
from forge.llm.providers.openai import OpenAICredentials, OpenAIProvider
from forge.llm.providers.schema import ModelProviderName
from forge.models.json_schema import JSONSchema
from pydantic import Field, SecretStr

if TYPE_CHECKING:
    from autogpt.app.config import AppConfig

logger = logging.getLogger(__name__)


class BlockAgentSettings(AgentSettings):
    enabled_components: list[str] = Field(default_factory=list)


class OutputComponent(CommandProvider):
    def get_commands(self) -> Iterator[Command]:
        yield self.output
    
    @command(
        parameters={
            "output": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="Output data to be returned.",
                required=True,
            ),
        },
    )
    def output(self, output: str) -> str:
        """Use this to output the result."""
        return output


class BlockAgent(Agent):
    def __init__(
        self,
        settings: BlockAgentSettings,
        llm_provider: MultiProvider,
        file_storage: FileStorage,
        app_config: AppConfig,
    ):
        super().__init__(settings, llm_provider, file_storage, app_config)

        self.output = OutputComponent()

        # Disable components
        for attr_name in list(self.__dict__.keys()):
            attr_value = getattr(self, attr_name)
            if not isinstance(attr_value, AgentComponent):
                continue
            component_name = type(attr_value).__name__
            if component_name != "SystemComponent" and component_name not in settings.enabled_components:
                delattr(self, attr_name)


class AutoGPTAgentBlock(Block):
    class Input(BlockSchema):
        task: str
        input: str
        openai_api_key: BlockFieldSecret = BlockFieldSecret(key="openai_api_key")
        enabled_components: list[str] = Field(default_factory=lambda: [OutputComponent.__name__])
        disabled_commands: list[str] = Field(default_factory=list)
        fast_mode: bool = False
    
    class Output(BlockSchema):
        result: str

    def __init__(self):
        super().__init__(
            id="d2e2ecd2-9ae6-422d-8dfe-ceca500ce6a6",
            input_schema=AutoGPTAgentBlock.Input,
            output_schema=AutoGPTAgentBlock.Output,
            test_input={
                "task": "Make calculations and use output command to output the result.",
                "input": "5 + 3",
                "openai_api_key": "openai_api_key",
                "enabled_components": [OutputComponent.__name__],
                "disabled_commands": ["finish"],
                "fast_mode": True,
            },
            test_output=[
                ("result", "8"),
            ],
            test_mock={
                "get_provider": lambda _: MultiProvider(),
                "get_result": lambda _: "8",
            }
        )

    @staticmethod
    def get_provider(openai_api_key: str) -> MultiProvider:
        # LLM provider
        settings = OpenAIProvider.default_settings.model_copy()
        settings.credentials = OpenAICredentials(api_key=SecretStr(openai_api_key))
        openai_provider = OpenAIProvider(settings=settings)
        
        multi_provider = MultiProvider()
        # HACK: Add OpenAI provider to the multi provider with api key
        multi_provider._provider_instances[ModelProviderName.OPENAI] = openai_provider

        return multi_provider

    @staticmethod
    def get_result(agent: BlockAgent) -> str:
        # Execute agent
        for tries in range(3):
            try:
                proposal = asyncio.run(agent.propose_action())
                break
            except Exception as e:
                if tries == 2:
                    raise e

        result = asyncio.run(agent.execute(proposal))

        return str(result)

    def run(self, input_data: Input) -> BlockOutput:
        # Set up configuration
        config = ConfigBuilder.build_config_from_env()
        # Disable commands
        config.disabled_commands.extend(input_data.disabled_commands)

        # Storage
        local = config.file_storage_backend == FileStorageBackendName.LOCAL
        restrict_to_root = not local or config.restrict_to_workspace
        file_storage = get_storage(
            config.file_storage_backend,
            root_path=Path("data"),
            restrict_to_root=restrict_to_root,
        )
        file_storage.initialize()

        # State
        state = BlockAgentSettings(
            agent_id="TemporaryAgentID",
            name="WrappedAgent",
            description="Wrapped agent for the Agent Server.",
            task=f"Your task: {input_data.task}\n"
                 f"Input data: {input_data.input}",
            enabled_components=input_data.enabled_components,
        )
        # Switch big brain mode
        state.config.big_brain = not input_data.fast_mode
        provider = self.get_provider(input_data.openai_api_key.get())

        agent = BlockAgent(state, provider, file_storage, config)

        result = self.get_result(agent)

        yield "result", result
