from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Iterator

from autogpt.agents.agent import Agent, AgentSettings
from autogpt.app.config import ConfigBuilder
from autogpt_server.data.block import Block, BlockOutput, BlockSchema
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
from forge.models.json_schema import JSONSchema
from pydantic import Field

if TYPE_CHECKING:
    from autogpt.app.config import AppConfig

logger = logging.getLogger(__name__)


class BlockAgentSettings(AgentSettings):
    disabled_components: list[str] = Field(default_factory=list)


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
            if type(attr_value).__name__ in settings.disabled_components:
                delattr(self, attr_name)


class AutoGPTAgentBlock(Block):
    class Input(BlockSchema):
        task: str
        input: str
        disabled_components: list[str] = Field(default_factory=list)
        disabled_commands: list[str] = Field(default_factory=list)
        fast_mode: bool = False
    
    class Output(BlockSchema):
        result: str

    def __init__(self):
        super().__init__(
            id="d2e2ecd2-9ae6-422d-8dfe-ceca500ce6a6",
            input_schema=AutoGPTAgentBlock.Input,
            output_schema=AutoGPTAgentBlock.Output,
        )

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

        # LLM provider
        multi_provider = MultiProvider()
        for model in [config.smart_llm, config.fast_llm]:
            # Ensure model providers for configured LLMs are available
            multi_provider.get_model_provider(model)

        # State
        state = BlockAgentSettings(
            agent_id="TemporaryAgentID",
            name="WrappedAgent",
            description="Wrapped agent for the Agent Server.",
            task=f"Your task: {input_data.task}\n"
                 f"Input data: {input_data.input}",
            disabled_components=input_data.disabled_components,
        )
        # Switch big brain mode
        state.config.big_brain = not input_data.fast_mode

        agent = BlockAgent(state, multi_provider, file_storage, config)

        # Execute agent
        for tries in range(3):
            try:
                proposal = asyncio.run(agent.propose_action())
                break
            except Exception as e:
                if tries == 2:
                    raise e

        result = asyncio.run(agent.execute(proposal))

        yield "result", str(result)
