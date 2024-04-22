from __future__ import annotations

import inspect
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import sentry_sdk
from pydantic import Field

from autogpt.agents.prompt_strategies.one_shot import OneShotAgentPromptStrategy
from autogpt.commands.execute_code import CodeExecutorComponent
from autogpt.commands.git_operations import GitOperationsComponent
from autogpt.commands.image_gen import ImageGeneratorComponent
from autogpt.commands.system import SystemComponent
from autogpt.commands.user_interaction import UserInteractionComponent
from autogpt.commands.web_search import WebSearchComponent
from autogpt.commands.web_selenium import WebSeleniumComponent
from autogpt.components.event_history import EventHistoryComponent
from autogpt.core.configuration import Configurable
from autogpt.core.prompting import ChatPrompt
from autogpt.core.resource.model_providers import ChatMessage, ChatModelProvider
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    ChatModelResponse,
)
from autogpt.core.runner.client_lib.logging.helpers import dump_prompt
from autogpt.file_storage.base import FileStorage
from autogpt.llm.providers.openai import get_openai_command_specs
from autogpt.logs.log_cycle import (
    CURRENT_CONTEXT_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.logs.utils import fmt_kwargs
from autogpt.models.action_history import (
    ActionErrorResult,
    ActionInterruptedByHuman,
    ActionResult,
    ActionSuccessResult,
)
from autogpt.models.command import Command, CommandOutput
from autogpt.utils.exceptions import (
    AgentException,
    AgentTerminated,
    CommandExecutionError,
    InvalidArgumentError,
    UnknownCommandError,
)

from .base import (
    BaseAgent,
    BaseAgentConfiguration,
    BaseAgentSettings,
    ThoughtProcessOutput,
)
from .features.agent_file_manager import FileManagerComponent
from .features.context import ContextComponent
from .features.watchdog import WatchdogComponent
from .protocols import (
    AfterExecute,
    AfterParse,
    CommandProvider,
    DirectiveProvider,
    MessageProvider,
)

if TYPE_CHECKING:
    from autogpt.config import Config

logger = logging.getLogger(__name__)


class AgentConfiguration(BaseAgentConfiguration):
    pass


class AgentSettings(BaseAgentSettings):
    config: AgentConfiguration = Field(default_factory=AgentConfiguration)


class Agent(BaseAgent, Configurable[AgentSettings]):
    default_settings: AgentSettings = AgentSettings(
        name="Agent",
        description=__doc__ if __doc__ else "",
    )

    def __init__(
        self,
        settings: AgentSettings,
        llm_provider: ChatModelProvider,
        file_storage: FileStorage,
        legacy_config: Config,
    ):
        super().__init__(settings)

        self.llm_provider = llm_provider
        self.ai_profile = settings.ai_profile
        self.directives = settings.directives
        prompt_config = OneShotAgentPromptStrategy.default_configuration.copy(deep=True)
        prompt_config.use_functions_api = settings.config.use_functions_api
        self.prompt_strategy = OneShotAgentPromptStrategy(prompt_config, logger)
        self.commands: list[Command] = []

        # Components
        self.system = SystemComponent(legacy_config, settings.ai_profile)
        self.history = EventHistoryComponent(
            settings.history,
            self.send_token_limit,
            lambda x: self.llm_provider.count_tokens(x, self.llm.name),
            legacy_config,
            llm_provider,
        )
        self.user_interaction = UserInteractionComponent(legacy_config)
        self.file_manager = FileManagerComponent(settings, file_storage)
        self.code_executor = CodeExecutorComponent(
            self.file_manager.workspace,
            settings,
            legacy_config,
        )
        self.git_ops = GitOperationsComponent(legacy_config)
        self.image_gen = ImageGeneratorComponent(
            self.file_manager.workspace, legacy_config
        )
        self.web_search = WebSearchComponent(legacy_config)
        self.web_selenium = WebSeleniumComponent(legacy_config, llm_provider, self.llm)
        self.context = ContextComponent(self.file_manager.workspace)
        self.watchdog = WatchdogComponent(settings.config, settings.history)

        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        """Timestamp the agent was created; only used for structured debug logging."""

        self.log_cycle_handler = LogCycleHandler()
        """LogCycleHandler for structured debug logging."""

        self.event_history = settings.history
        self.legacy_config = legacy_config

    async def propose_action(self) -> ThoughtProcessOutput:
        """Proposes the next action to execute, based on the task and current state.

        Returns:
            The command name and arguments, if any, and the agent's thoughts.
        """
        self.reset_trace()

        # Get directives
        resources = await self.run_pipeline(DirectiveProvider.get_resources)
        constraints = await self.run_pipeline(DirectiveProvider.get_constraints)
        best_practices = await self.run_pipeline(DirectiveProvider.get_best_practices)

        directives = self.state.directives.copy(deep=True)
        directives.resources += resources
        directives.constraints += constraints
        directives.best_practices += best_practices

        # Get commands
        self.commands = await self.run_pipeline(CommandProvider.get_commands)
        self._remove_disabled_commands()

        # Get messages
        messages = await self.run_pipeline(MessageProvider.get_messages)

        prompt: ChatPrompt = self.prompt_strategy.build_prompt(
            messages=messages,
            task=self.state.task,
            ai_profile=self.state.ai_profile,
            ai_directives=directives,
            commands=get_openai_command_specs(self.commands),
            include_os_info=self.legacy_config.execute_local_commands,
        )

        self.log_cycle_handler.log_count_within_cycle = 0
        self.log_cycle_handler.log_cycle(
            self.state.ai_profile.ai_name,
            self.created_at,
            self.config.cycle_count,
            prompt.raw(),
            CURRENT_CONTEXT_FILE_NAME,
        )

        logger.debug(f"Executing prompt:\n{dump_prompt(prompt)}")
        output = await self.complete_and_parse(prompt)
        self.config.cycle_count += 1

        return output

    async def complete_and_parse(
        self, prompt: ChatPrompt, exception: Optional[Exception] = None
    ) -> ThoughtProcessOutput:
        if exception:
            prompt.messages.append(ChatMessage.system(f"Error: {exception}"))

        response: ChatModelResponse[
            ThoughtProcessOutput
        ] = await self.llm_provider.create_chat_completion(
            prompt.messages,
            model_name=self.llm.name,
            completion_parser=self.parse_and_validate_response,
            functions=(
                get_openai_command_specs(self.commands)
                if self.config.use_functions_api
                else []
            ),
        )
        result = response.parsed_result

        self.log_cycle_handler.log_cycle(
            self.state.ai_profile.ai_name,
            self.created_at,
            self.config.cycle_count,
            result.thoughts,
            NEXT_ACTION_FILE_NAME,
        )

        await self.run_pipeline(AfterParse.after_parse, result)

        return result

    def parse_and_validate_response(
        self, llm_response: AssistantChatMessage
    ) -> ThoughtProcessOutput:
        parsed_response = self.prompt_strategy.parse_response_content(llm_response)

        # Validate command arguments
        command_name = parsed_response.command_name
        command = self._get_command(command_name)
        if arg_errors := command.validate_args(parsed_response.command_args)[1]:
            fmt_errors = [
                f"{'.'.join(str(p) for p in f.path)}: {f.message}"
                if f.path
                else f.message
                for f in arg_errors
            ]
            raise InvalidArgumentError(
                f"The set of arguments supplied for {command_name} is invalid:\n"
                + "\n".join(fmt_errors)
            )

        return parsed_response

    async def execute(
        self,
        command_name: str,
        command_args: dict[str, str] = {},
        user_input: str = "",
    ) -> ActionResult:
        result: ActionResult

        if command_name == "human_feedback":
            result = ActionInterruptedByHuman(feedback=user_input)
            self.log_cycle_handler.log_cycle(
                self.state.ai_profile.ai_name,
                self.created_at,
                self.config.cycle_count,
                user_input,
                USER_INPUT_FILE_NAME,
            )

        else:
            # Get commands
            self.commands = await self.run_pipeline(CommandProvider.get_commands)
            self._remove_disabled_commands()

            try:
                return_value = await self._execute_command(
                    command_name=command_name,
                    arguments=command_args,
                )

                result = ActionSuccessResult(outputs=return_value)
            except AgentTerminated:
                raise
            except AgentException as e:
                result = ActionErrorResult.from_exception(e)
                logger.warning(
                    f"{command_name}({fmt_kwargs(command_args)}) raised an error: {e}"
                )
                sentry_sdk.capture_exception(e)

            result_tlength = self.llm_provider.count_tokens(str(result), self.llm.name)
            if result_tlength > self.send_token_limit // 3:
                result = ActionErrorResult(
                    reason=f"Command {command_name} returned too much output. "
                    "Do not execute this command again with the same arguments."
                )

        await self.run_pipeline(AfterExecute.after_execute, result)

        logger.debug("\n".join(self.trace))

        return result

    async def _execute_command(
        self,
        command_name: str,
        arguments: dict[str, str],
    ) -> CommandOutput:
        """Execute the command and return the result

        Args:
            command_name (str): The name of the command to execute
            arguments (dict): The arguments for the command

        Returns:
            str: The result of the command
        """
        # Execute a native command with the same name or alias, if it exists
        command = self._get_command(command_name)
        try:
            result = command(**arguments)
            if inspect.isawaitable(result):
                return await result
            return result
        except AgentException:
            raise
        except Exception as e:
            raise CommandExecutionError(str(e))

    def _get_command(self, command_name: str) -> Command:
        for command in reversed(self.commands):
            if command_name in command.names:
                return command

        raise UnknownCommandError(
            f"Cannot execute command '{command_name}': unknown command."
        )

    def _remove_disabled_commands(self) -> None:
        self.commands = [
            command
            for command in self.commands
            if not any(
                name in self.legacy_config.disabled_commands for name in command.names
            )
        ]

    def find_obscured_commands(self) -> list[Command]:
        seen_names = set()
        obscured_commands = []
        for command in reversed(self.commands):
            # If all of the command's names have been seen, it's obscured
            if seen_names.issuperset(command.names):
                obscured_commands.append(command)
            else:
                seen_names.update(command.names)
        return list(reversed(obscured_commands))
