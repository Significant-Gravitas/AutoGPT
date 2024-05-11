from __future__ import annotations

import inspect
import logging
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import sentry_sdk
from forge.agent.base import BaseAgent, BaseAgentConfiguration, BaseAgentSettings
from forge.agent.protocols import (
    AfterExecute,
    AfterParse,
    CommandProvider,
    DirectiveProvider,
    MessageProvider,
)
from forge.command.command import Command, CommandOutput
from forge.components.action_history.event_history_component import (
    EventHistoryComponent,
)
from forge.components.action_history.model import (
    ActionErrorResult,
    ActionInterruptedByHuman,
    ActionResult,
    ActionSuccessResult,
    EpisodicActionHistory,
)
from forge.components.code_executor.code_executor import CodeExecutorComponent
from forge.components.context.context import AgentContext, ContextComponent
from forge.components.file_manager import FileManagerComponent
from forge.components.git_operations import GitOperationsComponent
from forge.components.image_gen import ImageGeneratorComponent
from forge.components.system import SystemComponent
from forge.components.user_interaction import UserInteractionComponent
from forge.components.watchdog import WatchdogComponent
from forge.components.web_search import WebSearchComponent, WebSeleniumComponent
from forge.file_storage.base import FileStorage
from forge.llm.prompting.schema import ChatPrompt
from forge.llm.providers import (
    AssistantFunctionCall,
    ChatMessage,
    ChatModelProvider,
    ChatModelResponse,
)
from forge.llm.providers.utils import function_specs_from_commands
from forge.models.config import Configurable
from forge.utils.exceptions import (
    AgentException,
    AgentTerminated,
    CommandExecutionError,
    UnknownCommandError,
)
from pydantic import Field

from autogpt.app.log_cycle import (
    CURRENT_CONTEXT_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.core.runner.client_lib.logging.helpers import dump_prompt

from .prompt_strategies.one_shot import (
    OneShotAgentActionProposal,
    OneShotAgentPromptStrategy,
)

if TYPE_CHECKING:
    from forge.config.config import Config

logger = logging.getLogger(__name__)


class AgentConfiguration(BaseAgentConfiguration):
    pass


class AgentSettings(BaseAgentSettings):
    config: AgentConfiguration = Field(default_factory=AgentConfiguration)

    history: EpisodicActionHistory[OneShotAgentActionProposal] = Field(
        default_factory=EpisodicActionHistory[OneShotAgentActionProposal]
    )
    """(STATE) The action history of the agent."""

    context: AgentContext = Field(default_factory=AgentContext)


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
        prompt_config.use_functions_api = (
            settings.config.use_functions_api
            # Anthropic currently doesn't support tools + prefilling :(
            and self.llm.provider_name != "anthropic"
        )
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
        self.context = ContextComponent(self.file_manager.workspace, settings.context)
        self.watchdog = WatchdogComponent(settings.config, settings.history)

        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        """Timestamp the agent was created; only used for structured debug logging."""

        self.log_cycle_handler = LogCycleHandler()
        """LogCycleHandler for structured debug logging."""

        self.event_history = settings.history
        self.legacy_config = legacy_config

    async def propose_action(self) -> OneShotAgentActionProposal:
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
            commands=function_specs_from_commands(self.commands),
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
    ) -> OneShotAgentActionProposal:
        if exception:
            prompt.messages.append(ChatMessage.system(f"Error: {exception}"))

        response: ChatModelResponse[
            OneShotAgentActionProposal
        ] = await self.llm_provider.create_chat_completion(
            prompt.messages,
            model_name=self.llm.name,
            completion_parser=self.prompt_strategy.parse_response_content,
            functions=prompt.functions,
            prefill_response=prompt.prefill_response,
        )
        result = response.parsed_result

        self.log_cycle_handler.log_cycle(
            self.state.ai_profile.ai_name,
            self.created_at,
            self.config.cycle_count,
            result.thoughts.dict(),
            NEXT_ACTION_FILE_NAME,
        )

        await self.run_pipeline(AfterParse.after_parse, result)

        return result

    async def execute(
        self,
        proposal: OneShotAgentActionProposal,
        user_feedback: str = "",
    ) -> ActionResult:
        tool = proposal.use_tool

        # Get commands
        self.commands = await self.run_pipeline(CommandProvider.get_commands)
        self._remove_disabled_commands()

        try:
            return_value = await self._execute_tool(tool)

            result = ActionSuccessResult(outputs=return_value)
        except AgentTerminated:
            raise
        except AgentException as e:
            result = ActionErrorResult.from_exception(e)
            logger.warning(f"{tool} raised an error: {e}")
            sentry_sdk.capture_exception(e)

        result_tlength = self.llm_provider.count_tokens(str(result), self.llm.name)
        if result_tlength > self.send_token_limit // 3:
            result = ActionErrorResult(
                reason=f"Command {tool.name} returned too much output. "
                "Do not execute this command again with the same arguments."
            )

        await self.run_pipeline(AfterExecute.after_execute, result)

        logger.debug("\n".join(self.trace))

        return result

    async def do_not_execute(
        self, denied_proposal: OneShotAgentActionProposal, user_feedback: str
    ) -> ActionResult:
        result = ActionInterruptedByHuman(feedback=user_feedback)
        self.log_cycle_handler.log_cycle(
            self.state.ai_profile.ai_name,
            self.created_at,
            self.config.cycle_count,
            user_feedback,
            USER_INPUT_FILE_NAME,
        )

        await self.run_pipeline(AfterExecute.after_execute, result)

        logger.debug("\n".join(self.trace))

        return result

    async def _execute_tool(self, tool_call: AssistantFunctionCall) -> CommandOutput:
        """Execute the command and return the result

        Args:
            tool_call (AssistantFunctionCall): The tool call to execute

        Returns:
            str: The execution result
        """
        # Execute a native command with the same name or alias, if it exists
        command = self._get_command(tool_call.name)
        try:
            result = command(**tool_call.arguments)
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
