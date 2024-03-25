from __future__ import annotations

import inspect
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from autogpts.autogpt.autogpt.agents.components import BuildPrompt, Component, ResponseHandler
from autogpts.autogpt.autogpt.agents.prompt_strategies.one_shot_component import OneShotComponent
from autogpts.autogpt.autogpt.llm.providers.openai import get_openai_command_specs
import sentry_sdk
from pydantic import Field

from autogpt.core.configuration import Configurable
from autogpt.core.prompting import ChatPrompt
from autogpt.core.resource.model_providers import (
    AssistantChatMessage,
    ChatMessage,
    ChatModelProvider,
)
from autogpt.file_storage.base import FileStorage
from autogpt.llm.api_manager import ApiManager
from autogpt.logs.log_cycle import (
    CURRENT_CONTEXT_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.logs.utils import fmt_kwargs
from autogpt.models.action_history import (
    Action,
    ActionErrorResult,
    ActionInterruptedByHuman,
    ActionResult,
    ActionSuccessResult,
)
from autogpt.models.command import Command, CommandOutput
from autogpt.models.context_item import ContextItem

from .base import AgentThoughts, BaseAgent, BaseAgentConfiguration, BaseAgentSettings, CommandArgs, CommandName, ComponentAgent
from .features.agent_file_manager import FileManagerComponent
from .features.context import ContextComponent
from .features.watchdog import WatchdogComponent
from .prompt_strategies.one_shot import (
    OneShotAgentPromptConfiguration,
    OneShotAgentPromptStrategy,
)
from .utils.exceptions import (
    AgentException,
    AgentTerminated,
    CommandExecutionError,
    DuplicateOperationError,
    UnknownCommandError,
)


if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.models.command_registry import CommandRegistry

logger = logging.getLogger(__name__)


class AgentConfiguration(BaseAgentConfiguration):
    pass


class AgentSettings(BaseAgentSettings):
    config: AgentConfiguration = Field(default_factory=AgentConfiguration)
    prompt_config: OneShotAgentPromptConfiguration = Field(
        default_factory=(
            lambda: OneShotAgentPromptStrategy.default_configuration.copy(deep=True)
        )
    )

ThoughtProcessOutput = tuple[CommandName, CommandArgs, AgentThoughts] #TODO remove


class ExtraAgentComponent(Component, BuildPrompt, ResponseHandler):
    """Things from the Agent/BaseAgent that should be moved to a component."""

    

    def build_prompt(
        self,
        result: BuildPrompt.Result,
    ) -> None:
        # Clock
        result.extra_messages.append(
            ChatMessage.system(f"The current time and date is {time.strftime('%c')}"),
        )

        # Add budget information (if any) to prompt
        api_manager = ApiManager()
        if api_manager.get_total_budget() > 0.0:
            remaining_budget = (
                api_manager.get_total_budget() - api_manager.get_total_cost()
            )
            if remaining_budget < 0:
                remaining_budget = 0

            budget_msg = ChatMessage.system(
                f"Your remaining API budget is ${remaining_budget:.3f}"
                + (
                    " BUDGET EXCEEDED! SHUT DOWN!\n\n"
                    if remaining_budget == 0
                    else " Budget very nearly exceeded! Shut down gracefully!\n\n"
                    if remaining_budget < 0.005
                    else " Budget nearly exceeded. Finish up.\n\n"
                    if remaining_budget < 0.01
                    else ""
                ),
            )
            logger.debug(budget_msg)
            result.extra_messages.append(budget_msg)

        #TODO this just need to be passed to OneShot build_prompt
        if include_os_info is None:
            include_os_info = self.legacy_config.execute_local_commands
    

    def on_before_think(
        self,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ChatPrompt:
        """Called after constructing the prompt but before executing it.

        Calls the `on_planning` hook of any enabled and capable plugins, adding their
        output to the prompt.

        Params:
            prompt: The prompt that is about to be executed.
            scratchpad: An object for plugins to write additional prompt elements to.
                (E.g. commands, constraints, best practices)

        Returns:
            The prompt to execute
        """
        current_tokens_used = self.llm_provider.count_message_tokens(
            prompt.messages, self.llm.name
        )
        plugin_count = len(self.config.plugins)
        for i, plugin in enumerate(self.config.plugins):
            if not plugin.can_handle_on_planning():
                continue
            plugin_response = plugin.on_planning(scratchpad, prompt.raw())
            if not plugin_response or plugin_response == "":
                continue
            message_to_add = ChatMessage.system(plugin_response)
            tokens_to_add = self.llm_provider.count_message_tokens(
                message_to_add, self.llm.name
            )
            if current_tokens_used + tokens_to_add > self.send_token_limit:
                logger.debug(f"Plugin response too long, skipping: {plugin_response}")
                logger.debug(f"Plugins remaining at stop: {plugin_count - i}")
                break
            prompt.messages.insert(
                -1, message_to_add
            )  # HACK: assumes cycle instruction to be at the end
            current_tokens_used += tokens_to_add
        return prompt

    def on_response(
        self,
        llm_response: ChatModelResponse,
        prompt: ChatPrompt,
        scratchpad: PromptScratchpad,
    ) -> ThoughtProcessOutput:
        """Called upon receiving a response from the chat model.

        Calls `self.parse_and_process_response()`.

        Params:
            llm_response: The raw response from the chat model.
            prompt: The prompt that was executed.
            scratchpad: An object containing additional prompt elements from plugins.
                (E.g. commands, constraints, best practices)

        Returns:
            The parsed command name and command args, if any, and the agent thoughts.
        """

        return llm_response.parsed_result

        # TODO: update memory/context


class SimpleAgent(ComponentAgent, Configurable[AgentSettings]):

    default_settings: AgentSettings = AgentSettings(
        name="Agent",
        description=__doc__ if __doc__ else "",
    )

    prompt_strategy: OneShotAgentPromptStrategy

    def __init__(
        self,
        settings: AgentSettings,
        llm_provider: ChatModelProvider,
        file_storage: FileStorage,
        legacy_config: Config,
    ):
        super().__init__(settings, llm_provider)

        # self.prompt_strategy = OneShotAgentPromptStrategy(
        #     configuration=settings.prompt_config,
        #     logger=logger,
        # )

        self.extra = ExtraAgentComponent()
        self.context = ContextComponent()
        self.file_manager = FileManagerComponent(settings, file_storage)
        self.watchdog = WatchdogComponent(settings.config, settings.history)
        self.prompt_strategy = OneShotComponent()

        # Override component ordering
        self.components = [
            self.extra, self.file_manager, self.context, self.watchdog, self.prompt_strategy,
        ]

        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        """Timestamp the agent was created; only used for structured debug logging."""

        self.log_cycle_handler = LogCycleHandler()
        """LogCycleHandler for structured debug logging."""

        self.event_history = settings.history
        self.legacy_config = legacy_config

    #TODO kcze legacy fuction - remove
    def on_before_think(self, *args, **kwargs) -> ChatPrompt:
        # prompt = super().on_before_think(*args, **kwargs)
        #TODO this from super()
        # current_tokens_used = self.llm_provider.count_message_tokens(
        #     prompt.messages, self.llm.name
        # )
        # plugin_count = len(self.config.plugins)
        # for i, plugin in enumerate(self.config.plugins):
        #     if not plugin.can_handle_on_planning():
        #         continue
        #     plugin_response = plugin.on_planning(scratchpad, prompt.raw())
        #     if not plugin_response or plugin_response == "":
        #         continue
        #     message_to_add = ChatMessage.system(plugin_response)
        #     tokens_to_add = self.llm_provider.count_message_tokens(
        #         message_to_add, self.llm.name
        #     )
        #     if current_tokens_used + tokens_to_add > self.send_token_limit:
        #         logger.debug(f"Plugin response too long, skipping: {plugin_response}")
        #         logger.debug(f"Plugins remaining at stop: {plugin_count - i}")
        #         break
        #     prompt.messages.insert(
        #         -1, message_to_add
        #     )  # HACK: assumes cycle instruction to be at the end
        #     current_tokens_used += tokens_to_add
        ####################

        self.log_cycle_handler.log_count_within_cycle = 0
        self.log_cycle_handler.log_cycle(
            self.state.ai_profile.ai_name,
            self.created_at,
            self.config.cycle_count,
            prompt.raw(),
            CURRENT_CONTEXT_FILE_NAME,
        )
        return prompt

    async def propose_action(self) -> ThoughtProcessOutput:
        """Proposes the next action to execute, based on the task and current state.

        Returns:
            The command name and arguments, if any, and the agent's thoughts.
        """

        #TODO kcze update guidelines

        # Execute build prompt_data
        prompt_data: BuildPrompt.Result = BuildPrompt.Result()
        self.foreach_components("build_prompt", prompt_data)

        # Get final prompt
        prompt = ChatPrompt(messages=[])
        self.foreach_components("get_prompt", prompt)

        # Get commands
        commands: list[Command] = []
        self.foreach_components("get_commands", commands)

        prompt = self.on_before_think()
        # prompt = self.on_before_think(prompt, scratchpad=self._prompt_scratchpad)

        # logger.debug(f"Executing prompt:\n{dump_prompt(prompt)}")
        response = await self.llm_provider.create_chat_completion(
            prompt.messages,
            functions=(
                get_openai_command_specs(
                    commands
                )
                if self.config.use_functions_api
                else []
            ),
            model_name=self.llm.name,
            completion_parser=lambda r: self.parse_and_process_response(
                r,
                prompt,
            ),
        )
        self.config.cycle_count += 1

        return response.parsed_result

    #TODO kcze legacy function - move to Component
    def parse_and_process_response(
        self, llm_response: AssistantChatMessage, *args, **kwargs
    ) -> SimpleAgent.ThoughtProcessOutput:
        # for plugin in self.config.plugins:
        #     if not plugin.can_handle_post_planning():
        #         continue
        #     llm_response.content = plugin.post_planning(llm_response.content or "")

        (
            command_name,
            arguments,
            assistant_reply_dict,
        ) = self.prompt_strategy.parse_response_content(llm_response)

        # Check if command_name and arguments are already in the event_history
        if self.event_history.matches_last_command(command_name, arguments):
            raise DuplicateOperationError(
                f"The command {command_name} with arguments {arguments} "
                f"has been just executed."
            )

        self.log_cycle_handler.log_cycle(
            self.state.ai_profile.ai_name,
            self.created_at,
            self.config.cycle_count,
            assistant_reply_dict,
            NEXT_ACTION_FILE_NAME,
        )

        if command_name:
            self.event_history.register_action(
                Action(
                    name=command_name,
                    args=arguments,
                    reasoning=assistant_reply_dict["thoughts"]["reasoning"],
                )
            )

        return command_name, arguments, assistant_reply_dict

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
            try:
                return_value = await self.execute_command(
                    command_name=command_name,
                    arguments=command_args,
                )

                # Intercept ContextItem if one is returned by the command
                #TODO kcze to OnExecute in contextComponent
                if type(return_value) is tuple and isinstance(
                    return_value[1], ContextItem
                ):
                    context_item = return_value[1]
                    return_value = return_value[0]
                    logger.debug(
                        f"Command {command_name} returned a ContextItem: {context_item}"
                    )
                    self.context.context.add(context_item)

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

            for plugin in self.config.plugins:
                if not plugin.can_handle_post_command():
                    continue
                if result.status == "success":
                    result.outputs = plugin.post_command(command_name, result.outputs)
                elif result.status == "error":
                    result.reason = plugin.post_command(command_name, result.reason)

        # Update action history
        self.event_history.register_result(result)
        await self.event_history.handle_compression(
            self.llm_provider, self.legacy_config
        )

        return result

    async def execute_command(
        self,
        command_name: str,
        arguments: dict[str, str],
    ) -> CommandOutput:
        """Execute the command and return the result

        Args:
            command_name (str): The name of the command to execute
            arguments (dict): The arguments for the command
            agent (Agent): The agent that is executing the command

        Returns:
            str: The result of the command
        """
        # Execute a native command with the same name or alias, if it exists
        if command := self.command_registry.get_command(command_name):
            try:
                result = command(**arguments, agent=self)
                if inspect.isawaitable(result):
                    return await result
                return result
            except AgentException:
                raise
            except Exception as e:
                raise CommandExecutionError(str(e))

        raise UnknownCommandError(
            f"Cannot execute command '{command_name}': unknown command."
        )
