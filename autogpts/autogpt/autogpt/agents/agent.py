from __future__ import annotations

import inspect
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.models.command_registry import CommandRegistry

from pydantic import Field

from autogpt.core.configuration import Configurable
from autogpt.core.prompting import ChatPrompt
from autogpt.core.resource.model_providers import (
    AssistantChatMessageDict,
    ChatMessage,
    ChatModelProvider,
)
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
from autogpt.models.command import CommandOutput
from autogpt.models.context_item import ContextItem

from .base import BaseAgent, BaseAgentConfiguration, BaseAgentSettings
from .features.context import ContextMixin
from .features.file_workspace import FileWorkspaceMixin
from .features.watchdog import WatchdogMixin
from .prompt_strategies.one_shot import (
    OneShotAgentPromptConfiguration,
    OneShotAgentPromptStrategy,
)
from .utils.exceptions import (
    AgentException,
    AgentTerminated,
    CommandExecutionError,
    UnknownCommandError,
)

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


class Agent(
    ContextMixin,
    FileWorkspaceMixin,
    WatchdogMixin,
    BaseAgent,
    Configurable[AgentSettings],
):
    """AutoGPT's primary Agent; uses one-shot prompting."""

    default_settings: AgentSettings = AgentSettings(
        name="Agent",
        description=__doc__,
    )

    prompt_strategy: OneShotAgentPromptStrategy

    def __init__(
        self,
        settings: AgentSettings,
        llm_provider: ChatModelProvider,
        command_registry: CommandRegistry,
        legacy_config: Config,
    ):
        prompt_strategy = OneShotAgentPromptStrategy(
            configuration=settings.prompt_config,
            logger=logger,
        )
        super().__init__(
            settings=settings,
            llm_provider=llm_provider,
            prompt_strategy=prompt_strategy,
            command_registry=command_registry,
            legacy_config=legacy_config,
        )

        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        """Timestamp the agent was created; only used for structured debug logging."""

        self.log_cycle_handler = LogCycleHandler()
        """LogCycleHandler for structured debug logging."""

    def build_prompt(
        self,
        *args,
        extra_messages: Optional[list[ChatMessage]] = None,
        include_os_info: Optional[bool] = None,
        **kwargs,
    ) -> ChatPrompt:
        if not extra_messages:
            extra_messages = []

        # Clock
        extra_messages.append(
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
            extra_messages.append(budget_msg)

        if include_os_info is None:
            include_os_info = self.legacy_config.execute_local_commands

        return super().build_prompt(
            *args,
            extra_messages=extra_messages,
            include_os_info=include_os_info,
            **kwargs,
        )

    def on_before_think(self, *args, **kwargs) -> ChatPrompt:
        prompt = super().on_before_think(*args, **kwargs)

        self.log_cycle_handler.log_count_within_cycle = 0
        self.log_cycle_handler.log_cycle(
            self.ai_profile.ai_name,
            self.created_at,
            self.config.cycle_count,
            prompt.raw(),
            CURRENT_CONTEXT_FILE_NAME,
        )
        return prompt

    def parse_and_process_response(
        self, llm_response: AssistantChatMessageDict, *args, **kwargs
    ) -> Agent.ThoughtProcessOutput:
        for plugin in self.config.plugins:
            if not plugin.can_handle_post_planning():
                continue
            llm_response["content"] = plugin.post_planning(
                llm_response.get("content", "")
            )

        (
            command_name,
            arguments,
            assistant_reply_dict,
        ) = self.prompt_strategy.parse_response_content(llm_response)

        self.log_cycle_handler.log_cycle(
            self.ai_profile.ai_name,
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
                self.ai_profile.ai_name,
                self.created_at,
                self.config.cycle_count,
                user_input,
                USER_INPUT_FILE_NAME,
            )

        else:
            for plugin in self.config.plugins:
                if not plugin.can_handle_pre_command():
                    continue
                command_name, command_args = plugin.pre_command(
                    command_name, command_args
                )

            try:
                return_value = await execute_command(
                    command_name=command_name,
                    arguments=command_args,
                    agent=self,
                )

                # Intercept ContextItem if one is returned by the command
                if type(return_value) is tuple and isinstance(
                    return_value[1], ContextItem
                ):
                    context_item = return_value[1]
                    return_value = return_value[0]
                    logger.debug(
                        f"Command {command_name} returned a ContextItem: {context_item}"
                    )
                    self.context.add(context_item)

                result = ActionSuccessResult(outputs=return_value)
            except AgentTerminated:
                raise
            except AgentException as e:
                result = ActionErrorResult.from_exception(e)
                logger.warning(
                    f"{command_name}({fmt_kwargs(command_args)}) raised an error: {e}"
                )

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

        return result


#############
# Utilities #
#############


async def execute_command(
    command_name: str,
    arguments: dict[str, str],
    agent: Agent,
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
    if command := agent.command_registry.get_command(command_name):
        try:
            result = command(**arguments, agent=agent)
            if inspect.isawaitable(result):
                return await result
            return result
        except AgentException:
            raise
        except Exception as e:
            raise CommandExecutionError(str(e))

    # Handle non-native commands (e.g. from plugins)
    if agent._prompt_scratchpad:
        for name, command in agent._prompt_scratchpad.commands.items():
            if (
                command_name == name
                or command_name.lower() == command.description.lower()
            ):
                try:
                    return command.method(**arguments)
                except AgentException:
                    raise
                except Exception as e:
                    raise CommandExecutionError(str(e))

    raise UnknownCommandError(
        f"Cannot execute command '{command_name}': unknown command."
    )
