from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from autogpt.config import AIConfig, Config
    from autogpt.llm.base import ChatModelResponse, ChatSequence
    from autogpt.memory.vector import VectorMemory
    from autogpt.models.command_registry import CommandRegistry

from autogpt.json_utils.utilities import extract_dict_from_response, validate_dict
from autogpt.llm.api_manager import ApiManager
from autogpt.llm.base import Message
from autogpt.llm.utils import count_string_tokens
from autogpt.logs.log_cycle import (
    CURRENT_CONTEXT_FILE_NAME,
    FULL_MESSAGE_HISTORY_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.models.agent_actions import (
    Action,
    ActionErrorResult,
    ActionInterruptedByHuman,
    ActionResult,
    ActionSuccessResult,
)
from autogpt.models.command import CommandOutput
from autogpt.models.context_item import ContextItem

from .base import BaseAgent
from .features.context import ContextMixin
from .features.workspace import WorkspaceMixin
from .utils.exceptions import (
    AgentException,
    CommandExecutionError,
    InvalidAgentResponseError,
    UnknownCommandError,
)

logger = logging.getLogger(__name__)


class Agent(ContextMixin, WorkspaceMixin, BaseAgent):
    """Agent class for interacting with Auto-GPT."""

    def __init__(
        self,
        ai_config: AIConfig,
        command_registry: CommandRegistry,
        memory: VectorMemory,
        triggering_prompt: str,
        config: Config,
        cycle_budget: Optional[int] = None,
    ):
        super().__init__(
            ai_config=ai_config,
            command_registry=command_registry,
            config=config,
            default_cycle_instruction=triggering_prompt,
            cycle_budget=cycle_budget,
        )

        self.memory = memory
        """VectorMemoryProvider used to manage the agent's context (TODO)"""

        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        """Timestamp the agent was created; only used for structured debug logging."""

        self.log_cycle_handler = LogCycleHandler()
        """LogCycleHandler for structured debug logging."""

    def construct_base_prompt(self, *args, **kwargs) -> ChatSequence:
        if kwargs.get("prepend_messages") is None:
            kwargs["prepend_messages"] = []

        # Clock
        kwargs["prepend_messages"].append(
            Message("system", f"The current time and date is {time.strftime('%c')}"),
        )

        # Add budget information (if any) to prompt
        api_manager = ApiManager()
        if api_manager.get_total_budget() > 0.0:
            remaining_budget = (
                api_manager.get_total_budget() - api_manager.get_total_cost()
            )
            if remaining_budget < 0:
                remaining_budget = 0

            budget_msg = Message(
                "system",
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

            if kwargs.get("append_messages") is None:
                kwargs["append_messages"] = []
            kwargs["append_messages"].append(budget_msg)

        # # Include message history in base prompt
        # kwargs["with_message_history"] = True

        return super().construct_base_prompt(*args, **kwargs)

    def on_before_think(self, *args, **kwargs) -> ChatSequence:
        prompt = super().on_before_think(*args, **kwargs)

        self.log_cycle_handler.log_count_within_cycle = 0
        self.log_cycle_handler.log_cycle(
            self.ai_config.ai_name,
            self.created_at,
            self.cycle_count,
            self.message_history.raw(),
            FULL_MESSAGE_HISTORY_FILE_NAME,
        )
        self.log_cycle_handler.log_cycle(
            self.ai_config.ai_name,
            self.created_at,
            self.cycle_count,
            prompt.raw(),
            CURRENT_CONTEXT_FILE_NAME,
        )
        return prompt

    def execute(
        self,
        command_name: str,
        command_args: dict[str, str] = {},
        user_input: str = "",
    ) -> ActionResult:
        result: ActionResult

        if command_name == "human_feedback":
            result = ActionInterruptedByHuman(user_input)
            self.message_history.add(
                "user",
                "I interrupted the execution of the command you proposed "
                f"to give you some feedback: {user_input}",
            )
            self.log_cycle_handler.log_cycle(
                self.ai_config.ai_name,
                self.created_at,
                self.cycle_count,
                user_input,
                USER_INPUT_FILE_NAME,
            )

        else:
            for plugin in self.config.plugins:
                if not plugin.can_handle_pre_command():
                    continue
                command_name, arguments = plugin.pre_command(command_name, command_args)

            try:
                return_value = execute_command(
                    command_name=command_name,
                    arguments=command_args,
                    agent=self,
                )

                # Intercept ContextItem if one is returned by the command
                if type(return_value) == tuple and isinstance(
                    return_value[1], ContextItem
                ):
                    context_item = return_value[1]
                    return_value = return_value[0]
                    logger.debug(
                        f"Command {command_name} returned a ContextItem: {context_item}"
                    )
                    self.context.add(context_item)

                result = ActionSuccessResult(return_value)
            except AgentException as e:
                result = ActionErrorResult(e.message, e)

            result_tlength = count_string_tokens(str(result), self.llm.name)
            history_tlength = count_string_tokens(
                self.event_history.fmt_paragraph(), self.llm.name
            )
            if result_tlength + history_tlength > self.send_token_limit:
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

        # Check if there's a result from the command append it to the message
        if result.status == "success":
            self.message_history.add(
                "system",
                f"Command {command_name} returned: {result.outputs}",
                "action_result",
            )
        elif result.status == "error":
            message = f"Command {command_name} failed: {result.reason}"

            # Append hint to the error message if the exception has a hint
            if (
                result.error
                and isinstance(result.error, AgentException)
                and result.error.hint
            ):
                message = message.rstrip(".") + f". {result.error.hint}"

            self.message_history.add("system", message, "action_result")

        # Update action history
        self.event_history.register_result(result)

        return result

    def parse_and_process_response(
        self, llm_response: ChatModelResponse, *args, **kwargs
    ) -> Agent.ThoughtProcessOutput:
        if not llm_response.content:
            raise InvalidAgentResponseError("Assistant response has no text content")

        response_content = llm_response.content

        for plugin in self.config.plugins:
            if not plugin.can_handle_post_planning():
                continue
            response_content = plugin.post_planning(response_content)

        assistant_reply_dict = extract_dict_from_response(response_content)

        _, errors = validate_dict(assistant_reply_dict, self.config)
        if errors:
            raise InvalidAgentResponseError(
                "Validation of response failed:\n  "
                + ";\n  ".join([str(e) for e in errors])
            )

        # Get command name and arguments
        command_name, arguments = extract_command(
            assistant_reply_dict, llm_response, self.config
        )
        response = command_name, arguments, assistant_reply_dict

        self.log_cycle_handler.log_cycle(
            self.ai_config.ai_name,
            self.created_at,
            self.cycle_count,
            assistant_reply_dict,
            NEXT_ACTION_FILE_NAME,
        )

        self.event_history.register_action(
            Action(
                name=command_name,
                args=arguments,
                reasoning=assistant_reply_dict["thoughts"]["reasoning"],
            )
        )

        return response


def extract_command(
    assistant_reply_json: dict, assistant_reply: ChatModelResponse, config: Config
) -> tuple[str, dict[str, str]]:
    """Parse the response and return the command name and arguments

    Args:
        assistant_reply_json (dict): The response object from the AI
        assistant_reply (ChatModelResponse): The model response from the AI
        config (Config): The config object

    Returns:
        tuple: The command name and arguments

    Raises:
        json.decoder.JSONDecodeError: If the response is not valid JSON

        Exception: If any other error occurs
    """
    if config.openai_functions:
        if assistant_reply.function_call is None:
            raise InvalidAgentResponseError("No 'function_call' in assistant reply")
        assistant_reply_json["command"] = {
            "name": assistant_reply.function_call.name,
            "args": json.loads(assistant_reply.function_call.arguments),
        }
    try:
        if not isinstance(assistant_reply_json, dict):
            raise InvalidAgentResponseError(
                f"The previous message sent was not a dictionary {assistant_reply_json}"
            )

        if "command" not in assistant_reply_json:
            raise InvalidAgentResponseError("Missing 'command' object in JSON")

        command = assistant_reply_json["command"]
        if not isinstance(command, dict):
            raise InvalidAgentResponseError("'command' object is not a dictionary")

        if "name" not in command:
            raise InvalidAgentResponseError("Missing 'name' field in 'command' object")

        command_name = command["name"]

        # Use an empty dictionary if 'args' field is not present in 'command' object
        arguments = command.get("args", {})

        return command_name, arguments

    except json.decoder.JSONDecodeError:
        raise InvalidAgentResponseError("Invalid JSON")

    except Exception as e:
        raise InvalidAgentResponseError(str(e))


def execute_command(
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
            return command(**arguments, agent=agent)
        except AgentException:
            raise
        except Exception as e:
            raise CommandExecutionError(str(e))

    # Handle non-native commands (e.g. from plugins)
    for name, command in agent.prompt_generator.commands.items():
        if command_name == name or command_name.lower() == command.description.lower():
            try:
                return command.function(**arguments)
            except AgentException:
                raise
            except Exception as e:
                raise CommandExecutionError(str(e))

    raise UnknownCommandError(
        f"Cannot execute command '{command_name}': unknown command."
    )
