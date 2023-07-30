from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Optional

if TYPE_CHECKING:
    from autogpt.config import AIConfig, Config
    from autogpt.llm.base import ChatModelResponse, ChatSequence
    from autogpt.memory.vector import VectorMemory
    from autogpt.models.command_registry import CommandRegistry

from autogpt.agents.utils.exceptions import (
    AgentException,
    CommandExecutionError,
    InvalidAgentResponseError,
    UnknownCommandError,
)
from autogpt.json_utils.utilities import extract_dict_from_response, validate_dict
from autogpt.llm.base import Message
from autogpt.llm.utils import count_string_tokens
from autogpt.logs import logger
from autogpt.logs.log_cycle import (
    CURRENT_CONTEXT_FILE_NAME,
    FULL_MESSAGE_HISTORY_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.models.agent_actions import (
    ActionErrorResult,
    ActionHistory,
    ActionInterruptedByHuman,
    ActionResult,
    ActionSuccessResult,
)
from autogpt.models.command import CommandOutput
from autogpt.memory.agent_history import ActionHistory
from autogpt.workspace import Workspace

from .base import AgentThoughts, BaseAgent, CommandArgs, CommandName

PLANNING_AGENT_SYSTEM_PROMPT = """You are an AI agent named {ai_name}"""


class PlanningAgent(BaseAgent):
    """Agent class for interacting with Auto-GPT."""

    ThoughtProcessID = Literal["plan", "action", "evaluate"]

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

        self.workspace = Workspace(config.workspace_path, config.restrict_to_workspace)
        """Workspace that the agent has access to, e.g. for reading/writing files."""

        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        """Timestamp the agent was created; only used for structured debug logging."""

        self.log_cycle_handler = LogCycleHandler()
        """LogCycleHandler for structured debug logging."""

        self.action_history = ActionHistory()

        self.plan: list[str] = []
        """List of steps that the Agent plans to take"""

    def construct_base_prompt(
        self, thought_process_id: ThoughtProcessID, **kwargs
    ) -> ChatSequence:
        prepend_messages = kwargs["prepend_messages"] = kwargs.get(
            "prepend_messages", []
        )

        match thought_process_id:
            case "plan" | "action":
                # Add the current plan to the prompt, if any
                if self.plan:
                    plan_section = [
                        "## Plan",
                        "To complete your task, you have made the following plan:",
                    ]
                    plan_section += [f"{i}. {s}" for i, s in enumerate(self.plan, 1)]

                    # Add the actions so far to the prompt
                    if self.action_history:
                        plan_section += [
                            "\n### Progress",
                            "So far, you have executed the following actions based on the plan:",
                        ]
                        for i, cycle in enumerate(self.action_history, 1):
                            if not (cycle.action and cycle.result):
                                logger.warn(f"Incomplete action in history: {cycle}")
                                continue

                            plan_section.append(
                                f"{i}. You executed the command `{cycle.action.format_call()}`, "
                                f"which gave the result `{cycle.result}`."
                            )

                    prepend_messages.append(Message("system", "\n".join(plan_section)))

            case "evaluate":
                pass
            case _:
                raise NotImplementedError(
                    f"Unknown thought process '{thought_process_id}'"
                )

        return super().construct_base_prompt(
            thought_process_id=thought_process_id, **kwargs
        )

    def on_before_think(self, *args, **kwargs) -> ChatSequence:
        prompt = super().on_before_think(*args, **kwargs)

        self.log_cycle_handler.log_count_within_cycle = 0
        self.log_cycle_handler.log_cycle(
            self.ai_config.ai_name,
            self.created_at,
            self.cycle_count,
            self.history.raw(),
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

                result = ActionSuccessResult(return_value)
            except AgentException as e:
                result = ActionErrorResult(e.message, e)

            result_tlength = count_string_tokens(str(result), self.llm.name)
            memory_tlength = count_string_tokens(
                str(self.history.summary_message()), self.llm.name
            )
            if result_tlength + memory_tlength > self.send_token_limit:
                result = ActionErrorResult(
                    reason=f"Command {command_name} returned too much output. "
                    "Do not execute this command again with the same arguments."
                )

            for plugin in self.config.plugins:
                if not plugin.can_handle_post_command():
                    continue
                if result.status == "success":
                    result.results = plugin.post_command(command_name, result.results)
                elif result.status == "error":
                    result.reason = plugin.post_command(command_name, result.reason)

        # Check if there's a result from the command append it to the message
        if result.status == "success":
            self.history.add(
                "system",
                f"Command {command_name} returned: {result.results}",
                "action_result",
            )
        elif result.status == "error":
            message = f"Command {command_name} failed: {result.reason}"
            if (
                result.error
                and isinstance(result.error, AgentException)
                and result.error.hint
            ):
                message = message.rstrip(".") + f". {result.error.hint}"
            self.history.add("system", message, "action_result")

        return result

    def parse_and_process_response(
        self, llm_response: ChatModelResponse, *args, **kwargs
    ) -> PlanningAgent.ThoughtProcessOutput:
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
    agent: PlanningAgent,
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
    for command in agent.ai_config.prompt_generator.commands:
        if (
            command_name == command.label.lower()
            or command_name == command.name.lower()
        ):
            try:
                return command.function(**arguments)
            except AgentException:
                raise
            except Exception as e:
                raise CommandExecutionError(str(e))

    raise UnknownCommandError(
        f"Cannot execute command '{command_name}': unknown command."
    )
