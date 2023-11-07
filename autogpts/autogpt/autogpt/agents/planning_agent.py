from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from autogpt.config import Config
    from autogpt.llm.base import ChatModelResponse, ChatSequence
    from autogpt.memory.vector import VectorMemory
    from autogpt.models.command_registry import CommandRegistry

from autogpt.agents.utils.exceptions import AgentException, InvalidAgentResponseError
from autogpt.json_utils.utilities import extract_dict_from_response, validate_dict
from autogpt.llm.base import Message
from autogpt.llm.utils import count_string_tokens
from autogpt.logs.log_cycle import (
    CURRENT_CONTEXT_FILE_NAME,
    NEXT_ACTION_FILE_NAME,
    USER_INPUT_FILE_NAME,
    LogCycleHandler,
)
from autogpt.models.action_history import (
    ActionErrorResult,
    ActionInterruptedByHuman,
    ActionResult,
    ActionSuccessResult,
)
from autogpt.models.context_item import ContextItem

from .agent import execute_command, extract_command
from .base import BaseAgent
from .features.context import ContextMixin
from .features.file_workspace import FileWorkspaceMixin

logger = logging.getLogger(__name__)


class PlanningAgent(ContextMixin, FileWorkspaceMixin, BaseAgent):
    """Agent class for interacting with AutoGPT."""

    ThoughtProcessID = Literal["plan", "action", "evaluate"]

    def __init__(
        self,
        command_registry: CommandRegistry,
        memory: VectorMemory,
        triggering_prompt: str,
        config: Config,
        cycle_budget: Optional[int] = None,
    ):
        super().__init__(
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

        self.plan: list[str] = []
        """List of steps that the Agent plans to take"""

    def construct_base_prompt(
        self, thought_process_id: ThoughtProcessID, **kwargs
    ) -> ChatSequence:
        prepend_messages = kwargs["prepend_messages"] = kwargs.get(
            "prepend_messages", []
        )

        # Add the current plan to the prompt, if any
        if self.plan:
            plan_section = [
                "## Plan",
                "To complete your task, you have composed the following plan:",
            ]
            plan_section += [f"{i}. {s}" for i, s in enumerate(self.plan, 1)]

            # Add the actions so far to the prompt
            if self.event_history:
                plan_section += [
                    "\n### Progress",
                    "So far, you have executed the following actions based on the plan:",
                ]
                for i, cycle in enumerate(self.event_history, 1):
                    if not (cycle.action and cycle.result):
                        logger.warn(f"Incomplete action in history: {cycle}")
                        continue

                    plan_section.append(
                        f"{i}. You executed the command `{cycle.action.format_call()}`, "
                        f"which gave the result `{cycle.result}`."
                    )

            prepend_messages.append(Message("system", "\n".join(plan_section)))

        if self.context:
            context_section = [
                "## Context",
                "Below is information that may be relevant to your task. These take up "
                "part of your working memory, which is limited, so when a context item is "
                "no longer relevant for your plan, use the `close_context_item` command to "
                "free up some memory."
                "\n",
                self.context.format_numbered(),
            ]
            prepend_messages.append(Message("system", "\n".join(context_section)))

        match thought_process_id:
            case "plan":
                # TODO: add planning instructions; details about what to pay attention to when planning
                pass
            case "action":
                # TODO: need to insert the functions here again?
                pass
            case "evaluate":
                # TODO: insert latest action (with reasoning) + result + evaluation instructions
                pass
            case _:
                raise NotImplementedError(
                    f"Unknown thought process '{thought_process_id}'"
                )

        return super().construct_base_prompt(
            thought_process_id=thought_process_id, **kwargs
        )

    def response_format_instruction(self, thought_process_id: ThoughtProcessID) -> str:
        match thought_process_id:
            case "plan":
                # TODO: add planning instructions; details about what to pay attention to when planning
                response_format = f"""```ts
                interface Response {{
                    thoughts: {{
                        // Thoughts
                        text: string;
                        // A short logical explanation about how the action is part of the earlier composed plan
                        reasoning: string;
                        // Constructive self-criticism
                        criticism: string;
                    }};
                    // A plan to achieve the goals with the available resources and/or commands.
                    plan: Array<{{
                        // An actionable subtask
                        subtask: string;
                        // Criterium to determine whether the subtask has been completed
                        completed_if: string;
                    }}>;
                }}
                ```"""
                pass
            case "action":
                # TODO: need to insert the functions here again?
                response_format = """```ts
                interface Response {
                    thoughts: {
                        // Thoughts
                        text: string;
                        // A short logical explanation about how the action is part of the earlier composed plan
                        reasoning: string;
                        // Constructive self-criticism
                        criticism: string;
                    };
                    // The action to take, from the earlier specified list of commands
                    command: {
                        name: string;
                        args: Record<string, any>;
                    };
                }
                ```"""
                pass
            case "evaluate":
                # TODO: insert latest action (with reasoning) + result + evaluation instructions
                response_format = f"""```ts
                interface Response {{
                    thoughts: {{
                        // Thoughts
                        text: string;
                        reasoning: string;
                        // Constructive self-criticism
                        criticism: string;
                    }};
                    result_evaluation: {{
                        // A short logical explanation of why the given partial result does or does not complete the corresponding subtask
                        reasoning: string;
                        // Whether the current subtask has been completed
                        completed: boolean;
                        // An estimate of the progress (0.0 - 1.0) that has been made on the subtask with the actions that have been taken so far
                        progress: float;
                    }};
                }}
                ```"""
                pass
            case _:
                raise NotImplementedError(
                    f"Unknown thought process '{thought_process_id}'"
                )

        response_format = re.sub(
            r"\n\s+",
            "\n",
            response_format,
        )

        return (
            f"Respond strictly with JSON. The JSON should be compatible with "
            "the TypeScript type `Response` from the following:\n"
            f"{response_format}\n"
        )

    def on_before_think(self, *args, **kwargs) -> ChatSequence:
        prompt = super().on_before_think(*args, **kwargs)

        self.log_cycle_handler.log_count_within_cycle = 0
        self.log_cycle_handler.log_cycle(
            self.ai_profile.ai_name,
            self.created_at,
            self.cycle_count,
            self.event_history.episodes,
            "event_history.json",
        )
        self.log_cycle_handler.log_cycle(
            self.ai_profile.ai_name,
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
            result = ActionInterruptedByHuman(feedback=user_input)
            self.log_cycle_handler.log_cycle(
                self.ai_profile.ai_name,
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
                    self.context.add(return_value[1])
                    return_value = return_value[0]

                result = ActionSuccessResult(outputs=return_value)
            except AgentException as e:
                result = ActionErrorResult.from_exception(e)

            result_tlength = count_string_tokens(str(result), self.llm.name)
            memory_tlength = count_string_tokens(
                str(self.event_history.fmt_paragraph()), self.llm.name
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
                    result.outputs = plugin.post_command(command_name, result.outputs)
                elif result.status == "error":
                    result.reason = plugin.post_command(command_name, result.reason)

        return result

    def parse_and_process_response(
        self,
        llm_response: ChatModelResponse,
        thought_process_id: ThoughtProcessID,
        *args,
        **kwargs,
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
            self.ai_profile.ai_name,
            self.created_at,
            self.cycle_count,
            assistant_reply_dict,
            NEXT_ACTION_FILE_NAME,
        )
        return response
