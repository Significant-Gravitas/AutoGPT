from __future__ import annotations

import uuid
import enum
import json
import platform
import re
from logging import Logger
from typing import Optional, TYPE_CHECKING

import distro

if TYPE_CHECKING:
    from autogpt.core.agents.simple import SimpleAgent

from pydantic import BaseModel

from autogpt.core.configuration import SystemConfiguration, UserConfigurable

# prompting
from autogpt.core.prompting.base import (
    PlanningPromptStrategiesConfiguration,
    PlanningPromptStrategy,
    PromptStrategiesConfiguration,
)
from autogpt.core.prompting.schema import (
    LanguageModelClassification,
)

from autogpt.core.resource.model_providers import (
    AssistantChatMessageDict,
    ChatMessage,
    CompletionModelFunction,
    OpenAIProvider,
    ChatPrompt,
)

from autogpt.core.utils.json_schema import JSONSchema

from autogpt.core.prompting.utils import json_loads, to_numbered_list, to_string_list



DEFAULT_TRIGGERING_PROMPT = (
    "Determine exactly one command to use next based on the given goals "
    "and the progress you have made so far, "
    "and respond using the JSON schema specified previously:"
)

class ThinkStrategyFunctionNames(str, enum.Enum):
    THINK: str = "think"


class ThinkStrategyConfiguration(PlanningPromptStrategiesConfiguration):
    model_classification: LanguageModelClassification = (
        LanguageModelClassification.FAST_MODEL_16K
    )

class ThinkStrategy(PlanningPromptStrategy):
    default_configuration = ThinkStrategyConfiguration()
    STRATEGY_NAME = "think"

    FIRST_SYSTEM_PROMPT_TEMPLATE = DEFAULT_TRIGGERING_PROMPT

    FUNCTION_THINK = CompletionModelFunction(
        name=ThinkStrategyFunctionNames.THINK,
        description="Seals the iterative process of refining requirements. It gets activated when the user communicates satisfaction with the requirements, signaling readiness to finalize the current list of goals.",
        parameters={
            "goal_list": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                minItems=1,
                maxItems=5,
                items=JSONSchema(
                    type=JSONSchema.Type.STRING,
                ),
                description="List of user requirements that emerged from prior interactions. Each entry in the list stands for a distinct and atomic requirement or aim expressed by the user.",
                required=True,
            )
        },
    )

    def __init__(
        self,
        logger: Logger,
        model_classification: LanguageModelClassification,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._logger = logger
        self._model_classification = model_classification
        self._config = self.default_configuration

        self._functions = [ThinkStrategy.FUNCTION_THINK]

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    def build_prompt(
        self, agent: SimpleAgent, instruction: str, thought_process_id, **kwargs
    ) -> ChatPrompt:
        # #self._provider : OpenAIProvider = kwargs['provider']
        model_name = kwargs["model_name"]

        #
        # STEP 1 : List all functions available
        #

        if not instruction:
            raise ValueError("No instruction given")        
                             

        # System message
        response_format_instr = self.response_format_instruction(
            agent=agent,
            thought_process_id=thought_process_id,
            model_name=model_name,
        )
        if response_format_instr:
            self._append_messages.append(ChatMessage.system(response_format_instr))

        # User Message
        instruction_msg = ChatMessage.user(instruction)
        instruction_tlength = agent._openai_provider.count_message_tokens(
            instruction_msg, model_name
        )



        messages: list[ChatMessage] = self.construct_base_prompt(
            agent=agent,
            thought_process_id=thought_process_id,
            append_messages=self._append_messages,
            reserve_tokens=instruction_tlength,
        )

        # ADD user input message ("triggering prompt")
        messages.append(instruction_msg)

        messages: list[ChatMessage] = agent._loop.on_before_think(
            messages=messages,
            thought_process_id=thought_process_id,
            instruction=instruction,
        )

        return ChatPrompt(
            messages=messages,
            functions=self.get_functions(),  # self._agent._tool_registry.dump_tools()
            function_call=ThinkStrategyFunctionNames.THINK,
            default_function_call="human_feedback",
        )


    # NOTE : based on planning_agent.py
    def construct_base_prompt(
        self, agent: SimpleAgent, thought_process_id: str, **kwargs
    ) -> list[ChatMessage]:
        # Add the current plan to the prompt, if any
        if agent.plan:
            plan_section = [
                "## Plan",
                "To complete your task, you have composed the following plan:",
            ]
            plan_section += [f"{i}. {s}" for i, s in enumerate(agent.plan, 1)]

            # Add the actions so far to the prompt
            if agent.event_history:
                plan_section += [
                    "\n### Progress",
                    "So far, you have executed the following actions based on the plan:",
                ]
                for i, cycle in enumerate(agent.event_history, 1):
                    if not (cycle.action and cycle.result):
                        agent._logger.warn(f"Incomplete action in history: {cycle}")
                        continue

                    plan_section.append(
                        f"{i}. You executed the command `{cycle.action.format_call()}`, "
                        f"which gave the result `{cycle.result}`."
                    )

            self._prepend_messages.append(ChatMessage.system("\n".join(plan_section)))

        # NOTE : PLANCE HOLDER
        # if agent.context:
        #     context_section = [
        #         "## Context",
        #         "Below is information that may be relevant to your task. These take up "
        #         "part of your working memory, which is limited, so when a context item is "
        #         "no longer relevant for your plan, use the `close_context_item` command to "
        #         "free up some memory."
        #         "\n",
        #         self.context.format_numbered(),
        #     ]
        #     self._prepend_messages.append(ChatMessage.system("\n".join(context_section)))

        # match thought_process_id:
        #     case "plan":
        #         # TODO: add planning instructions; details about what to pay attention to when planning
        #         pass
        #     case "action":
        #         # TODO: need to insert the functions here again?
        #         pass
        #     case "evaluate":
        #         # TODO: insert latest action (with reasoning) + result + evaluation instructions
        #         pass
        #     case _:
        #         raise NotImplementedError(
        #             f"Unknown thought process '{thought_process_id}'"
        #         )

        messages = super().construct_base_prompt(
            agent=agent, thought_process_id=thought_process_id, **kwargs
        )

        return messages

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> dict:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.

        """
        try:
            parsed_response = json_loads(response_content["function_call"]["arguments"])
        except Exception:
            self._agent._logger.warning(parsed_response)

        parsed_response["name"] = response_content["function_call"]["name"]

        return parsed_response

    def save(self):
        pass
