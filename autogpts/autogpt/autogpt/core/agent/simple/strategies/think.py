from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel
from logging import Logger
import uuid
import enum

from autogpt.core.configuration import SystemConfiguration, UserConfigurable

#prompting
from autogpt.core.prompting.base import (
    PlanningPromptStrategy,
    PromptStrategiesConfiguration,
)
from autogpt.core.prompting.schema import (
    LanguageModelClassification,
    ChatPrompt,
    ChatMessage
)
from autogpt.core.prompting.utils import json_loads, to_numbered_list, to_string_list


from autogpt.core.utils.json_schema import JSONSchema
from autogpt.core.resource.model_providers import (
    CompletionModelFunction,
    ChatMessage,
    AssistantChatMessageDict,
    OpenAIProvider
)

if TYPE_CHECKING:
    from autogpt.core.agent.simple import SimpleAgent


class ThinkStrategyFunctionNames(str, enum.Enum):
    THINK: str = "think"

class ThinkStrategyConfiguration(PromptStrategiesConfiguration):
    model_classification: LanguageModelClassification = (
        LanguageModelClassification.FAST_MODEL_16K
    )

DEFAULT_TRIGGERING_PROMPT = (
    "Determine exactly one command to use next based on the given goals "
    "and the progress you have made so far, "
    "and respond using the JSON schema specified previously:"
)

class ThinkStrategy(PlanningPromptStrategy):
    default_configuration = ThinkStrategyConfiguration()
    STRATEGY_NAME = "think"

    SYSTEM_PROMPT_INIT = DEFAULT_TRIGGERING_PROMPT

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
            
        self._functions = [
            ThinkStrategy.FUNCTION_THINK
        ]

    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification
    
    def build_prompt(
        self, 
        agent : SimpleAgent,
        instruction : str,
        thought_process_id,
        **kwargs
    ) -> ChatPrompt:
        
        # #self._provider : OpenAIProvider = kwargs['provider']
        model_name = kwargs["model_name"]

        #
        # STEP 1 : List all functions available
        # 

        if not instruction:
            raise ValueError("No instruction given")

        # Sysem message
        response_format_instr = self.response_format_instruction(agent= agent, thought_process_id = thought_process_id, model_name=model_name, )
        if response_format_instr:
            self._append_messages.append(ChatMessage.system(response_format_instr))

        # User Message
        instruction_msg = ChatMessage.user(instruction)
        instruction_tlength = agent._openai_provider.count_message_tokens(
            instruction_msg, model_name
        )

        self._append_messages: list[ChatMessage] = []

        messages : list[ChatMessage] = self.construct_base_prompt(
            agent = agent,
            thought_process_id= thought_process_id,
            append_messages=self._append_messages,
            reserve_tokens= instruction_tlength,
        )

        # ADD user input message ("triggering prompt")
        messages.append(instruction_msg)

        messages : list[ChatMessage] = agent._loop.on_before_think(
                                                    messages = messages,
                                                    thought_process_id = thought_process_id, 
                                                    instruction = instruction)

        return  ChatPrompt(
            messages= messages ,
            functions= self.get_functions(), #self._agent._tool_registry.dump_tools()
            function_call= ThinkStrategyFunctionNames.THINK,
            default_function_call='human_feedback'
            )


    # NOTE : based on autogpt agent.py 
    # This can be expanded to support multiple types of (inter)actions within an agent
    def response_format_instruction(self, agent : SimpleAgent , thought_process_id : str, model_name : str) -> str:
        if thought_process_id != "one-shot":
            raise NotImplementedError(f"Unknown thought process '{thought_process_id}'")

        RESPONSE_FORMAT_WITH_COMMAND = """```ts
        interface Response {
            thoughts: {
                // Thoughts
                text: string;
                reasoning: string;
                // Short markdown-style bullet list that conveys the long-term plan
                plan: string;
                // Constructive self-criticism
                criticism: string;
                // Summary of thoughts to say to the user
                speak: string;
            };
            command: {
                name: string;
                args: Record<string, any>;
            };
        }
        ```"""

        RESPONSE_FORMAT_WITHOUT_COMMAND = """```ts
        interface Response {
            thoughts: {
                // Thoughts
                text: string;
                reasoning: string;
                // Short markdown-style bullet list that conveys the long-term plan
                plan: string;
                // Constructive self-criticism
                criticism: string;
                // Summary of thoughts to say to the user
                speak: string;
            };
        }
        ```"""

        import re
        # use_functions : bool  = agent._openai_provider.has_function_call_api(model_name = self._model_classification)
        use_functions : bool  = agent._openai_provider.has_function_call_api(model_name=model_name)
        response_format :str  = re.sub(
            r"\n\s+",
            "\n",
            RESPONSE_FORMAT_WITHOUT_COMMAND
            if use_functions
            else RESPONSE_FORMAT_WITH_COMMAND,
        )

        use_functions = use_functions
        return (
            f"Respond strictly with JSON{', and also specify a command to use through a function_call' if use_functions else ''}. "
            "The JSON should be compatible with the TypeScript type `Response` from the following:\n"
            f"{response_format}"
        )
    
    # NOTE : based on planning_agent.py 
    def construct_base_prompt(
            self, 
            agent : SimpleAgent, 
            thought_process_id: str, 
            **kwargs
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

            messages = super().construct_base_prompt( agent = agent, 
                thought_process_id=thought_process_id, **kwargs
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
