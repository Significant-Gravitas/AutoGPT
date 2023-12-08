from __future__ import annotations

import abc
import re
from typing import TYPE_CHECKING, Optional

from AFAAS.app.core.utils.json_schema import JSONSchema

if TYPE_CHECKING:
    from AFAAS.app.core.agents.planner.main import PlannerAgent

from AFAAS.app.core.configuration import SystemConfiguration
from AFAAS.app.core.prompting.schema import \
    LanguageModelClassification
from AFAAS.app.core.prompting.utils.utils import json_loads
from AFAAS.app.core.resource.model_providers import (
    AbstractLanguageModelProvider, AssistantChatMessageDict, ChatModelResponse,
    ChatPrompt, CompletionModelFunction)

RESPONSE_SCHEMA = JSONSchema(
    type=JSONSchema.Type.OBJECT,
    properties={
        "thoughts": JSONSchema(
            type=JSONSchema.Type.OBJECT,
            required=True,
            properties={
                "limits": JSONSchema(
                    description="Express your limitations (Context Limitation, Token Limitation, Cognitive Limitation) if you were an autonomous program hosted on a server and relying on a Large Language Model to take decision",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "overcome_limit": JSONSchema(
                    description="How you woud overcome this limit (if any)",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "reasoning": JSONSchema(
                    description="Your process of thoughts",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "criticism": JSONSchema(
                    description="Constructive self-criticism of your process of thoughts",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "plan": JSONSchema(
                    description="Short markdown-style bullet list that conveys your plan",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "self_feedback": JSONSchema(
                    description="if you were to do it again what would you told yourself",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
            },
        ),
        "command": JSONSchema(
            type=JSONSchema.Type.OBJECT,
            required=True,
            properties={
                "name": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "args": JSONSchema(
                    type=JSONSchema.Type.OBJECT,
                    required=True,
                ),
            },
        ),
    },
)


class DefaultParsedResponse(dict):
    id: str
    type: str
    command_name: str
    command_args: dict
    assistant_reply_dict: dict


class PromptStrategiesConfiguration(SystemConfiguration):
    temperature: float
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    frequency_penalty: Optional[float] = None  # Avoid repeting oneselfif coding 0.3
    presence_penalty: Optional[float] = None  # Avoid certain subjects


from AFAAS.app.core.agents.base.features.agentmixin import \
    AgentMixin


class AbstractPromptStrategy(AgentMixin, abc.ABC):
    STRATEGY_NAME: str
    default_configuration: PromptStrategiesConfiguration

    @property
    @abc.abstractmethod
    def model_classification(self) -> LanguageModelClassification:
        ...

    @abc.abstractmethod
    def build_prompt(self, *_, **kwargs) -> ChatPrompt:
        ...

    @abc.abstractmethod
    def parse_response_content(self, response_content: AssistantChatMessageDict):
        ...

    @abc.abstractmethod
    def set_tools(self, **kwargs):
        ...


class BasePromptStrategy(AbstractPromptStrategy):
    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    # TODO : This implementation is shit :)
    def get_tools(self) -> list[CompletionModelFunction]:
        """
        Returns a list of functions related to refining user context.

        Returns:
            list: A list of CompletionModelFunction objects detailing each function's purpose and parameters.

        Example:
            >>> strategy = RefineUserContextStrategy(...)
            >>> functions = strategy.get_functions()
            >>> print(functions[0].name)
            'refine_requirements'
        """
        return self._tools

    def get_tools_names(self) -> list[str]:
        """
        Returns a list of names of functions related to refining user context.

        Returns:
            list: A list of strings, each representing the name of a function.

        Example:
            >>> strategy = RefineUserContextStrategy(...)
            >>> function_names = strategy.get_tools_names()
            >>> print(function_names)
            ['refine_requirements']
        """
        return [item.name for item in self._tools]

    # This can be expanded to support multiple types of (inter)actions within an agent
    @abc.abstractmethod
    def response_format_instruction(
        self, language_model_provider: AbstractLanguageModelProvider, model_name: str
    ) -> str:
        use_oa_tools_api = language_model_provider.has_oa_tool_calls_api(
            model_name=model_name
        )

        response_schema = RESPONSE_SCHEMA.copy(deep=True)
        if (
            use_oa_tools_api
            and response_schema.properties
            and "command" in response_schema.properties
        ):
            del response_schema.properties["command"]

        # Unindent for performance
        response_format: str = re.sub(
            r"\n\s+",
            "\n",
            response_schema.to_typescript_object_interface("Response"),
        )

        if use_oa_tools_api:
            return (
                f"Respond strictly with a JSON of type `Response` :\n"
                f"{response_format}"
            )

        return (
            f"Respond strictly with JSON{', and also specify a command to use through a tool_calls' if use_oa_tools_api else ''}. "
            "The JSON should be compatible with the TypeScript type `Response` from the following:\n"
            f"{response_format}"
        )

    ###
    ### parse_response_content
    ###
    def default_parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> list[DefaultParsedResponse]:
        """Parse the actual text response from the objective model.

        Args:
            response_content: The raw response content from the objective model.

        Returns:
            The parsed response.

        """
        assistant_return_list: list[DefaultParsedResponse] = []
        assistant_reply_dict = response_content["content"]

        for tool in response_content["tool_calls"]:
            try:
                command_args = json_loads(tool["function"]["arguments"])
            except Exception:
                self._agent._logger.warning(command_args)

            ###
            ### NEW
            ###
            command_name = tool["function"]["name"]

            assistant_return_list.append(
                DefaultParsedResponse(
                    id=tool["id"],
                    type=tool["type"],
                    command_name=command_name,
                    command_args=command_args,
                    assistant_reply_dict=assistant_reply_dict,
                )
            )

        return assistant_return_list

    @staticmethod
    def get_autocorrection_response(response: ChatModelResponse):
        return response.parsed_result[0]["command_args"]["note_to_agent"]
