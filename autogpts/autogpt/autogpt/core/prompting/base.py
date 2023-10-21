from __future__ import annotations
import abc

from pydantic import validator
import re
from typing import TYPE_CHECKING, Union, Optional
from autogpts.autogpt.autogpt.core.utils.json_schema import JSONSchema

if TYPE_CHECKING:
    from autogpts.autogpt.autogpt.core.agents.planner.main import PlannerAgent
    from autogpts.autogpt.autogpt.core.agents.base.main import BaseAgent

from autogpts.autogpt.autogpt.core.prompting.utils.utils import json_loads, to_numbered_list
from autogpts.autogpt.autogpt.core.configuration import SystemConfiguration
from autogpts.autogpt.autogpt.core.prompting.schema import LanguageModelClassification


from autogpts.autogpt.autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)

from autogpts.autogpt.autogpt.core.resource.model_providers import (
    BaseChatModelProvider,
    ModelProviderName,
    OpenAIModelName,
    AssistantChatMessageDict,
    ChatMessage,
    ChatPrompt,
    CompletionModelFunction,
)


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


class PromptStrategiesConfiguration(SystemConfiguration):
    temperature : float = 0.9 #if coding 0.05
    top_p: Optional[float] = None,
    max_tokens : Optional[int] = None,
    frequency_penalty: Optional[float] = None # Avoid repeting oneselfif coding 0.3
    presence_penalty : Optional[float] = None # Avoid certain subjects


from autogpts.autogpt.autogpt.core.agents.base.features.agentmixin import AgentMixin


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


class BasePromptStrategy(AbstractPromptStrategy):
    @property
    def model_classification(self) -> LanguageModelClassification:
        return self._model_classification

    # TODO : This implementation is shit :)
    def get_functions(self) -> list[CompletionModelFunction]:
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
        return self._functions

    # TODO : This implementation is shit :)
    def get_functions_names(self) -> list[str]:
        """
        Returns a list of names of functions related to refining user context.

        Returns:
            list: A list of strings, each representing the name of a function.

        Example:
            >>> strategy = RefineUserContextStrategy(...)
            >>> function_names = strategy.get_functions_names()
            >>> print(function_names)
            ['refine_requirements']
        """
        return [item.name for item in self._functions]

    # NOTE : based on autogpt agent.py
    # This can be expanded to support multiple types of (inter)actions within an agent
    def response_format_instruction(self, agent: PlannerAgent, model_name: str) -> str:
        use_functions_api = agent._chat_model_provider.has_function_call_api(
            model_name=model_name
        )

        response_schema = RESPONSE_SCHEMA.copy(deep=True)
        if (
            use_functions_api
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

        return (
            f"Respond strictly with JSON{', and also specify a command to use through a function_call' if use_functions_api else ''}. "
            "The JSON should be compatible with the TypeScript type `Response` from the following:\n"
            f"{response_format}"
        )
    

    ###
    ### parse_response_content
    ###
    def default_parse_response_content(
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

        ###
        ### NEW
        ###
        command_name = response_content["function_call"]["name"]
        command_args = parsed_response
        assistant_reply_dict = response_content["content"]

        return command_name, command_args, assistant_reply_dict

