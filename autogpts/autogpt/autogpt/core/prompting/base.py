from __future__ import annotations
import abc

from pydantic import validator
import re
from typing import TYPE_CHECKING, Union
from autogpt.core.utils.json_schema import JSONSchema

if TYPE_CHECKING:
    from autogpt.core.agents.simple.main import PlannerAgent
    from autogpt.core.agents.base.main import BaseAgent

from autogpt.core.prompting.utils.utils import json_loads, to_numbered_list
from autogpt.core.configuration import SystemConfiguration
from autogpt.core.prompting.schema import LanguageModelClassification


from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)

from autogpt.core.resource.model_providers import (
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
                    description="Briefly express your limitations as an Agent that is hosted on a server and interact with a LLM, which has specific limitations (Context Limitation, Token Limitation, Cognitive Limitation)",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "text": JSONSchema(
                    description="Thoughts",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "reasoning": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "plan": JSONSchema(
                    description="Short markdown-style bullet list that conveys the long-term plan",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "criticism": JSONSchema(
                    description="Constructive self-criticism",
                    type=JSONSchema.Type.STRING,
                    required=True,
                ),
                "speak": JSONSchema(
                    description="Summary of thoughts, to say to user",
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
    pass



from autogpt.core.agents.base.features.agentmixin import AgentMixin
class AbstractPromptStrategy(AgentMixin, abc.ABC):
    STRATEGY_NAME: str
    default_configuration: SystemConfiguration

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
    def response_format_instruction(
        self, agent: PlannerAgent,  model_name: str) -> str:

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

