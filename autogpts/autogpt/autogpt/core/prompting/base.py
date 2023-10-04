from __future__ import annotations
import abc

from pydantic import validator
from typing import TYPE_CHECKING, Union
from autogpt.core.utils.json_schema import JSONSchema

if TYPE_CHECKING:
    from autogpt.core.agents.simple.main import PlannerAgent
    from autogpt.core.agents.base.main import BaseAgent

from autogpt.core.prompting.utils.utils import json_loads, to_numbered_list
from autogpt.core.configuration import SystemConfiguration
from autogpt.core.prompting.schema import (
    LanguageModelClassification,
    CompletionModelFunction,
)

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
)


RESPONSE_SCHEMA = JSONSchema(
    type=JSONSchema.Type.OBJECT,
    properties={
        "thoughts": JSONSchema(
            type=JSONSchema.Type.OBJECT,
            required=True,
            properties={
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



class PromptStrategy(abc.ABC):
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


class BasePromptStrategy(PromptStrategy):
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

