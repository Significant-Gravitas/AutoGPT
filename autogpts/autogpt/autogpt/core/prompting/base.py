import abc
from typing import Generic, TypedDict, TypeVar#, Unpack

from autogpt.core.configuration import SystemConfiguration
from autogpt.core.resource.model_providers import AssistantChatMessageDict

from .schema import ChatPrompt, LanguageModelClassification

IN = TypeVar("IN", bound=TypedDict)
OUT = TypeVar("OUT")


class PromptStrategy(abc.ABC, Generic[IN, OUT]):
    default_configuration: SystemConfiguration

    @property
    @abc.abstractmethod
    def model_classification(self) -> LanguageModelClassification:
        ...

    @abc.abstractmethod
    def build_prompt(self, *_, **kwargs: [IN]) -> ChatPrompt:
        ...

    @abc.abstractmethod
    def parse_response_content(self, response_content: AssistantChatMessageDict) -> OUT:
        ...
