import abc
from typing import Generic, TypeVar

from autogpt.core.configuration import SystemConfiguration

from .schema import (
    LanguageModelClassification,
    LanguageModelPrompt,
)


IN = TypeVar("IN", bound=dict)
OUT = TypeVar("OUT")


class PromptStrategy(abc.ABC, Generic[IN, OUT]):
    default_configuration: SystemConfiguration

    @property
    @abc.abstractmethod
    def model_classification(self) -> LanguageModelClassification:
        ...

    @abc.abstractmethod
    def build_prompt(self, *_, **kwargs: IN) -> LanguageModelPrompt:
        ...

    @abc.abstractmethod
    def parse_response_content(self, response_content: dict) -> OUT:
        ...
