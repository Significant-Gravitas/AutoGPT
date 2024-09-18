import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from forge.llm.providers import AssistantChatMessage

from .schema import ChatPrompt, LanguageModelClassification


class PromptStrategy(abc.ABC):
    @property
    @abc.abstractmethod
    def llm_classification(self) -> LanguageModelClassification:
        ...

    @abc.abstractmethod
    def build_prompt(self, *_, **kwargs) -> ChatPrompt:
        ...

    @abc.abstractmethod
    def parse_response_content(self, response: "AssistantChatMessage") -> Any:
        ...
