from abc import abstractmethod
from typing import Iterator, Protocol, runtime_checkable

from autogpt.agents.base import ThoughtProcessOutput
from autogpt.agents.components import AgentComponent
from autogpt.core.prompting.schema import ChatPrompt
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    ChatMessage,
)
from autogpt.models.action_history import ActionResult
from autogpt.models.command import Command


class DirectiveProvider(AgentComponent):
    def get_constraints(self) -> Iterator[str]:
        return iter([])

    def get_resources(self) -> Iterator[str]:
        return iter([])

    def get_best_practices(self) -> Iterator[str]:
        return iter([])


class CommandProvider(AgentComponent):
    @abstractmethod
    def get_commands(self) -> Iterator[Command]:
        ...


class MessageProvider(AgentComponent):
    @abstractmethod
    def get_messages(self) -> Iterator[ChatMessage]:
        ...


@runtime_checkable
class PromptStrategy(Protocol):
    def build_prompt(
        self,
        **kwargs
    ) -> ChatPrompt:
        ...

    def parse_response(
        self, response: AssistantChatMessage
    ) -> ThoughtProcessOutput:
        ...


class AfterParse(AgentComponent):
    @abstractmethod
    def after_parsing(self, result: ThoughtProcessOutput) -> None:
        ...


class AfterExecute(AgentComponent):
    @abstractmethod
    def after_execution(self, result: ActionResult) -> None:
        ...
