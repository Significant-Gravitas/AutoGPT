from typing import Iterator, Protocol, runtime_checkable

from autogpt.agents.base import ThoughtProcessOutput
from autogpt.core.prompting.schema import ChatPrompt
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    ChatMessage,
)
from autogpt.models.command import Command



@runtime_checkable
class BuildPrompt(Protocol):
    def build_prompt(
        self,
        messages: list[ChatMessage],
        commands: list[Command],
        prompt: ChatPrompt,
    ) -> ChatPrompt:
        ...


@runtime_checkable
class MessageProvider(Protocol):
    def get_messages(self) -> Iterator[ChatMessage]:
        ...


@runtime_checkable
class ProposeAction(Protocol):
    def propose_action(self, result: ThoughtProcessOutput) -> None:
        ...


@runtime_checkable
class CommandProvider(Protocol):
    def get_commands(self) -> Iterator[Command]:
        ...


@runtime_checkable
class ParseResponse(Protocol):
    def parse_response(
        self, result: ThoughtProcessOutput, llm_response: AssistantChatMessage
    ) -> ThoughtProcessOutput:
        ...


@runtime_checkable
class GuidelinesProvider(Protocol):
    def get_contraints(self) -> list[str]:
        ...

    def get_resources(self) -> list[str]:
        ...

    def get_best_practices(self) -> list[str]:
        ...


@runtime_checkable
class OnExecute(Protocol):
    def on_execute(self) -> None:
        ...
