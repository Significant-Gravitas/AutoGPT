from typing import Iterator, Protocol, runtime_checkable

from autogpt.agents.base import ThoughtProcessOutput
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    ChatMessage,
)
from autogpt.models.command import Command
from autogpt.agents.components import Single
from autogpt.models.action_history import ActionResult
from autogpt.config.ai_directives import AIDirectives
from autogpt.config.ai_profile import AIProfile
from autogpt.core.prompting.schema import ChatPrompt


@runtime_checkable
class DirectivesProvider(Protocol):
    def get_contraints(self) -> Iterator[str]:
        return iter([])

    def get_resources(self) -> Iterator[str]:
        return iter([])

    def get_best_practices(self) -> Iterator[str]:
        return iter([])


@runtime_checkable
class CommandProvider(Protocol):
    def get_commands(self) -> Iterator[Command]: ...


@runtime_checkable
class MessageProvider(Protocol):
    def get_messages(self) -> Iterator[ChatMessage]: ...


@runtime_checkable
class BuildPrompt(Protocol):
    def build_prompt(
        self,
        messages: list[ChatMessage],
        commands: list[Command],
        task: str,
        profile: AIProfile,
        directives: AIDirectives,
    ) -> Single[ChatPrompt]: ...


@runtime_checkable
class ParseResponse(Protocol):
    def parse_response(
        self, result: ThoughtProcessOutput, response: AssistantChatMessage
    ) -> Single[ThoughtProcessOutput]: ...


@runtime_checkable
class AfterParsing(Protocol):
    def after_parsing(self, result: ThoughtProcessOutput) -> None: ...


@runtime_checkable
class AfterExecution(Protocol):
    def after_execution(self, result: ActionResult) -> None: ...
