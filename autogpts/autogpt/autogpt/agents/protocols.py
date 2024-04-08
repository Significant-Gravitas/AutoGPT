from abc import abstractmethod
from typing import Iterator

from autogpt.agents.base import ThoughtProcessOutput
from autogpt.agents.components import AgentComponent, Single
from autogpt.config.ai_directives import AIDirectives
from autogpt.config.ai_profile import AIProfile
from autogpt.core.prompting.schema import ChatPrompt
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    ChatMessage,
)
from autogpt.models.action_history import ActionResult
from autogpt.models.command import Command


class DirectiveProvider(AgentComponent):
    def get_contraints(self) -> Iterator[str]:
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


class BuildPrompt(AgentComponent):
    @abstractmethod
    def build_prompt(
        self,
        messages: list[ChatMessage],
        commands: list[Command],
        task: str,
        profile: AIProfile,
        directives: AIDirectives,
    ) -> Single[ChatPrompt]:
        ...


class ParseResponse(AgentComponent):
    @abstractmethod
    def parse_response(
        self, result: ThoughtProcessOutput, response: AssistantChatMessage
    ) -> Single[ThoughtProcessOutput]:
        ...


class AfterParse(AgentComponent):
    @abstractmethod
    def after_parsing(self, result: ThoughtProcessOutput) -> None:
        ...


class AfterExecute(AgentComponent):
    @abstractmethod
    def after_execution(self, result: ActionResult) -> None:
        ...
