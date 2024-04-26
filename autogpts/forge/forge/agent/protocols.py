from abc import abstractmethod
from typing import TYPE_CHECKING, Iterator

from forge.agent.components import AgentComponent

if TYPE_CHECKING:
    from autogpts.autogpt.autogpt.agents.base import ThoughtProcessOutput

    from forge.chat_message import ChatMessage
    from forge.command.command import Command
    from forge.components.event_history.action_history import ActionResult


class DirectiveProvider(AgentComponent):
    def get_constraints(self) -> Iterator[str]:
        return iter([])

    def get_resources(self) -> Iterator[str]:
        return iter([])

    def get_best_practices(self) -> Iterator[str]:
        return iter([])


class CommandProvider(AgentComponent):
    @abstractmethod
    def get_commands(self) -> Iterator["Command"]:
        ...


class MessageProvider(AgentComponent):
    @abstractmethod
    def get_messages(self) -> Iterator["ChatMessage"]:
        ...


class AfterParse(AgentComponent):
    @abstractmethod
    def after_parse(self, result: "ThoughtProcessOutput") -> None:
        ...


class ExecutionFailure(AgentComponent):
    @abstractmethod
    def execution_failure(self, error: Exception) -> None:
        ...


class AfterExecute(AgentComponent):
    @abstractmethod
    def after_execute(self, result: "ActionResult") -> None:
        ...
