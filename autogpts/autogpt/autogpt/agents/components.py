from typing import Any, Iterator, Optional, Protocol, Type, TypeVar, cast, runtime_checkable
from dataclasses import dataclass

from autogpt.core.prompting.schema import ChatPrompt
from autogpt.models.command import Command
from autogpt.core.resource.model_providers.schema import AssistantChatMessage, ChatMessage

class Component():
    run_after: list[type['Component']] = []

    @classmethod
    def get_dependencies(cls) -> list[Type['Component']]:
        return cls.run_after

class ComponentError(Exception):
    pass

class PipelineError(ComponentError):
    pass

class ComponentSystemError(ComponentError):
    pass

@runtime_checkable
class BuildPrompt(Protocol):
    @dataclass
    class Result:
        extra_messages: list[ChatMessage] = []
        extra_commands: list = []

    def build_prompt(self, result: Result) -> None:
        ...

    #TODO move to separate protocol
    def get_prompt(self, result: Result, prompt: ChatPrompt) -> None:
        pass
    

@runtime_checkable
class ProposeAction(Protocol):
    @dataclass
    class Result:
        command_name: str
        command_args: Any
        thoughts: Any

    def propose_action(self, result: Result) -> None:
        ...


@runtime_checkable
class CommandProvider(Protocol):
    def get_commands(self) -> Iterator[Command]:
        ...


# @runtime_checkable
# class ResponseHandler(Protocol):
#     def parse_process_response(self, llm_repsonse: AssistantChatMessage):
#         ...


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