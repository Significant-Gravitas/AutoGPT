from typing import Any, Optional, Protocol, Type, TypeVar, cast, runtime_checkable
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

    # pass result to the next module
    def build_prompt(self, result: Result) -> Result:
        ... # Just build the prompt data

    def get_prompt(self, result: Result, prompt: ChatPrompt) -> ChatPrompt:
        return prompt
    

@runtime_checkable
class ProposeAction(Protocol):
    @dataclass
    class Result:
        command_name: str
        command_args: Any
        thoughts: Any

    def propose_action(self, result: Result) -> Result:
        ...


@runtime_checkable
class CommandProvider(Protocol):
    def get_commands(self, commands: list[Command]) -> list[Command]:
        ...


@runtime_checkable
class ResponseHandler(Protocol):
    def parse_process_response(self, llm_repsonse: AssistantChatMessage):
        ...