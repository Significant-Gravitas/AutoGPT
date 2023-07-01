import logging
from abc import ABC
from dataclasses import InitVar, dataclass, field
from typing import Any, Callable, Optional

from autogpt.config import Config

from .command_parameter import CommandParameter


class GenericCommand(ABC):
    # also has name
    def execute(self, *args, **kwargs) -> Any:
        ...

    def generate_instance(self, arguments, agent=None):
        ...


@dataclass
class PromptCommand(GenericCommand):
    label: str
    name: str
    args: str
    function: Callable[..., Any]

    def execute(self, *args, **kwargs) -> Any:
        if "agent" in kwargs:
            kwargs.pop("agent")
        return self.function(*args, **kwargs)

    def get_instance(self, arguments, agent=None):
        return CommandInstance(self, arguments, agent=agent)


class Command(GenericCommand):
    """A class representing a command.

    Attributes:
        name (str): The name of the command.
        description (str): A brief description of what the command does.
        parameters (list): The parameters of the function that the command executes.
    """

    METHOD = None

    def __init__(
        self,
        name: str,
        description: str,
        method: Callable[..., Any],
        parameters: list[CommandParameter],
        enabled: bool | Callable[[Config], bool] = True,
        disabled_reason: Optional[str] = None,
        instancecls: Optional[
            "CommandInstance"
        ] = "CommandInstance",  # not really needed right now
        max_seen_to_stop: Optional[int] = None,
        stop_if_looped: Optional[bool] = True,
    ):
        self.name = name
        self.description = description
        self.method = method
        self.parameters = parameters
        self.enabled = enabled
        self.disabled_reason = disabled_reason
        self.instancecls = instancecls
        self.max_seen_to_stop = max_seen_to_stop
        self.stop_if_looped = stop_if_looped

    def __call__(self, *args, **kwargs) -> Any:
        self.execute(*args, **kwargs)

    def get_instance(self, arguments, agent=None):
        return self.instancecls(self, arguments, agent=agent)

    def execute(self, *args, **kwargs):
        if hasattr(kwargs, "config") and callable(self.enabled):
            self.enabled = self.enabled(kwargs["config"])
        if not self.enabled:
            if self.disabled_reason:
                return f"Command '{self.name}' is disabled: {self.disabled_reason}"
            return f"Command '{self.name}' is disabled"
        if self.method is None:
            return self.METHOD(self, *args, **kwargs)  # a class Variable
        return self.method(*args, **kwargs)

    def __str__(self) -> str:
        params = [
            f"{param.name}: {param.type if param.required else f'Optional[{param.type}]'}"
            for param in self.parameters
        ]
        return f"{self.name}: {self.description}, params: ({', '.join(params)})"

    # override this
    def calculate_hash(self, **kwargs):
        from frozendict import frozendict

        return hash((self.name, frozendict(**kwargs)))


@dataclass
class CommandInstance:
    command: GenericCommand
    arguments: dict
    agent: InitVar = None

    def __hash__(self):
        if self._hash is None:
            self._hash = self.command.calculate_hash(
                self.arguments
            )  # should be calulcated before applying path
        return self._hash

    def __post_init__(self, agent):
        self._hash = None
        self._agent = agent

    def _resolve_pathlike_command_args(self):
        if "directory" in self.arguments and self.arguments["directory"] in {"", "/"}:
            self.arguments["directory"] = str(self.agent.workspace.root)
        else:
            for pathlike in ["filename", "directory", "clone_path"]:
                if pathlike in self.arguments:
                    self.arguments[pathlike] = str(
                        self.agent.workspace.get_path(self.arguments[pathlike])
                    )

    def execute(self):
        try:
            if self.agent:
                self._resolve_pathlike_command_args()
        except:
            logging.error("error in resolving pathlike command args")
            raise
        return self.command.execute(**self.arguments, agent=self._agent)

    @property
    def name(self):
        return self.command.name

    def __str__(self):
        return self.name
