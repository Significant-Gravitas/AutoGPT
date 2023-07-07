import logging
from abc import ABC
from dataclasses import InitVar, dataclass
from typing import Any, Callable, Optional, Type

from autogpt.config import Config
from common.common import simple_exception_handling

from .command_parameter import CommandParameter


class GenericCommand(ABC):
    # also has name
    # @property
    # @abstractmethod
    # def name(self):
    #     ... (should be fixed)

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

    def generate_instance(self, arguments, agent=None):
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
        method: Optional[Callable[..., Any]],
        parameters: list[CommandParameter],
        enabled: bool | Callable[[Config], bool] = True,
        disabled_reason: Optional[str] = None,
        instancecls: Optional[
            Type["CommandInstance"]
        ] = None,  # not really needed right now
        max_seen_to_stop: Optional[int] = None,
        stop_if_looped: Optional[bool] = True,
    ):
        self.name = name
        self.description = description
        self.method = method
        self.parameters = parameters
        self.enabled = enabled
        self.disabled_reason = disabled_reason
        self.instancecls = CommandInstance if instancecls is None else instancecls
        self.max_seen_to_stop = max_seen_to_stop
        self.stop_if_looped = stop_if_looped

    def __call__(self, *args, **kwargs) -> Any:
        self.execute(*args, **kwargs)

    @simple_exception_handling(lambda_on_exc=lambda: VoidCommandInstance())
    def generate_instance(self, arguments, agent=None):
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


class GenericCommandInstance:
    pass


@dataclass
class CommandInstance(GenericCommandInstance):
    command: GenericCommand
    arguments: dict
    agent: InitVar = None
    """
    A class that represent a instance of command to execute with concrete parameters. 
    This function has also access to agent
    """

    def __hash__(self):
        assert self._argfixed
        if self._hash is None:
            self._hash = self.command.calculate_hash(**self.arguments)
        return self._hash

    def __post_init__(self, agent):
        self._hash = None
        self.agent = agent
        self._argfixed = False
        if self.agent:
            try:
                self._resolve_pathlike_command_args()

                # We dont want the hash to be ambigous, and it is only calculated once so arguments should be frozen.
            except:
                logging.error("error in resolving pathlike command args {self}")
        self._argfixed = True

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
        return self.command.execute(**self.arguments, agent=self.agent)

    @property
    def name(self):
        return self.command.name

    def __str__(self):
        return f"<inst of {self.name} -  {hex(hash(self))}>"

    @property
    def is_void(self):
        return False

    @property
    def should_ignore(self) -> bool:
        if self.agent is None:
            return False
        return self.name in self.agent.config.commands_to_ignore

    @property
    def should_stop(self) -> bool:
        if self.agent is None:
            return False
        return self.name in self.agent.config.commands_to_stop


class VoidCommandInstance(GenericCommandInstance):
    CMD = Command(name="", description="empty command", method=None, parameters=[])

    @property
    def name(self):
        return ""

    @property
    def command(self):
        return self.CMD

    @property
    def arguments(self):
        return {}

    @property
    def is_void(self):
        return True

    @property
    def should_ignore(self) -> bool:
        return False

    @property
    def should_stop(self) -> bool:
        return True

    def __hash__(self) -> int:
        return 0
