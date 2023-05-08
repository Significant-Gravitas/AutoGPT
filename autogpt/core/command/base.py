import abc
import importlib
import inspect
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass
class CommandResult:
    ok: bool
    message: str


class Command(abc.ABC):
    """A class representing a command. Commands are actions which an agent can take.

    Attributes:
            name (str): The name of the command.
            description (str): A brief description of what the command does.
            signature (str): The signature of the function that theicommand executes. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        description: str,
        method: Callable[..., Any],
        signature: str = "",
        enabled: bool = True,
        disabled_reason: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.method = method
        self.signature = signature if signature else str(inspect.signature(self.method))
        self.enabled = enabled
        self.disabled_reason = disabled_reason

    def __call__(self, *args, **kwargs) -> CommandResult:
        if not self.enabled:
            return CommandResult(
                False, f"Command '{self.name}' is disabled: {self.disabled_reason}"
            )

        args, kwargs = self.__pre_call__(*args, **kwargs)

        command_result = self.method(*args, **kwargs)
        if isinstance(command_result, str):
            command_result = CommandResult(True, command_result)

        command_result = self.__post_hook__(command_result)
        return command_result

    def __pre_call__(self, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        # Return the unmodified *args and **kwargs by default, as a tuple and a dictionary
        return args, kwargs

    def __post_call__(self, command_result) -> CommandResult:
        # Return the unmodified command_result by default
        return command_result

    def __str__(self) -> str:
        return f"{self.name}: {self.description}, args: {self.signature}"


class CommandRegistry(abc.ABC):
    """
    The CommandRegistry class is a manager for a collection of Command objects.
    It allows the registration, and retrieval of Command objects,
    as well as the scanning and loading of command plugins from a specified
    directory.
    """

    AUTO_GPT_COMMAND_IDENTIFIER = "auto_gpt_command"

    def __init__(self):
        self.commands = {}

    def register_command(self, cmd: Command) -> None:
        self.commands[cmd.name] = cmd

    def list_commands(self) -> None:
        pass

    def get_command(self, command_name: str) -> Command:
        if command_name not in self.commands:
            raise KeyError(f"Command '{command_name}' not found in registry.")
        return self.commands[command_name]

    def execute_command(self, command_name: str, **kwargs) -> CommandResult:
        try:
            if command_name not in self.commands:
                raise KeyError(f"Command '{command_name}' not found in registry.")
            command = self.commands[command_name]

            command_result = command(**kwargs)
            return command_result

        except Exception as e:
            return CommandResult(False, f"Error: {str(e)}")

    def import_commands(self, module_name: str) -> None:
        """
        Imports the specified Python module containing command plugins.

        This method imports the associated module and registers any functions or
        classes that are decorated with the `AUTO_GPT_COMMAND_IDENTIFIER` attribute
        as `Command` objects. The registered `Command` objects are then added to the
        `commands` dictionary of the `CommandRegistry` object.

        Args:
            module_name (str): The name of the module to import for command plugins.
        """

        module = importlib.import_module(module_name)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            # Register decorated functions
            if hasattr(attr, self.AUTO_GPT_COMMAND_IDENTIFIER) and getattr(
                attr, self.AUTO_GPT_COMMAND_IDENTIFIER
            ):
                self.register(attr.command)
            # Register command classes
            elif (
                inspect.isclass(attr) and issubclass(attr, Command) and attr != Command
            ):
                cmd_instance = attr()
                self.register(cmd_instance)
