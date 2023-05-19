import functools
import importlib
import inspect
from typing import Any, Callable, Optional, Union

from autogpt.logs import logger

# Unique identifier for auto-gpt commands
AUTO_GPT_COMMAND_IDENTIFIER = "auto_gpt_command"


class Command:
    """A class representing a command.

    Attributes:
        name (str): The name of the command.
        description (str): A brief description of what the command does.
        signature (str): The signature of the function that the command executes. Defaults to None.
    """

    def __init__(
        self,
        name: str,
        description: str,
        method: Callable[..., Any],
        signature: str = "",
        enabled: bool = True,
        disabled_reason: Optional[str] = None,
        name_alias: Optional[Union[str, list]] = None,
    ):
        self.name = name
        self.description = description
        self.method = method
        self.signature = signature if signature else str(inspect.signature(self.method))
        self.enabled = enabled
        self.disabled_reason = disabled_reason
        if name_alias and isinstance(name_alias, str):
            self.name_alias = [name_alias]
        elif name_alias and isinstance(name_alias, list):
            self.name_alias = name_alias
        else:
            self.name_alias = []

    def __call__(self, *args, **kwargs) -> Any:
        if not self.enabled:
            return f"Command '{self.name}' is disabled: {self.disabled_reason}"
        return self.method(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.name}: {self.description}, args: {self.signature}"


class CommandRegistry:
    """
    The CommandRegistry class is a manager for a collection of Command objects.
    It allows the registration, modification, and retrieval of Command objects,
    as well as the scanning and loading of command plugins from a specified
    directory.
    """

    def __init__(self):
        self.commands = {}

    def _import_module(self, module_name: str) -> Any:
        return importlib.import_module(module_name)

    def _reload_module(self, module: Any) -> Any:
        return importlib.reload(module)

    def register(self, cmd: Command) -> None:
        if cmd.name in self.commands:
            logger.warn(
                f"Command '{cmd.name}' already registered and will be overwritten!"
            )
        self.commands[cmd.name] = cmd
        for command_name_alias in cmd.name_alias:
            if command_name_alias == cmd.name:
                continue
            self.commands[command_name_alias] = cmd

    def unregister(self, command_name: str):
        if command_name in self.commands:
            cmd = self.commands[command_name]
            for command_name_alias in cmd.name_alias:
                if command_name_alias not in self.commands:
                    continue
                del self.commands[command_name_alias]
            del self.commands[command_name]
        else:
            raise KeyError(f"Command '{command_name}' not found in registry.")

    def reload_commands(self) -> None:
        """Reloads all loaded command plugins."""
        for command_name in self.commands:
            cmd = self.commands[command_name]
            module = self._import_module(cmd.__module__)
            reloaded_module = self._reload_module(module)
            if hasattr(reloaded_module, "register"):
                reloaded_module.register(self)

    def get_command(self, command_name: str) -> Callable[..., Any]:
        return self.commands[command_name]

    def call(self, command_name: str, **kwargs) -> Any:
        if command_name not in self.commands:
            raise KeyError(f"Command '{command_name}' not found in registry.")
        cmd = self.commands[command_name]
        return cmd(**kwargs)

    def command_prompt(self) -> str:
        """
        Returns a string representation of all registered `Command` objects for use in a prompt
        """
        commands_list = [
            f"{idx + 1}. {str(cmd)}" for idx, cmd in enumerate(self.commands.values())
        ]
        return "\n".join(commands_list)

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
            if hasattr(attr, AUTO_GPT_COMMAND_IDENTIFIER) and getattr(
                attr, AUTO_GPT_COMMAND_IDENTIFIER
            ):
                self.register(attr.command)
            # Register command classes
            elif (
                inspect.isclass(attr) and issubclass(attr, Command) and attr != Command
            ):
                cmd_instance = attr()
                self.register(cmd_instance)


def command(
    name: str,
    description: str,
    signature: str = "",
    enabled: bool = True,
    disabled_reason: Optional[str] = None,
    name_alias: Optional[Union[str, list]] = None,
) -> Callable[..., Any]:
    """The command decorator is used to create Command objects from ordinary functions."""

    if not enabled:
        if disabled_reason is not None:
            logger.debug(f"Command '{name}' is disabled: {disabled_reason}")
        return lambda func: func

    def decorator(func: Callable[..., Any]) -> Command:
        cmd = Command(
            name=name,
            description=description,
            method=func,
            signature=signature,
            enabled=enabled,
            disabled_reason=disabled_reason,
            name_alias=name_alias,
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)

        wrapper.command = cmd

        setattr(wrapper, AUTO_GPT_COMMAND_IDENTIFIER, True)

        return wrapper

    return decorator
