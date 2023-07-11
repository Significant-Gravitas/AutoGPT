import importlib
import inspect
from typing import Any

from autogpt.command_decorator import AUTO_GPT_COMMAND_IDENTIFIER
from autogpt.logs import logger
from autogpt.models.command import Command


class CommandRegistry:
    """
    The CommandRegistry class is a manager for a collection of Command objects.
    It allows the registration, modification, and retrieval of Command objects,
    as well as the scanning and loading of command plugins from a specified
    directory.
    """

    commands: dict[str, Command] = {}
    commands_aliases: dict[str, Command] = {}

    def __contains__(self, command_name: str):
        return command_name in self.commands or command_name in self.commands_aliases

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

        if cmd.name in self.commands_aliases:
            logger.warn(
                f"Command '{cmd.name}' will overwrite alias with the same name of "
                f"'{self.commands_aliases[cmd.name]}'!"
            )
        for alias in cmd.aliases:
            self.commands_aliases[alias] = cmd

    def unregister(self, command: Command) -> None:
        if command.name in self.commands:
            del self.commands[command.name]
            for alias in command.aliases:
                del self.commands_aliases[alias]
        else:
            raise KeyError(f"Command '{command.name}' not found in registry.")

    def reload_commands(self) -> None:
        """Reloads all loaded command plugins."""
        for cmd_name in self.commands:
            cmd = self.commands[cmd_name]
            module = self._import_module(cmd.__module__)
            reloaded_module = self._reload_module(module)
            if hasattr(reloaded_module, "register"):
                reloaded_module.register(self)

    def get_command(self, name: str) -> Command | None:
        if name in self.commands:
            return self.commands[name]

        if name in self.commands_aliases:
            return self.commands_aliases[name]

    def call(self, command_name: str, **kwargs) -> Any:
        if command := self.get_command(command_name):
            return command(**kwargs)
        raise KeyError(f"Command '{command_name}' not found in registry")

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
