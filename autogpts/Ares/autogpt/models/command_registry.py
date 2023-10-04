from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from autogpt.config import Config

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

    commands: dict[str, Command]
    commands_aliases: dict[str, Command]

    # Alternative way to structure the registry; currently redundant with self.commands
    categories: dict[str, CommandCategory]

    @dataclass
    class CommandCategory:
        name: str
        title: str
        description: str
        commands: list[Command] = field(default_factory=list[Command])
        modules: list[ModuleType] = field(default_factory=list[ModuleType])

    def __init__(self):
        self.commands = {}
        self.commands_aliases = {}
        self.categories = {}

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

    @staticmethod
    def with_command_modules(modules: list[str], config: Config) -> CommandRegistry:
        new_registry = CommandRegistry()

        logger.debug(
            f"The following command categories are disabled: {config.disabled_command_categories}"
        )
        enabled_command_modules = [
            x for x in modules if x not in config.disabled_command_categories
        ]

        logger.debug(
            f"The following command categories are enabled: {enabled_command_modules}"
        )

        for command_module in enabled_command_modules:
            new_registry.import_command_module(command_module)

        # Unregister commands that are incompatible with the current config
        incompatible_commands: list[Command] = []
        for command in new_registry.commands.values():
            if callable(command.enabled) and not command.enabled(config):
                command.enabled = False
                incompatible_commands.append(command)

        for command in incompatible_commands:
            new_registry.unregister(command)
            logger.debug(
                f"Unregistering incompatible command: {command.name}, "
                f"reason - {command.disabled_reason or 'Disabled by current config.'}"
            )

        return new_registry

    def import_command_module(self, module_name: str) -> None:
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

        category = self.register_module_category(module)

        for attr_name in dir(module):
            attr = getattr(module, attr_name)

            command = None

            # Register decorated functions
            if getattr(attr, AUTO_GPT_COMMAND_IDENTIFIER, False):
                command = attr.command

            # Register command classes
            elif (
                inspect.isclass(attr) and issubclass(attr, Command) and attr != Command
            ):
                command = attr()

            if command:
                self.register(command)
                category.commands.append(command)

    def register_module_category(self, module: ModuleType) -> CommandCategory:
        if not (category_name := getattr(module, "COMMAND_CATEGORY", None)):
            raise ValueError(f"Cannot import invalid command module {module.__name__}")

        if category_name not in self.categories:
            self.categories[category_name] = CommandRegistry.CommandCategory(
                name=category_name,
                title=getattr(
                    module, "COMMAND_CATEGORY_TITLE", category_name.capitalize()
                ),
                description=getattr(module, "__doc__", ""),
            )

        category = self.categories[category_name]
        if module not in category.modules:
            category.modules.append(module)

        return category
