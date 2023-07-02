import importlib
import inspect
from typing import Any, Callable

from autogpt.command_decorator import AUTO_GPT_COMMAND_IDENTIFIER
from autogpt.logs import logger
from autogpt.models.command import Command, GenericCommand, PromptCommand
from common import common


class UnkownCommand(KeyError):
    def __init__(self, command_name):
        self.command_name = command_name
        self.message = (
            f"Unknown command '{command_name}'. Please refer to the 'COMMANDS'"
        )
        " list for available commands and only respond in the specified JSON"
        " format."


common.default_not_detailed_errors.add(UnkownCommand)  # shared?


class AgentCommandRegistry:
    def __init__(self, agent, commandregistry):
        self.agent = agent
        self.commandregistry = commandregistry

    @common.simple_exception_handling(err_to_throw=[UnkownCommand], log_if_thrown=True)
    def get_command(self, name: str) -> GenericCommand:
        # hopefully we can resolve here
        cmd = self.commandregistry.get_command(name)

        # If the command is found, call it with the provided arguments
        if cmd:
            return cmd

        # TODO: Remove commands below after they are moved to the command registry.
        command_name = self.map_command_synonyms(name.lower())

        # TODO: Change these to take in a file rather than pasted code, if
        # non-file is given, return instructions "Input should be a python
        # filepath, write your code to file and try again
        for command in self.agent.ai_config.prompt_generator.commands:
            if (
                command_name == command["label"].lower()
                or command_name == command["name"].lower()
            ):
                return PromptCommand(**command)

        raise UnkownCommand(command_name)

    @staticmethod
    def map_command_synonyms(command_name: str):
        """Takes the original command name given by the AI, and checks if the
        string matches a list of common/known hallucinations
        """
        synonyms = [
            ("write_file", "write_to_file"),
            ("create_file", "write_to_file"),
            ("search", "google"),
        ]
        for seen_command, actual_command_name in synonyms:
            if command_name == seen_command:
                return actual_command_name
        return command_name


class CommandRegistry:
    """
    The CommandRegistry class is a manager for a collection of Command objects.
    It allows the registration, modification, and retrieval of Command objects,
    as well as the scanning and loading of command plugins from a specified
    directory.
    """

    commands: dict[str, Command]

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

    def unregister(self, command_name: str):
        if command_name in self.commands:
            del self.commands[command_name]
        else:
            raise KeyError(f"Command '{command_name}' not found in registry.")

    def reload_commands(self) -> None:
        """Reloads all loaded command plugins."""
        for cmd_name in self.commands:
            cmd = self.commands[cmd_name]
            module = self._import_module(cmd.__module__)
            reloaded_module = self._reload_module(module)
            if hasattr(reloaded_module, "register"):
                reloaded_module.register(self)

    def call(self, command_name: str, **kwargs) -> Any:
        if command_name not in self.commands:
            raise KeyError(f"Command '{command_name}' not found in registry.")
        command = self.commands[command_name]
        return command(**kwargs)

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

    def get_command(self, name: str) -> GenericCommand:
        # hopefully we can resolve here
        return self.commands.get(name)
