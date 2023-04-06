import os
import sys
import importlib
import inspect
from typing import Callable, Any, List

# Unique identifier for auto-gpt commands
AUTO_GPT_COMMAND_IDENTIFIER = "auto_gpt_command"

class Command:
    """A class representing a command.

    Attributes:
        name (str): The name of the command.
        description (str): A brief description of what the command does.
        method (Callable[..., Any]): The function that the command executes.
        signature (str): The signature of the function that the command executes. Defaults to None.
    """

    def __init__(self, name: str, description: str, method: Callable[..., Any], signature: str = None):
        self.name = name
        self.description = description
        self.method = method
        self.signature = signature if signature else str(inspect.signature(self.method))

    def __call__(self, *args, **kwargs) -> Any:
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

    def register_command(self, cmd: Command) -> None:
        self.commands[cmd.name] = cmd

    def reload_commands(self) -> None:
        """Reloads all loaded command plugins."""
        for cmd_name in self.commands:
            cmd = self.commands[cmd_name]
            module = self._import_module(cmd.__module__)
            reloaded_module = self._reload_module(module)
            if hasattr(reloaded_module, "register"):
                reloaded_module.register(self)

    def get_command(self, name: str) -> Callable[..., Any]:
        return self.commands.get(name)

    def list_commands(self) -> List[str]:
        return [str(cmd) for cmd in self.commands.values()]

    def command_prompt(self) -> str:
        """
        Returns a string representation of all registered `Command` objects for use in a prompt
        """
        commands_list = [f"{idx + 1}. {str(cmd)}" for idx, cmd in enumerate(self.commands.values())]
        return "\n".join(commands_list)

    def scan_directory_for_plugins(self, directory: str) -> None:
        """
        Scans the specified directory for Python files containing command plugins.

       For each file in the directory that ends with ".py", this method imports the associated module and registers any
       functions or classes that are decorated with the `AUTO_GPT_COMMAND_IDENTIFIER` attribute as `Command` objects.
       The registered `Command` objects are then added to the `commands` dictionary of the `CommandRegistry` object.

       Args:
           directory (str): The directory to scan for command plugins.
       """

        for file in os.listdir(directory):
            if file.endswith(".py"):
                module_name = file[:-3]
                module = importlib.import_module(module_name)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    # Register decorated functions
                    if hasattr(attr, AUTO_GPT_COMMAND_IDENTIFIER) and getattr(attr, AUTO_GPT_COMMAND_IDENTIFIER):
                        self.register_command(attr.register_command)
                    # Register command classes
                    elif inspect.isclass(attr) and issubclass(attr, Command) and attr != Command:
                        cmd_instance = attr()
                        self.register_command(cmd_instance)


def command(name: str, description: str, signature: str = None) -> Callable[..., Any]:
    """The command decorator is used to create Command objects from ordinary functions."""
    def decorator(func: Callable[..., Any]) -> Command:
        cmd = Command(name=name, description=description, method=func, signature=signature)

        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)

        wrapper.register_command = cmd
        setattr(wrapper, AUTO_GPT_COMMAND_IDENTIFIER, True)
        return wrapper

    return decorator

