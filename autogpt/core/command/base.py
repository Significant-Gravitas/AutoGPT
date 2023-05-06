import abc
import importlib
import inspect
from typing import Any, Callable, Optional, Tuple, Dict, TypeVar, List
from dataclasses import dataclass

@dataclass
class CommandResult:
    ok: bool
    message: str

T = TypeVar("T")
@dataclass
class CommandPayload:
    command_name: str
    payload: T

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
            return CommandResult(False, f"Command '{self.name}' is disabled: {self.disabled_reason}")
        
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


class CommandBus:
    '''A class which manages a collection of Command objects. 
    It allows the registration of commands  and is responsible for executing them.
    '''
    def __init__(self):
        self.commands = dict()
    
    def register_command(self, cmd: Command) -> None:
        self.commands[cmd.name] = cmd

    def generate_command_list(self, enabled_skills: List[str]) -> List[str]:
        return [
            f"{command_name} args: {command['format']} - {command['instruction']}"
            for command_name, command in self.commands.items()
            if command_name in enabled_skills
        ]
    
    def execute_command(self, command_name: str, **kwargs) -> CommandResult:
        try:
            if command_name not in self.commands:
                raise KeyError(f"Command '{command_name}' not found in registry.")
            
            command = self.commands[command_name]

            if not command.enabled:
                return CommandResult(False, f"Command '{command.name}' is disabled: {command.disabled_reason}")
            
            args, kwargs = command.__pre_call__(*args, **kwargs)
            
            command_result = command.method(*args, **kwargs)
            if isinstance(command_result, str):
                command_result = CommandResult(True, command_result)
            
            command_result = command.__post_hook__(command_result)
            return command_result
            
        except Exception as e:
            return CommandResult(False, f"Error: {str(e)}")
        
class CommandHandler(abc.ABC):
    '''A class which contains commands and is responsible for Registering them to the command bus.
    '''
    @abc.abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abc.abstractmethod
    def register_to(self, command_bus) -> None:
        pass


class SomeCommandHandler(CommandHandler):
    def __init__(self, some_external_dependency) -> None:
        self.some_external_dependency = some_external_dependency
    
    def command_with_external_dependency(self, payload) -> Command:
       res = self.some_external_dependency(payload)
       return CommandResult(True, res)
    
    def another_with_external_dependency(self, payload) -> Command:
       res = self.some_external_dependency(payload)
       return CommandResult(True, res)
       
    def register_to(self, command_bus: CommandBus) -> None:
        # Register Command 1
        command = Command(
            name="command_with_external_dependency",
            description="A command using an external dependency",
            method=self.command_with_external_dependency,
            signature="(payload)",
        )

        command_bus.register_command("command_with_external_dependency", command)

        # Register Command 2
        command = Command(
            name="another_with_external_dependency",
            description="A command using an external dependency",
            method=self.another_with_external_dependency,
            signature="(payload)",
        )
        command_bus.register_command("another_with_external_dependency", command)




'''
Example usage
'''

command_bus = CommandBus()
# This can be a function or a class
some_external_dependency = lambda payload: f"Processed: {payload}"

command_handler = SomeCommandHandler(some_external_dependency, command_bus)
command_handler.register_to(command_bus)

command_handler = SomeOtherCommandHandler(different_external_depencies, command_bus)
command_handler.register_to(command_bus)


# Pass command_bus to agent and execute command with.
command_bus.execute_command("command_with_external_dependency")
