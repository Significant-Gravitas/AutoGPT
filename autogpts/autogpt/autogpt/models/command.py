from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

if TYPE_CHECKING:
    from autogpt.agents.base import CommandArgs

from .command_parameter import CommandParameter
from .context_item import ContextItem

CommandReturnValue = Any
CommandOutput = CommandReturnValue | tuple[CommandReturnValue, ContextItem]


class ValidityResult(NamedTuple):
    """Command `is_valid` result"""

    is_valid: bool
    reason: str = ""


class Command:
    """A class representing a command.

    Attributes:
        name (str): The name of the command.
        description (str): A brief description of what the command does.
        parameters (list): The parameters of the function that the command executes.
    """

    def __init__(
        self,
        names: list[str],
        description: str,
        method: Callable[..., CommandOutput],
        parameters: list[CommandParameter],
        is_valid: Callable[[CommandArgs], ValidityResult] = lambda a: ValidityResult(
            True
        ),
    ):
        # Check if all parameters are provided
        if not self._parameters_match(method, parameters):
            raise ValueError(
                f"Command {names[0]} has different parameters than provided schema"
            )
        self.names = names
        self.description = description
        self.method = method
        self.parameters = parameters
        self.is_valid = is_valid

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.method)

    @staticmethod
    def from_decorated_function(func: Callable) -> Command:
        return Command(
            names=getattr(func, "names", [func.__name__]),
            description=getattr(func, "description", ""),
            method=func,
            parameters=getattr(func, "parameters", []),
            is_valid=getattr(func, "is_valid", lambda a: ValidityResult(True)),
        )

    def _parameters_match(
        self, func: Callable, parameters: list[CommandParameter]
    ) -> bool:
        # Get the function's signature
        signature = inspect.signature(func)
        # Extract parameter names, ignoring 'self' for methods
        func_param_names = [
            param.name
            for param in signature.parameters.values()
            if param.name != "self"
        ]
        names = [param.name for param in parameters]
        # Check if sorted lists of names/keys are equal
        return sorted(func_param_names) == sorted(names)

    def __call__(self, *args, **kwargs) -> Any:
        return self.method(*args, **kwargs)

    def __str__(self) -> str:
        params = [
            f"{param.name}: "
            + ("%s" if param.spec.required else "Optional[%s]") % param.spec.type.value
            for param in self.parameters
        ]
        return (
            f"{self.names[0]}: {self.description.rstrip('.')}. "
            f"Params: ({', '.join(params)})"
        )
