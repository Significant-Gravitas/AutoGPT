from typing import Any, Callable, Optional

from autogpt.config import Config

from .command_parameter import CommandParameter


class Command:
    """A class representing a command.

    Attributes:
        name (str): The name of the command.
        description (str): A brief description of what the command does.
        parameters (list): The parameters of the function that the command executes.
    """

    def __init__(
        self,
        name: str,
        description: str,
        method: Callable[..., Any],
        parameters: list[CommandParameter],
        enabled: bool | Callable[[Config], bool] = True,
        disabled_reason: Optional[str] = None,
        aliases: list[str] = [],
    ):
        self.name = name
        self.description = description
        self.method = method
        self.parameters = parameters
        self.enabled = enabled
        self.disabled_reason = disabled_reason
        self.aliases = aliases

    def __call__(self, *args, **kwargs) -> Any:
        if hasattr(kwargs, "config") and callable(self.enabled):
            self.enabled = self.enabled(kwargs["config"])
        if not self.enabled:
            if self.disabled_reason:
                return f"Command '{self.name}' is disabled: {self.disabled_reason}"
            return f"Command '{self.name}' is disabled"
        return self.method(*args, **kwargs)

    def __str__(self) -> str:
        params = [
            f"{param.name}: {param.type if param.required else f'Optional[{param.type}]'}"
            for param in self.parameters
        ]
        return f"{self.name}: {self.description}, params: ({', '.join(params)})"
