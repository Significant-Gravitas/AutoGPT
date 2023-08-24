from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Optional

if TYPE_CHECKING:
    from autogpt.agents.base import BaseAgent
    from autogpt.config import Config

from .command_parameter import CommandParameter
from .context_item import ContextItem

CommandReturnValue = Any
CommandOutput = CommandReturnValue | tuple[CommandReturnValue, ContextItem]


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
        method: Callable[..., CommandOutput],
        parameters: list[CommandParameter],
        enabled: Literal[True] | Callable[[Config], bool] = True,
        disabled_reason: Optional[str] = None,
        aliases: list[str] = [],
        available: Literal[True] | Callable[[BaseAgent], bool] = True,
    ):
        self.name = name
        self.description = description
        self.method = method
        self.parameters = parameters
        self.enabled = enabled
        self.disabled_reason = disabled_reason
        self.aliases = aliases
        self.available = available

    def __call__(self, *args, agent: BaseAgent, **kwargs) -> Any:
        if callable(self.enabled) and not self.enabled(agent.config):
            if self.disabled_reason:
                return f"Command '{self.name}' is disabled: {self.disabled_reason}"
            return f"Command '{self.name}' is disabled"

        if callable(self.available) and not self.available(agent):
            return f"Command '{self.name}' is not available"

        return self.method(*args, **kwargs, agent=agent)

    def __str__(self) -> str:
        params = [
            f"{param.name}: {param.type if param.required else f'Optional[{param.type}]'}"
            for param in self.parameters
        ]
        return f"{self.name}: {self.description.rstrip('.')}. Params: ({', '.join(params)})"
