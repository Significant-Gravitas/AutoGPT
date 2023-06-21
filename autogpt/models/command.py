from typing import Any, Callable, Dict, Optional

from autogpt.config import Config


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
        signature: Dict[str, Dict[str, Any]],
        enabled: bool | Callable[[Config], bool] = True,
        disabled_reason: Optional[str] = None,
    ):
        self.name = name
        self.description = description
        self.method = method
        self.signature = signature
        self.enabled = enabled
        self.disabled_reason = disabled_reason

    def __call__(self, *args, **kwargs) -> Any:
        if hasattr(kwargs, "config") and callable(self.enabled):
            self.enabled = self.enabled(kwargs["config"])
        if not self.enabled:
            if self.disabled_reason:
                return f"Command '{self.name}' is disabled: {self.disabled_reason}"
            return f"Command '{self.name}' is disabled"
        return self.method(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.name}: {self.description}, args: {self.signature}"
