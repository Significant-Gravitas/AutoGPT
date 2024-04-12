from abc import ABC
from typing import Callable


class AgentComponent(ABC):
    run_after: list[type["AgentComponent"]] = []
    _enabled: Callable[[], bool] | bool = True
    _disabled_reason: str = ""

    @property
    def enabled(self) -> bool:
        if callable(self._enabled):
            return self._enabled()
        return self._enabled

    @property
    def disabled_reason(self) -> str:
        return self._disabled_reason


class ComponentError(Exception):
    """Error of a single component."""

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)


class ProtocolError(ComponentError):
    """Error of an entire pipeline of one component type."""


class PipelineError(ComponentError):
    """Error of a group of component types;
    multiple protocols."""
