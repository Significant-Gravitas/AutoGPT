from abc import ABC
from typing import Callable


class AgentComponent(ABC):
    run_after: list[type["AgentComponent"]] = []
    enabled: Callable[[], bool] | bool = True
    disabled_reason: str = ""


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
