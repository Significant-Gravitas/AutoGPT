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


class ComponentEndpointError(Exception):
    """Error of a single protocol method on a component."""

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)


class EndpointPipelineError(ComponentEndpointError):
    """Error of an entire pipeline of one endpoint."""


class ComponentSystemError(EndpointPipelineError):
    """Error of a group of pipelines;
    multiple different endpoints."""
