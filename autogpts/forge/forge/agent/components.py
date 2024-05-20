from __future__ import annotations

from abc import ABC
from typing import Callable, TypeVar

T = TypeVar("T", bound="AgentComponent")


class AgentComponent(ABC):
    """Base class for all agent components."""

    _run_after: list[type[AgentComponent]] = []
    _enabled: Callable[[], bool] | bool = True
    _disabled_reason: str = ""

    @property
    def enabled(self) -> bool:
        if callable(self._enabled):
            return self._enabled()
        return self._enabled

    @property
    def disabled_reason(self) -> str:
        """Return the reason this component is disabled."""
        return self._disabled_reason

    def run_after(self: T, *components: type[AgentComponent] | AgentComponent) -> T:
        """Set the components that this component should run after."""
        for component in components:
            t = component if isinstance(component, type) else type(component)
            if t not in self._run_after and t is not self.__class__:
                self._run_after.append(t)
        return self


class ComponentEndpointError(Exception):
    """Error of a single protocol method on a component."""

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)


class EndpointPipelineError(ComponentEndpointError):
    """Error of an entire pipline of one endpoint."""


class ComponentSystemError(EndpointPipelineError):
    """Error of a group of pipelines;
    multiple different enpoints."""
