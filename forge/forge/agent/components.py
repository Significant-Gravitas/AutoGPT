from __future__ import annotations

from abc import ABC
from typing import Callable, Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="AgentComponent")
C = TypeVar("C", bound=BaseModel)


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


class ConfigurableComponent(ABC, Generic[C]):
    """A component that can be configured with a Pydantic model."""

    def __init__(self, config: Optional[C] = None):
        self._config: Optional[C] = config

    @property
    def config(self) -> C:
        if self._config is None:
            raise ValueError(
                "Component is not configured. "
                "Call `super().__init__(config)` in `__init__` "
                "or set the `config` attribute manually."
            )
        return self._config

    @config.setter
    def config(self, config: C):
        self._config = config


class ComponentEndpointError(Exception):
    """Error of a single protocol method on a component."""

    def __init__(self, message: str, component: AgentComponent):
        self.message = message
        self.triggerer = component
        super().__init__(message)


class EndpointPipelineError(ComponentEndpointError):
    """Error of an entire pipline of one endpoint."""


class ComponentSystemError(EndpointPipelineError):
    """Error of a group of pipelines;
    multiple different enpoints."""
