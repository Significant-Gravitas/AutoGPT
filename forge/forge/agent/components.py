from __future__ import annotations

from abc import ABC, ABCMeta
from typing import Callable, Generic, Optional, TypeVar, get_args, get_type_hints

from forge.models.config import ComponentConfiguration

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


C = TypeVar("C", bound=ComponentConfiguration)


# class ConfigurableComponentMeta(ABCMeta):
#     def __call__(cls, *args, **kwargs):
#         if 'config' not in kwargs:
#             # Extract the type of the config from the generic type hint
#             generic_base = next(
#                 base for base in cls.__orig_bases__ if isinstance(base, Generic)
#             )
#             config_type = get_type_hints(generic_base)['C']
#             kwargs['config'] = config_type()  # Instantiate the config class
#         return super().__call__(*args, **kwargs)


class ConfigurableComponent(ABC, Generic[C]):
    """A component that can be configured with a Pydantic model."""

    def __init__(self, config: Optional[C] = None):
        self._config: Optional[C] = config

    @property
    def config(self) -> C:
        if not hasattr(self, "_config") or self._config is None:
            config_type = self._get_config_type()
            self._config = config_type()
        return self._config

    @config.setter
    def config(self, value: C):
        self._config = value

    @classmethod
    def _get_config_type(cls) -> type[C]:
        hints = get_type_hints(cls)
        return hints["config"]


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
