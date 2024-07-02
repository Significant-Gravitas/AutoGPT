from __future__ import annotations

from abc import ABC
from typing import Callable, ClassVar, Generic, Optional, TypeVar

from pydantic import BaseModel

from forge.models.config import _update_user_config_from_env, deep_update

AC = TypeVar("AC", bound="AgentComponent")
BM = TypeVar("BM", bound=BaseModel)


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

    def run_after(self: AC, *components: type[AgentComponent] | AgentComponent) -> AC:
        """Set the components that this component should run after."""
        for component in components:
            t = component if isinstance(component, type) else type(component)
            if t not in self._run_after and t is not self.__class__:
                self._run_after.append(t)
        return self


class ConfigurableComponent(ABC, Generic[BM]):
    """A component that can be configured with a Pydantic model."""

    config_class: ClassVar[type[BM]]  # type: ignore

    def __init__(self, configuration: Optional[BM]):
        self._config: Optional[BM] = None
        if configuration is not None:
            self.config = configuration

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if getattr(cls, "config_class", None) is None:
            raise NotImplementedError(
                f"ConfigurableComponent subclass {cls.__name__} "
                "must define config_class class attribute."
            )

    @property
    def config(self) -> BM:
        if not hasattr(self, "_config") or self._config is None:
            self.config = self.config_class()
        return self._config  # type: ignore

    @config.setter
    def config(self, config: BM):
        if not hasattr(self, "_config") or self._config is None:
            # Load configuration from environment variables
            updated = _update_user_config_from_env(config)
            config = self.config_class(**deep_update(config.model_dump(), updated))
        self._config = config


class ComponentEndpointError(Exception):
    """Error of a single protocol method on a component."""

    def __init__(self, message: str, component: AgentComponent):
        self.message = message
        self.triggerer = component
        super().__init__(message)


class EndpointPipelineError(ComponentEndpointError):
    """Error of an entire pipeline of one endpoint."""


class ComponentSystemError(EndpointPipelineError):
    """Error of a group of pipelines;
    multiple different endpoints."""
