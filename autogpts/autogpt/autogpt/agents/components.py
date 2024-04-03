from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class Single(Generic[T]):
    """A wrapper for a single result (non-pipeline) of a component function."""

    def __init__(self, value: T):
        self.value = value


class Component:
    run_after: list[type["Component"]] = []
    enabled: Callable[[], bool] | bool = True
    disabled_reason: str = ""


class ComponentError(Exception):
    """Error of a single component."""

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)


class ProtocolError(ComponentError):
    """Error of an entire pipeline of one component type."""

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)


class ComponentGroupError(ComponentError):
    """Error of a group of component types;
    multiple pipelines."""

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)


class ComponentSystemError(ComponentError):
    """Error of an entire system."""

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)
