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
    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)


class ProtocolError(ComponentError):
    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)


class ComponentGroupError(ComponentError):
    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)


class ComponentSystemError(ComponentError):
    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(message)
