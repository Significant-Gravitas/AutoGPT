from typing import Callable, Generic, Type, TypeVar

#TODO temporary wrapper for single-result component functions
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
    pass


class PipelineError(ComponentError):
    pass


class ComponentSystemError(ComponentError):
    pass
