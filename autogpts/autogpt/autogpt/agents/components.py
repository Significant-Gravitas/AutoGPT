from typing import Callable, Generic, Type, TypeVar

#TODO temporary wrapper for single-result component functions
T = TypeVar("T")
class Single(Generic[T]):
    def __init__(self, value: T):
        self.value = value
class Component:
    run_after: list[type["Component"]] = []
    enabled: Callable[[], bool] | bool = True

    @classmethod
    def get_dependencies(cls) -> list[Type["Component"]]:
        return cls.run_after


class ComponentError(Exception):
    pass


class PipelineError(ComponentError):
    pass


class ComponentSystemError(ComponentError):
    pass
