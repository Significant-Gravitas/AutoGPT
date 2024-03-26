from typing import Type

class Component:
    run_after: list[type["Component"]] = []

    @classmethod
    def get_dependencies(cls) -> list[Type["Component"]]:
        return cls.run_after


class ComponentError(Exception):
    pass


class PipelineError(ComponentError):
    pass


class ComponentSystemError(ComponentError):
    pass
