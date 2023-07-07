import functools
from typing import Any, Callable, Optional, Type, TypedDict

from autogpt.config import Config
from autogpt.models.command import Command, CommandInstance, CommandParameter

# Unique identifier for auto-gpt commands
AUTO_GPT_COMMAND_IDENTIFIER = "auto_gpt_command"


class CommandParameterSpec(TypedDict):
    type: str
    description: str
    required: bool


def command_attr(
    name: str,
    description: str,
    parameters: dict[str, CommandParameterSpec],
    enabled: bool | Callable[[Config], bool] = True,
    disabled_reason: Optional[str] = None,
    instancecls: Optional[CommandInstance] = CommandInstance,
    max_seen_to_stop: Optional[int] = None,
    stop_if_looped: Optional[bool] = True,
) -> Callable[..., Any]:
    """The command decorator is used to create Command objects from ordinary functions."""

    def decorator(cls):
        def my_init(self, method=None):
            typed_parameters = [
                CommandParameter(
                    name=param_name,
                    description=parameter.get("description"),
                    type=parameter.get("type", "string"),
                    required=parameter.get("required", False),
                )
                for param_name, parameter in parameters.items()
            ]
            Command.__init__(
                self,
                name=name,
                description=description,
                parameters=typed_parameters,
                enabled=enabled,
                disabled_reason=disabled_reason,
                instancecls=instancecls,
                max_seen_to_stop=max_seen_to_stop,
                stop_if_looped=stop_if_looped,
                method=method,
            )

        cls.__init__ = my_init
        return cls

    return decorator


@functools.wraps(command_attr)
def command(*args, **kwargs):
    class MyCommand(Command):
        pass

    @functools.singledispatch
    def decorator(func: Callable[..., Any]) -> Command:
        t = command_attr(*args, **kwargs)

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return func(*args, **kwargs)

        wrapper.command = t(MyCommand)(method=func)

        setattr(wrapper, AUTO_GPT_COMMAND_IDENTIFIER, True)

        return wrapper

    @decorator.register(type)
    def decorator_cls(cls: Type[Command]) -> Type[Command]:
        t = command_attr(*args, **kwargs)
        return t(cls)

    return decorator
