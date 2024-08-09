from __future__ import annotations

import inspect
from typing import Callable, Generic, ParamSpec, TypeVar

from .parameter import CommandParameter

P = ParamSpec("P")
CO = TypeVar("CO")  # command output


class Command(Generic[P, CO]):
    """A class representing a command.

    Attributes:
        name (str): The name of the command.
        description (str): A brief description of what the command does.
        parameters (list): The parameters of the function that the command executes.
    """

    def __init__(
        self,
        names: list[str],
        description: str,
        method: Callable[P, CO],
        parameters: list[CommandParameter],
    ):
        self.names = names
        self.description = description
        self.method = method
        self.parameters = parameters

        # Check if all parameters are provided
        if not self._parameters_match_signature():
            raise ValueError(
                f"Command {self.name} has different parameters than provided schema"
            )

    @property
    def name(self) -> str:
        return self.names[0]  # TODO: fallback to other name if first one is taken

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.method)

    @property
    def return_type(self) -> str:
        _type = inspect.signature(self.method).return_annotation
        if _type == inspect.Signature.empty:
            return "None"
        return _type.__name__

    @property
    def header(self) -> str:
        """Returns a function header representing the command's signature

        Examples:
        ```py
        def execute_python_code(code: str) -> str:

        async def extract_info_from_content(content: str, instruction: str, output_type: type[~T]) -> ~T:
        """  # noqa
        return (
            f"{'async ' if self.is_async else ''}"
            f"def {self.name}{inspect.signature(self.method)}:"
        )

    def _parameters_match_signature(self) -> bool:
        # Get the function's signature
        signature = inspect.signature(self.method)
        # Extract parameter names, ignoring 'self' for methods
        func_param_names = [
            param.name
            for param in signature.parameters.values()
            if param.name != "self"
        ]
        names = [param.name for param in self.parameters]
        # Check if sorted lists of names/keys are equal
        return sorted(func_param_names) == sorted(names)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> CO:
        return self.method(*args, **kwargs)

    def __str__(self) -> str:
        params = [
            f"{param.name}: "
            + ("%s" if param.spec.required else "Optional[%s]")
            % (param.spec.type.value if param.spec.type else "Any")
            for param in self.parameters
        ]
        return (
            f"{self.name}: {self.description.rstrip('.')}. "
            f"Params: ({', '.join(params)})"
        )

    def __get__(self, instance, owner):
        if instance is None:
            # Accessed on the class, not an instance
            return self
        # Bind the method to the instance
        return Command(
            self.names,
            self.description,
            self.method.__get__(instance, owner),
            self.parameters,
        )
