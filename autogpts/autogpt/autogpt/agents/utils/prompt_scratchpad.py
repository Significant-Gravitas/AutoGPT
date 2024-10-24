import logging
from typing import Callable

from pydantic import BaseModel, Field

from autogpt.core.resource.model_providers.schema import CompletionModelFunction
from autogpt.core.utils.json_schema import JSONSchema

logger = logging.getLogger("PromptScratchpad")


class CallableCompletionModelFunction(CompletionModelFunction):
    method: Callable


class PromptScratchpad(BaseModel):
    commands: dict[str, CallableCompletionModelFunction] = Field(default_factory=dict)
    resources: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    best_practices: list[str] = Field(default_factory=list)

    def add_constraint(self, constraint: str) -> None:
        """
        Add a constraint to the constraints list.

        Params:
            constraint (str): The constraint to be added.
        """
        if constraint not in self.constraints:
            self.constraints.append(constraint)

    def add_command(
        self,
        name: str,
        description: str,
        params: dict[str, str | dict],
        function: Callable,
    ) -> None:
        """
        Registers a command.

        *Should only be used by plugins.* Native commands should be added
        directly to the CommandRegistry.

        Params:
            name (str): The name of the command (e.g. `command_name`).
            description (str): The description of the command.
            params (dict, optional): A dictionary containing argument names and their
              types. Defaults to an empty dictionary.
            function (callable, optional): A callable function to be called when
                the command is executed. Defaults to None.
        """
        for p, s in params.items():
            invalid = False
            if type(s) is str and s not in JSONSchema.Type._value2member_map_:
                invalid = True
                logger.warning(
                    f"Cannot add command '{name}':"
                    f" parameter '{p}' has invalid type '{s}'."
                    f" Valid types are: {JSONSchema.Type._value2member_map_.keys()}"
                )
            elif isinstance(s, dict):
                try:
                    JSONSchema.from_dict(s)
                except KeyError:
                    invalid = True
            if invalid:
                return

        command = CallableCompletionModelFunction(
            name=name,
            description=description,
            parameters={
                name: JSONSchema(type=JSONSchema.Type._value2member_map_[spec])
                if type(spec) is str
                else JSONSchema.from_dict(spec)
                for name, spec in params.items()
            },
            method=function,
        )

        if name in self.commands:
            if description == self.commands[name].description:
                return
            logger.warning(
                f"Replacing command {self.commands[name]} with conflicting {command}"
            )
        self.commands[name] = command

    def add_resource(self, resource: str) -> None:
        """
        Add a resource to the resources list.

        Params:
            resource (str): The resource to be added.
        """
        if resource not in self.resources:
            self.resources.append(resource)

    def add_best_practice(self, best_practice: str) -> None:
        """
        Add an item to the list of best practices.

        Params:
            best_practice (str): The best practice item to be added.
        """
        if best_practice not in self.best_practices:
            self.best_practices.append(best_practice)
