""" A module for generating custom prompt strings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from autogpt.models.command_registry import CommandRegistry


class PromptGenerator:
    """
    A class for generating custom prompt strings based on constraints, commands,
        resources, and performance evaluations.
    """

    @dataclass
    class Command:
        label: str
        name: str
        params: dict[str, str]
        function: Optional[Callable]

        def __str__(self) -> str:
            """Returns a string representation of the command."""
            params_string = ", ".join(
                f'"{key}": "{value}"' for key, value in self.params.items()
            )
            return f'{self.label}: "{self.name}", params: ({params_string})'

    constraints: list[str]
    commands: list[Command]
    resources: list[str]
    best_practices: list[str]
    command_registry: CommandRegistry | None

    def __init__(self):
        self.constraints = []
        self.commands = []
        self.resources = []
        self.best_practices = []
        self.command_registry = None

    def add_constraint(self, constraint: str) -> None:
        """
        Add a constraint to the constraints list.

        Args:
            constraint (str): The constraint to be added.
        """
        self.constraints.append(constraint)

    def add_command(
        self,
        command_label: str,
        command_name: str,
        params: dict[str, str] = {},
        function: Optional[Callable] = None,
    ) -> None:
        """
        Add a command to the commands list with a label, name, and optional arguments.

        *Should only be used by plugins.* Native commands should be added
        directly to the CommandRegistry.

        Args:
            command_label (str): The label of the command.
            command_name (str): The name of the command.
            params (dict, optional): A dictionary containing argument names and their
              values. Defaults to None.
            function (callable, optional): A callable function to be called when
                the command is executed. Defaults to None.
        """

        self.commands.append(
            PromptGenerator.Command(
                label=command_label,
                name=command_name,
                params={name: type for name, type in params.items()},
                function=function,
            )
        )

    def add_resource(self, resource: str) -> None:
        """
        Add a resource to the resources list.

        Args:
            resource (str): The resource to be added.
        """
        self.resources.append(resource)

    def add_best_practice(self, best_practice: str) -> None:
        """
        Add an item to the list of best practices.

        Args:
            best_practice (str): The best practice item to be added.
        """
        self.best_practices.append(best_practice)

    def _generate_numbered_list(self, items: list[str], start_at: int = 1) -> str:
        """
        Generate a numbered list containing the given items.

        Args:
            items (list): A list of items to be numbered.
            start_at (int, optional): The number to start the sequence with; defaults to 1.

        Returns:
            str: The formatted numbered list.
        """
        return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start_at))

    def generate_prompt_string(
        self,
        *,
        additional_constraints: list[str] = [],
        additional_resources: list[str] = [],
        additional_best_practices: list[str] = [],
    ) -> str:
        """
        Generate a prompt string based on the constraints, commands, resources,
            and best practices.

        Returns:
            str: The generated prompt string.
        """

        return (
            "## Constraints\n"
            "You operate within the following constraints:\n"
            f"{self._generate_numbered_list(self.constraints + additional_constraints)}\n\n"
            "## Commands\n"
            "You have access to the following commands:\n"
            f"{self._generate_commands()}\n\n"
            "## Resources\n"
            "You can leverage access to the following resources:\n"
            f"{self._generate_numbered_list(self.resources + additional_resources)}\n\n"
            "## Best practices\n"
            f"{self._generate_numbered_list(self.best_practices + additional_best_practices)}"
        )

    def _generate_commands(self) -> str:
        command_strings = []
        if self.command_registry:
            command_strings += [
                str(cmd)
                for cmd in self.command_registry.commands.values()
                if cmd.enabled
            ]

        # Add commands from plugins etc.
        command_strings += [str(cmd) for cmd in self.commands]

        return self._generate_numbered_list(command_strings)
