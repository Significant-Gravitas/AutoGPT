""" A module for generating custom prompt strings."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, TypedDict

from autogpt.config import Config
from autogpt.json_utils.utilities import llm_response_schema

if TYPE_CHECKING:
    from autogpt.models.command_registry import CommandRegistry


class PromptGenerator:
    """
    A class for generating custom prompt strings based on constraints, commands,
        resources, and performance evaluations.
    """

    class Command(TypedDict):
        label: str
        name: str
        params: dict[str, str]
        function: Optional[Callable]

    constraints: list[str]
    commands: list[Command]
    resources: list[str]
    performance_evaluation: list[str]
    command_registry: CommandRegistry | None

    # TODO: replace with AIConfig
    name: str
    role: str
    goals: list[str]

    def __init__(self):
        self.constraints = []
        self.commands = []
        self.resources = []
        self.performance_evaluation = []
        self.command_registry = None

        self.name = "Bob"
        self.role = "AI"
        self.goals = []

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
        command_params = {name: type for name, type in params.items()}

        command: PromptGenerator.Command = {
            "label": command_label,
            "name": command_name,
            "params": command_params,
            "function": function,
        }

        self.commands.append(command)

    def _generate_command_string(self, command: Dict[str, Any]) -> str:
        """
        Generate a formatted string representation of a command.

        Args:
            command (dict): A dictionary containing command information.

        Returns:
            str: The formatted command string.
        """
        params_string = ", ".join(
            f'"{key}": "{value}"' for key, value in command["params"].items()
        )
        return f'{command["label"]}: "{command["name"]}", params: {params_string}'

    def add_resource(self, resource: str) -> None:
        """
        Add a resource to the resources list.

        Args:
            resource (str): The resource to be added.
        """
        self.resources.append(resource)

    def add_performance_evaluation(self, evaluation: str) -> None:
        """
        Add a performance evaluation item to the performance_evaluation list.

        Args:
            evaluation (str): The evaluation item to be added.
        """
        self.performance_evaluation.append(evaluation)

    def _generate_numbered_list(self, items: List[Any], item_type="list") -> str:
        """
        Generate a numbered list from given items based on the item_type.

        Args:
            items (list): A list of items to be numbered.
            item_type (str, optional): The type of items in the list.
                Defaults to 'list'.

        Returns:
            str: The formatted numbered list.
        """
        if item_type == "command":
            command_strings = []
            if self.command_registry:
                command_strings += [
                    str(item)
                    for item in self.command_registry.commands.values()
                    if item.enabled
                ]
            # terminate command is added manually
            command_strings += [self._generate_command_string(item) for item in items]
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(command_strings))
        else:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    def generate_prompt_string(self, config: Config) -> str:
        """
        Generate a prompt string based on the constraints, commands, resources,
            and performance evaluations.

        Returns:
            str: The generated prompt string.
        """
        return (
            f"Constraints:\n{self._generate_numbered_list(self.constraints)}\n\n"
            f"{generate_commands(self, config)}"
            f"Resources:\n{self._generate_numbered_list(self.resources)}\n\n"
            "Performance Evaluation:\n"
            f"{self._generate_numbered_list(self.performance_evaluation)}\n\n"
            "Respond with only valid JSON conforming to the following schema: \n"
            f"{json.dumps(llm_response_schema(config))}\n"
        )


def generate_commands(self, config: Config) -> str:
    """
    Generate a prompt string based on the constraints, commands, resources,
        and performance evaluations.

    Returns:
        str: The generated prompt string.
    """
    if config.openai_functions:
        return ""
    return (
        "Commands:\n"
        f"{self._generate_numbered_list(self.commands, item_type='command')}\n\n"
    )
