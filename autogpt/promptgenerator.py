""" A module for generating custom prompt strings."""
from __future__ import annotations

from typing import Any


class PromptGenerator:
    """
    A class for generating custom prompt strings based on constraints, commands,
        resources, and performance evaluations.
    """

    def __init__(self) -> None:
        """
        Initialize the PromptGenerator object with empty lists of constraints,
            commands, resources, and performance evaluations.
        """
        self.identity_directives = []
        self.prime_directives = []
        self.constraints = []
        self.commands = []
        self.categorized_commands = {}
        self.resources = []
        self.performance_evaluation = []

    def add_identity_directive(self, constraint: str) -> None:
        self.identity_directives.append(constraint)
        
    def add_prime_directive(self, constraint: str) -> None:
        self.prime_directives.append(constraint)

    def add_constraint(self, constraint: str) -> None:
        self.constraints.append(constraint)

    def add_command(self, command_label: str, command_name: str, args=None) -> None:
        """
        Add a command to the commands list with a label, name, and optional arguments.

        Args:
            command_label (str): The label of the command.
            command_name (str): The name of the command.
            args (dict, optional): A dictionary containing argument names and their
              values. Defaults to None.
        """
        if args is None:
            args = {}

        command_args = {arg_key: arg_value for arg_key, arg_value in args.items()}

        command = {
            "label": command_label,
            "name": command_name,
            "args": command_args,
        }

        self.commands.append(command)

    def _generate_command_string(self, command: dict[str, Any]) -> str:
        """
        Generate a formatted string representation of a command.

        Args:
            command (dict): A dictionary containing command information.

        Returns:
            str: The formatted command string.
        """
        args_string = ", ".join(
            f'"{key}": "{value}"' for key, value in command["args"].items()
        )
        return f'{command["label"]}: "{command["name"]}", args: {args_string}'

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

    def _generate_numbered_list(self, items: list[Any], item_type="list") -> str:
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
            return "\n".join(
                f"{i+1}. {self._generate_command_string(item)}"
                for i, item in enumerate(items)
            )
        else:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    #todo: rework to handle list of commands
    def add_commands_from_list(self, category: str, command_list: list[tuple[str, str, dict]]) -> None:
        """
        Add commands from a list of tuples containing command label, name, and arguments
        under a specified category.

        Args:
            category (str): The category under which the commands should be added.
            command_list (list[tuple[str, str, dict]]): A list of command tuples.
        """
        if category not in self.categorized_commands:
            self.categorized_commands[category] = []

        for command in command_list:
            label, name, args = command
            self.add_command(label, name, args)
            command_dict = {
                "label": label,
                "name": name,
                "args": args,
            }
            self.categorized_commands[category].append(command_dict)


    #todo: rework to handle list of commands
    def _generate_categorized_commands(self) -> str:
        """
        Generate a formatted string representation of the categorized commands.

        Returns:
            str: The formatted categorized commands string.
        """
        output = []
        for category, command_list in self.categorized_commands.items():
            output.append(f"{category.upper()}:")
            output.append(self._generate_numbered_list(command_list, item_type='command'))
            output.append("")
        return "\n".join(output)

    def generate_prompt_string(self) -> str:
        """
        Generate a prompt string based on the constraints, commands, resources,
            and performance evaluations.

        Returns:
            str: The generated prompt string.
        """
        output = (
            f"Your identity directives:\n{self._generate_numbered_list(self.identity_directives)}\n\n"
            f"Your prime directives:\n{self._generate_numbered_list(self.prime_directives)}\n\n"
            f"Your constraints:\n{self._generate_numbered_list(self.constraints)}\n\n"
            "Your Available commands - Use these by filling in the arguments <with_your_input>:\n"
            f"{self._generate_categorized_commands()}\n"
            "Note, only the arguments above will work, anything else will error\n\n"
            f"Your resources:\n{self._generate_numbered_list(self.resources)}\n\n"
            "Your performance eval metrics:\n"
            f"{self._generate_numbered_list(self.performance_evaluation)}"
        )
        # print(output)
        return output
    