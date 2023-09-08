""" A module for generating custom prompt strings."""
from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import distro

if TYPE_CHECKING:
    from autogpt.agents.base import BaseAgent
    from autogpt.config import AIConfig, AIDirectives, Config
    from autogpt.models.command_registry import CommandRegistry

from .utils import format_numbered_list

logger = logging.getLogger(__name__)


class PromptGenerator:
    """
    A class for generating custom prompt strings based on constraints, commands,
        resources, and performance evaluations.
    """

    ai_config: AIConfig

    best_practices: list[str]
    constraints: list[str]
    resources: list[str]

    commands: dict[str, Command]
    command_registry: CommandRegistry

    def __init__(
        self,
        ai_config: AIConfig,
        ai_directives: AIDirectives,
        command_registry: CommandRegistry,
    ):
        self.ai_config = ai_config
        self.best_practices = ai_directives.best_practices
        self.constraints = ai_directives.constraints
        self.resources = ai_directives.resources
        self.commands = {}
        self.command_registry = command_registry

    @dataclass
    class Command:
        name: str
        description: str
        params: dict[str, str]
        function: Optional[Callable]

        def __str__(self) -> str:
            """Returns a string representation of the command."""
            params_string = ", ".join(
                f'"{key}": "{value}"' for key, value in self.params.items()
            )
            return f'{self.name}: "{self.description.rstrip(".")}". Params: ({params_string})'

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
        params: dict[str, str] = {},
        function: Optional[Callable] = None,
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
        command = PromptGenerator.Command(
            name=name,
            description=description,
            params={name: type for name, type in params.items()},
            function=function,
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

    def construct_system_prompt(self, agent: BaseAgent) -> str:
        """Constructs a system prompt containing the most important information for the AI.

        Params:
            agent: The agent for which the system prompt is being constructed.

        Returns:
            str: The constructed system prompt.
        """

        for plugin in agent.config.plugins:
            if not plugin.can_handle_post_prompt():
                continue
            plugin.post_prompt(self)

        # Construct full prompt
        full_prompt_parts = (
            self._generate_intro_prompt()
            + self._generate_os_info(agent.config)
            + self._generate_body(
                agent=agent,
                additional_constraints=self._generate_budget_info(),
            )
            + self._generate_goals_info()
        )

        # Join non-empty parts together into paragraph format
        return "\n\n".join(filter(None, full_prompt_parts)).strip("\n")

    def _generate_intro_prompt(self) -> list[str]:
        """Generates the introduction part of the prompt.

        Returns:
            list[str]: A list of strings forming the introduction part of the prompt.
        """
        return [
            f"You are {self.ai_config.ai_name}, {self.ai_config.ai_role.rstrip('.')}.",
            "Your decisions must always be made independently without seeking "
            "user assistance. Play to your strengths as an LLM and pursue "
            "simple strategies with no legal complications.",
        ]

    def _generate_os_info(self, config: Config) -> list[str]:
        """Generates the OS information part of the prompt.

        Params:
            config (Config): The configuration object.

        Returns:
            str: The OS information part of the prompt.
        """
        if config.execute_local_commands:
            os_name = platform.system()
            os_info = (
                platform.platform(terse=True)
                if os_name != "Linux"
                else distro.name(pretty=True)
            )
            return [f"The OS you are running on is: {os_info}"]
        return []

    def _generate_budget_info(self) -> list[str]:
        """Generates the budget information part of the prompt.

        Returns:
            list[str]: The budget information part of the prompt, or an empty list.
        """
        if self.ai_config.api_budget > 0.0:
            return [
                f"It takes money to let you run. "
                f"Your API budget is ${self.ai_config.api_budget:.3f}"
            ]
        return []

    def _generate_goals_info(self) -> list[str]:
        """Generates the goals information part of the prompt.

        Returns:
            str: The goals information part of the prompt.
        """
        if self.ai_config.ai_goals:
            return [
                "\n".join(
                    [
                        "## Goals",
                        "For your task, you must fulfill the following goals:",
                        *[
                            f"{i+1}. {goal}"
                            for i, goal in enumerate(self.ai_config.ai_goals)
                        ],
                    ]
                )
            ]
        return []

    def _generate_body(
        self,
        agent: BaseAgent,
        *,
        additional_constraints: list[str] = [],
        additional_resources: list[str] = [],
        additional_best_practices: list[str] = [],
    ) -> list[str]:
        """
        Generates a prompt section containing the constraints, commands, resources,
        and best practices.

        Params:
            agent: The agent for which the prompt string is being generated.
            additional_constraints: Additional constraints to be included in the prompt string.
            additional_resources: Additional resources to be included in the prompt string.
            additional_best_practices: Additional best practices to be included in the prompt string.

        Returns:
            str: The generated prompt section.
        """

        return [
            "## Constraints\n"
            "You operate within the following constraints:\n"
            f"{format_numbered_list(self.constraints + additional_constraints)}",
            "## Resources\n"
            "You can leverage access to the following resources:\n"
            f"{format_numbered_list(self.resources + additional_resources)}",
            "## Commands\n"
            "You have access to the following commands:\n"
            f"{self.list_commands(agent)}",
            "## Best practices\n"
            f"{format_numbered_list(self.best_practices + additional_best_practices)}",
        ]

    def list_commands(self, agent: BaseAgent) -> str:
        """Lists the commands available to the agent.

        Params:
            agent: The agent for which the commands are being listed.

        Returns:
            str: A string containing a numbered list of commands.
        """
        command_strings = []
        if self.command_registry:
            command_strings += [
                str(cmd) for cmd in self.command_registry.list_available_commands(agent)
            ]

        # Add commands from plugins etc.
        command_strings += [str(cmd) for cmd in self.commands.values()]

        return format_numbered_list(command_strings)
