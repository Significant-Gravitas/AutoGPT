# sourcery skip: do-not-use-staticmethod
"""
A module that contains the AIConfig class object that contains the configuration
"""
from __future__ import annotations

import os
import platform
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import distro
import yaml

if TYPE_CHECKING:
    from autogpt.models.command_registry import CommandRegistry
    from autogpt.prompts.generator import PromptGenerator

# Soon this will go in a folder where it remembers more stuff about the run(s)


class AIConfig:
    """
    A class object that contains the configuration information for the AI

    Attributes:
        ai_name (str): The name of the AI.
        ai_role (str): The description of the AI's role.
        ai_goals (list): The list of objectives the AI is supposed to complete.
        api_budget (float): The maximum dollar value for API calls (0.0 means infinite)
        plugins (list): The list of objectives the AI is supposed to complete.
    """

    def __init__(
        self,
        ai_name: str = "",
        ai_role: str = "",
        ai_goals: list | None = None,
        api_budget: float = 0.0,
        plugins: Optional[List] = None,
    ) -> None:
        """
        Initialize a class instance

        Parameters:
            ai_name (str): The name of the AI.
            ai_role (str): The description of the AI's role.
            ai_goals (list): The list of objectives the AI is supposed to complete.
            api_budget (float): The maximum dollar value for API calls (0.0 means infinite)
            plugins (list): The list of objectives the AI is supposed to complete.
        Returns:
            None
        """
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.ai_goals = ai_goals or []
        self.api_budget = api_budget
        self.plugins = plugins or []
        self.prompt_generator: PromptGenerator | None = None
        self.command_registry: CommandRegistry | None = None

    @staticmethod
    def load(ai_name: str, config_file: str) -> Optional["AIConfig"]:
        """
        Load a specific AI configuration from the config file.

        Args:
            ai_name (str): The name of the AI configuration to load.
            config_file (str): The path to the configuration file.

        Returns:
            AIConfig: The loaded AI configuration, or None if no such configuration exists.
        """
        all_configs = AIConfig.load_all(config_file)  # type: ignore

        # Handle the case when the file is empty and `load_all` returns None
        if all_configs is None:
            return None

        if ai_name in all_configs:
            return all_configs[ai_name]
        else:
            return None

    @staticmethod
    def load_all(config_file: str) -> Dict[str, "AIConfig"]:
        """
        Load all AI configurations from the specified config file.

        Parameters:
            config_file (str): The path to the configuration file.

        Returns:
            Dict[str, AIConfig]: A dictionary containing all the loaded AI configurations.
        """
        try:
            with open(config_file, encoding="utf-8") as file:
                all_configs: Dict[str, Any] = yaml.safe_load(file) or {}
        except FileNotFoundError:
            all_configs = {}

        # Handle the case when the file is empty and yaml.safe_load returns None
        if all_configs is None:
            return {}

        configs = all_configs.get("configs", {}) or {}

        ai_configs = {}
        for ai_name, config_params in configs.items():
            ai_role = config_params.get("ai_role", "")
            ai_goals = [
                str(goal).strip("{}").replace("'", "").replace('"', "")
                if isinstance(goal, dict)
                else str(goal)
                for goal in config_params.get("ai_goals", [])
            ]
            api_budget = config_params.get("api_budget", 0.0)
            plugins = [
                str(plugin).strip("{}").replace("'", "").replace('"', "")
                if isinstance(plugin, dict)
                else str(plugin)
                for plugin in config_params.get("plugins", [])
            ]
            ai_configs[ai_name] = AIConfig(
                ai_name, ai_role, ai_goals, api_budget, plugins
            )

        return ai_configs

    def save(
        self,
        config_file: str,
        append: bool = False,
        old_ai_name: Optional[str] = None,
    ) -> None:
        """
        Saves the class parameters to the specified file YAML file path as a YAML file.

        Parameters:
            config_file(str): The path to the config YAML file.
                DEFAULT: "../ai_settings.yaml"
            append(bool): Whether to append the new configuration to the existing file.
                If False, the file will be overwritten. If True, the new configuration will be appended.
                DEFAULT: False
            old_ai_name(str, optional): The old AI name. If provided, the function will remove the existing
                configuration for old_ai_name before adding the new configuration.

        Returns:
            None

        Raises:
            ValueError: If the AI name is an empty string.
        """
        # Prevent saving if the ai_name is an empty string
        if not self.ai_name:
            raise ValueError(
                "The AI name cannot be empty. The configuration was not saved."
            )

        new_config = {
            self.ai_name: {
                "ai_role": self.ai_role,
                "ai_goals": self.ai_goals,
                "api_budget": self.api_budget,
                "plugins": self.plugins,
            }
        }

        if not append or not os.path.exists(config_file):
            all_configs: Dict[str, Dict[str, Any]] = {"configs": {}}
        else:
            with open(config_file, "r", encoding="utf-8") as file:
                all_configs = yaml.safe_load(file)

                # Handle the case when the file exists but is empty
                if all_configs is None:
                    all_configs = {"configs": {}}

        # Remove old configuration if old_ai_name is provided
        if old_ai_name and old_ai_name in all_configs["configs"]:
            del all_configs["configs"][old_ai_name]

        # Append the new config
        all_configs["configs"].update(new_config)

        with open(config_file, "w", encoding="utf-8") as file:
            file.write(yaml.dump(all_configs, allow_unicode=True))

    def delete(self, config_file: str, ai_name: str = "") -> None:
        """
        Deletes a configuration from the specified YAML file.

        Parameters:
            config_file(str): The path to the config YAML file.
                DEFAULT: "../ai_settings.yaml"
            ai_name(str): The name of the AI whose configuration is to be deleted.

        Raises:
            ValueError: If no AI name is provided, the configuration file does not exist, the file is empty,
                or no configuration is found for the specified AI.
        """

        # If no AI name is provided, exit
        if ai_name == "":
            raise ValueError(
                "No AI name provided. Please provide an AI name to delete its configuration."
            )

        # Load existing configurations
        if not os.path.exists(config_file):
            raise ValueError("No configurations to delete.")
        else:
            with open(config_file, "r", encoding="utf-8") as file:
                all_configs = yaml.safe_load(file)

                # Handle the case when the file exists but is empty
                if all_configs is None:
                    raise ValueError("No configurations to delete.")

        # Check if AI configuration exists
        if ai_name in all_configs["configs"]:
            del all_configs["configs"][ai_name]
        else:
            raise ValueError(f"No configuration found for AI '{ai_name}'.")

        # Save configurations back to file
        with open(config_file, "w", encoding="utf-8") as file:
            file.write(yaml.dump(all_configs, allow_unicode=True))

    def construct_full_prompt(
        self, prompt_generator: Optional[PromptGenerator] = None
    ) -> str:
        """
        Returns a prompt to the user with the class information in an organized fashion.

        Parameters:
            None

        Returns:
            full_prompt (str): A string containing the initial prompt for the user
              including the ai_name, ai_role, ai_goals, and api_budget.
        """

        prompt_start = (
            "Your decisions must always be made independently without"
            " seeking user assistance. Play to your strengths as an LLM and pursue"
            " simple strategies with no legal complications."
            ""
        )

        from autogpt.config import Config
        from autogpt.prompts.prompt import build_default_prompt_generator

        CFG = Config()
        if prompt_generator is None:
            prompt_generator = build_default_prompt_generator()
        prompt_generator.goals = self.ai_goals
        prompt_generator.name = self.ai_name
        prompt_generator.role = self.ai_role
        prompt_generator.command_registry = self.command_registry
        for plugin in CFG.plugins:
            if not plugin.can_handle_post_prompt():
                continue
            prompt_generator = plugin.post_prompt(prompt_generator)

        if CFG.execute_local_commands:
            # add OS info to prompt
            os_name = platform.system()
            os_info = (
                platform.platform(terse=True)
                if os_name != "Linux"
                else distro.name(pretty=True)
            )

            prompt_start += f"\nThe OS you are running on is: {os_info}"

        # Construct full prompt
        full_prompt = f"You are {prompt_generator.name}, {prompt_generator.role}\n{prompt_start}\n\nGOALS:\n\n"
        for i, goal in enumerate(self.ai_goals):
            full_prompt += f"{i+1}. {goal}\n"
        if self.api_budget > 0.0:
            full_prompt += f"\nIt takes money to let you run. Your API budget is ${self.api_budget:.3f}"
        self.prompt_generator = prompt_generator
        full_prompt += f"\n\n{prompt_generator.generate_prompt_string()}"
        return full_prompt
