# sourcery skip: do-not-use-staticmethod
"""
A module that contains the AIConfig class object that contains the configuration
"""
from __future__ import annotations

import os
import platform
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import distro
import yaml

if TYPE_CHECKING:
    from autogpt.models.command_registry import CommandRegistry
    from autogpt.prompts.generator import PromptGenerator


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

    def to_dict(self) -> dict:
        """
        Converts the AIConfig instance to a dictionary.

        Returns:
            dict: A dictionary representation of the AIConfig instance.
        """

        return {
            "ai_name": self.ai_name,
            "ai_role": self.ai_role,
            "ai_goals": self.ai_goals,
            "api_budget": self.api_budget,
            "plugins": self.plugins,
        }


    @staticmethod
    def update_old_config(ai_settings_file: str) -> Tuple[Dict[str, Dict[str, Any]], str]:
        """
        Checks and updates the provided configuration file to the latest format supporting multiple configurations.

        Parameters:
        ai_settings_file (str): The path to the configuration file.

        Returns:
        Tuple[Dict[str, Dict[str, Any]], str]: A dictionary of updated configurations and a status message regarding the transformation.
        """
        message = "no configuration(s) detected."
        config_updated = False

        if not os.path.exists(ai_settings_file):
            return {"configs": {}}, message

        with open(ai_settings_file, "r", encoding="utf-8") as file:
            all_configs = yaml.safe_load(file)
            # Handle the case when the file exists but is empty
            if all_configs is None:
                all_configs = {"configs": {}}
                config_updated = True
            else:
                # Check if the file is in the old format
                if "configs" not in all_configs:
                    if (
                        "ai_name" not in all_configs
                        or not all_configs["ai_name"].strip()
                    ):
                        # ai_name is missing or empty, treat the file as being empty
                        all_configs = {"configs": {}}
                        message = "no configuration(s) detected."
                        config_updated = True
                    else:
                        # The file is in the old format, transform it
                        message = f"successfully transformed to support multiple configurations."
                        old_config = all_configs

                        # Extract old configuration data
                        ai_name = old_config["ai_name"]
                        ai_role = old_config.get("ai_role", None)
                        ai_goals = old_config.get("ai_goals", None)
                        api_budget = old_config.get("api_budget", None)
                        # Create a new AIConfig instance with the old values
                        ai_config = AIConfig(
                            ai_name, ai_role, ai_goals, api_budget, plugins=[]
                        )

                        # Overwrite the old configuration with the new one
                        try:
                            ai_config.save(ai_settings_file, append=False)
                        except ValueError as e:
                            print(f"An error occurred: {e}")

                        # Convert the AIConfig instance to a dictionary
                        all_configs_dict = {ai_name: ai_config.to_dict()}

                        all_configs = {"configs": all_configs_dict}

                else:
                    # The file is not in the old format
                    message = f"healthy."

        # Save the updated configuration file if there was any update
        if config_updated:
            with open(ai_settings_file, "w", encoding="utf-8") as file:
                yaml.safe_dump(all_configs, file)

        return all_configs, message

    @staticmethod
    def load(ai_name: str, ai_settings_file: str) -> Tuple[Optional["AIConfig"], str]:
        """
        Load a specific AI configuration from the config file and return a status message.

        Args:
            ai_name (str): The name of the AI configuration to load.
            ai_settings_file (str): The path to the configuration file.

        Returns:
            Tuple[Optional[AIConfig], str]: The loaded AI configuration and a status message.
                                           If no such configuration exists, returns None and the status message.
        """
        all_configs, message = AIConfig.load_all(ai_settings_file)

        if all_configs is None:
            return None, message

        if ai_name in all_configs:
            return all_configs[ai_name], message
        else:
            return None, message

    @staticmethod
    def load_all(ai_settings_file: str) -> Tuple[Dict[str, "AIConfig"], str]:
        """
        Load all AI configurations from the specified config file and returns a status message.

        Parameters:
            ai_settings_file (str): The path to the configuration file.

        Returns:
            Tuple[Dict[str, AIConfig], str]: A tuple containing a dictionary of all loaded AI configurations and a string indicating the status of the operation.
        """
        all_configs, message = AIConfig.update_old_config(ai_settings_file)

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

        return ai_configs, message

    def save(
        self,
        ai_settings_file: str,
        append: bool = False,
        old_ai_name: Optional[str] = None,
    ) -> None:
        """
        Saves the class parameters to the specified file YAML file path as a YAML file.

        Parameters:
            ai_settings_file(str): The path to the config YAML file.
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

        config = {
            self.ai_name: {
                "ai_role": self.ai_role,
                "ai_goals": self.ai_goals,
                "api_budget": self.api_budget,
                "plugins": self.plugins,
            }
        }

        if not append or not os.path.exists(ai_settings_file):
            all_configs: Dict[str, Dict[str, Any]] = {"configs": {}}
        else:
            with open(ai_settings_file, "r", encoding="utf-8") as file:
                all_configs = yaml.safe_load(file)

                # Handle the case when the file exists but is empty
                if all_configs is None:
                    all_configs = {"configs": {}}

        # Remove old configuration if old_ai_name is provided
        if old_ai_name and old_ai_name in all_configs["configs"]:
            del all_configs["configs"][old_ai_name]

        # Append the new config
        all_configs["configs"].update(config)

        with open(ai_settings_file, "w", encoding="utf-8") as file:
            file.write(yaml.dump(all_configs, allow_unicode=True))

    def delete(self, ai_settings_file: str, ai_name: str = "") -> None:
        """
        Deletes a configuration from the specified YAML file.

        Parameters:
            ai_settings_file(str): The path to the config YAML file.
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
        if not os.path.exists(ai_settings_file):
            raise ValueError("No configurations to delete.")
        else:
            with open(ai_settings_file, "r", encoding="utf-8") as file:
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
        with open(ai_settings_file, "w", encoding="utf-8") as file:
            file.write(yaml.dump(all_configs, allow_unicode=True))


    def construct_full_prompt(
        self, config, prompt_generator: Optional[PromptGenerator] = None
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

        from autogpt.prompts.prompt import build_default_prompt_generator

        if prompt_generator is None:
            prompt_generator = build_default_prompt_generator(config)
        prompt_generator.goals = self.ai_goals
        prompt_generator.name = self.ai_name
        prompt_generator.role = self.ai_role
        prompt_generator.command_registry = self.command_registry
        for plugin in config.plugins:
            if not plugin.can_handle_post_prompt():
                continue
            prompt_generator = plugin.post_prompt(prompt_generator)

        if config.execute_local_commands:
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
