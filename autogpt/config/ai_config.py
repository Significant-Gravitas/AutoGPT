# sourcery skip: do-not-use-staticmethod
"""
A module that contains the AIConfig class object that contains the configuration
"""
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Optional, Type

import distro
import yaml

from autogpt.prompts.generator import PromptGenerator

class AIConfig:
    """
    A class object that contains the configuration information for the AI

    Attributes:
        project_name (str): The name of the Project.
        ai_name (str): The name of the AI.
        ai_role (str): The description of the AI's role.
        ai_goals (list): The list of objectives the AI is supposed to complete.
        api_budget (float): The maximum dollar value for API calls (0.0 means infinite)
    """

    def __init__(
        self, project_name: str = "",
        ai_name: str = "",
        ai_role: str = "",
        ai_goals: list | None = None,
        api_budget: float = 0.0,
    ) -> None:
        """
        Initialize a class instance

        Parameters:
            project_name (str): The name of the Project.
            ai_name (str): The name of the AI.
            ai_role (str): The description of the AI's role.
            ai_goals (list): The list of objectives the AI is supposed to complete.
            api_budget (float): The maximum dollar value for API calls (0.0 means infinite)
        Returns:
            None
        """
        if ai_goals is None:
            ai_goals = []
        self.project_name = project_name
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.ai_goals = ai_goals
        self.api_budget = api_budget
        self.prompt_generator = None
        self.command_registry = None

    # Save the current session to root
    SESSION_FILE = os.path.join(os.getcwd(), "ai_session.yaml")

    def get_save_file_path(self) -> str:
        """
        Get the save file path for the AI configuration based on the project name and AI name.

        If the project_name is not provided, the file will be saved in the agents folder.
        If the project_name is provided, the file will be saved in the corresponding project folder.

        Returns:
            str: The save file path for the AI configuration.
        """
        if self.project_name:
            formatted_ai_name = self.ai_name.lower().replace(" ", "_")
            formatted_project_name = self.project_name.lower().replace(" ", "_")
            save_file = os.path.join(
                os.getcwd(),
                "projects",
                f"{formatted_project_name}",
                "agents",
                f"ai_{formatted_ai_name}.yaml"
            )
        else:
            formatted_ai_name = self.ai_name.lower().replace(" ", "_")
            save_file = os.path.join(
                os.getcwd(),
                "projects",
                "free_agents",
                f"ai_{formatted_ai_name}.yaml"
            )
        return save_file

    @staticmethod
    def load(config_file: str = None) -> "AIConfig":
        """
        Returns class object with parameters (project_name, ai_name, ai_role, ai_goals, api_budget) loaded from
          yaml file if yaml file exists,
        else returns class with no parameters.

        Parameters:
           config_file (int): The path to the config yaml file.
             DEFAULT: "../ai_session.yaml"

        Returns:
            cls (object): An instance of given cls object
        """
        try:
            with open(config_file, encoding="utf-8") as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            config_params = {}

        if not config_params.get("ai_name", ""):
            return AIConfig()
        
        if not config_params.get("project_name", ""):
            ai_name = config_params.get("ai_name", "")
            ai_role = config_params.get("ai_role", "")
            ai_goals = config_params.get("ai_goals", [])
            api_budget = config_params.get("api_budget", 0.0)
        else:
            project_name = config_params.get("project_name", "")
            ai_name = config_params.get("ai_name", "")
            ai_role = config_params.get("ai_role", "")
            ai_goals = config_params.get("ai_goals", [])
            api_budget = config_params.get("api_budget", 0.0)

        return AIConfig(project_name, ai_name, ai_role, ai_goals, api_budget)

    def save(self, config_file: str = None) -> None:
        """
        Saves the class parameters to the specified file yaml file path as a yaml file.
        If the project_name is not empty, it will save the file in the appropriate
        project directory with the formatted AI name, otherwise it will save in the
        free_agents director.

        Parameters:
            config_file (str): The path to the config yaml file, optional.
            If not provided, it will be determined based on the project_name and ai_name.

        Returns:
            None
        """
        config = {}
        if self.project_name:
            if config_file is None:
                # Format the AI name and project name
                formatted_ai_name = self.ai_name.lower().replace(" ", "_")
                formatted_project_name = self.project_name.lower().replace(" ", "_")

                # Use the get_save_file_path() attribute with the formatted project name and AI name
                config_file = self.get_save_file_path().format(project_name=formatted_project_name, ai_name=formatted_ai_name)
            config = {
                "project_name": self.project_name,
                "ai_name": self.ai_name,
                "ai_role": self.ai_role,
                "ai_goals": self.ai_goals,
                "api_budget": self.api_budget,
            }
        elif config_file is None:
            # If config_file and project_name are not provided, use the default file path
            formatted_ai_name = self.ai_name.lower().replace(" ", "_")
            config_file = self.get_save_file_path().format(ai_name=formatted_ai_name)
            
            config = {
                "ai_name": self.ai_name,
                "ai_role": self.ai_role,
                "ai_goals": self.ai_goals,
                "api_budget": self.api_budget,
        }

        if config_file:  # Ensure the config_file is not empty before calling os.makedirs
            # Create the file if it doesn't exist
            config_directory = os.path.dirname(config_file)
            os.makedirs(config_directory, exist_ok=True)

            with open(config_file, "w", encoding="utf-8") as file:
                yaml.dump(config, file, allow_unicode=True)

            # Save the current session to ai_session.yaml
            self.save_session({'agent': self.get_save_file_path()})

            return config_file
        else:
            raise ValueError("Invalid config_file path")
        
    def save_session(self, config_file: str) -> None:
        """
        Saves the current session to ai_session.yaml.

        Parameters:
            config_file (str): The path to the config yaml file.

        Returns:
            None
        """
        with open(self.SESSION_FILE, "w", encoding="utf-8") as file:
            yaml.dump(config_file, file, allow_unicode=True)

    def construct_full_prompt(
        self, prompt_generator: Optional[PromptGenerator] = None
    ) -> str:
        """
        Returns a prompt to the user with the class information in an organized fashion.

        Parameters:
            None

        Returns:
            full_prompt (str): A string containing the initial prompt for the user
              including the project_name, ai_name, ai_role, ai_goals, and api_budget.
        """

        prompt_start = (
            "Your decisions must always be made independently without"
            " seeking user assistance. Play to your strengths as an LLM and pursue"
            " simple strategies with no legal complications."
            ""
        )

        from autogpt.config import Config
        from autogpt.prompts.prompt import build_default_prompt_generator

        cfg = Config()
        if prompt_generator is None:
            prompt_generator = build_default_prompt_generator()
        prompt_generator.project_name = self.project_name
        prompt_generator.goals = self.ai_goals
        prompt_generator.name = self.ai_name
        prompt_generator.role = self.ai_role
        prompt_generator.command_registry = self.command_registry
        for plugin in cfg.plugins:
            if not plugin.can_handle_post_prompt():
                continue
            prompt_generator = plugin.post_prompt(prompt_generator)

        if cfg.execute_local_commands:
            # add OS info to prompt
            os_name = platform.system()
            os_info = (
                platform.platform(terse=True)
                if os_name != "Linux"
                else distro.name(pretty=True)
            )

            prompt_start += f"\nThe OS you are running on is: {os_info}"

        # Construct full prompt
        full_prompt = f"You are working on {prompt_generator.project_name} as {prompt_generator.name}, {prompt_generator.role}\n{prompt_start}\n\nGOALS:\n\n"
        for i, goal in enumerate(self.ai_goals):
            full_prompt += f"{i+1}. {goal}\n"
        if self.api_budget > 0.0:
            full_prompt += f"\nIt takes money to let you run. Your API budget is ${self.api_budget:.3f}"
        self.prompt_generator = prompt_generator
        full_prompt += f"\n\n{prompt_generator.generate_prompt_string()}"
        return full_prompt
