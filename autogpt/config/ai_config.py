# sourcery skip: do-not-use-staticmethod
"""
A module that contains the AIConfig class object that contains the configuration
"""
from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import distro
import yaml

if TYPE_CHECKING:
    from autogpt.commands.command import CommandRegistry
    from autogpt.prompts.generator import PromptGenerator

# Soon this will go in a folder where it remembers more stuff about the run(s)
SAVE_FILE = str(Path(os.getcwd()) / "ai_settings.yaml")


class AIConfig:
    """
    A class object that contains the configuration information for the AI

    Attributes:
        ai_name (str): The name of the AI.
        ai_role (str): The description of the AI's role.
        ai_goals (list): The list of objectives the AI is supposed to complete.
        api_budget (float): The maximum dollar value for API calls (0.0 means infinite)
    """

    def __init__(
        self,
        ai_name: str = "",
        ai_role: str = "",
        ai_goals: list | None = None,
        api_budget: float = 0.0,
    ) -> None:
        """
        Initialize a class instance

        Parameters:
            ai_name (str): The name of the AI.
            ai_role (str): The description of the AI's role.
            ai_goals (list): The list of objectives the AI is supposed to complete.
            api_budget (float): The maximum dollar value for API calls (0.0 means infinite)
        Returns:
            None
        """
        if ai_goals is None:
            ai_goals = []
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.ai_goals = ai_goals
        self.api_budget = api_budget
        self.prompt_generator: PromptGenerator | None = None
        self.command_registry: CommandRegistry | None = None

    @staticmethod
    def load(config_file: str = SAVE_FILE) -> "AIConfig":
        """
        Returns a class object with parameters (ai_name, ai_role, ai_goals, api_budget) loaded from
        a yaml file if the yaml file exists. If ENV_AI_NAME or ENV_AI_ROLE are set in the environment, 
        they will be used instead of ai_name and ai_role from the yaml file.

        If ENV_GOALS_ONLY is set to True in the environment, only the goals specified in the 
        environment variables (PRE_GOAL_1-10 and POST_GOAL_1-10) will be used, ignoring the goals in the yaml file. 
        If ENV_GOALS_ONLY is not set or False, the goals from the yaml file will be used in addition 
        to the environment variable goals.

        If any of the environment variables are empty or not set, 
        it will be ignored.

        Parameters:
            config_file (str): The path to the config yaml file.
              DEFAULT: "../ai_settings.yaml"

        Returns:
            AIConfig (object): An instance of AIConfig object
        """
        try:
            with open(config_file, encoding="utf-8") as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader) or {}
        except FileNotFoundError:
            config_params = {}

        ai_name = os.getenv("ENV_AI_NAME")
        ai_role = os.getenv("ENV_AI_ROLE")
        ai_name = ai_name if ai_name and ai_name.strip() else config_params.get("ai_name", "")
        ai_role = ai_role if ai_role and ai_role.strip() else config_params.get("ai_role", "")
        api_budget = config_params.get("api_budget", 0.0)

        env_goals_only = os.getenv("ENV_GOALS_ONLY") == "True" if os.getenv("ENV_GOALS_ONLY") else False

        # Prepend goals from environment variables
        pre_goals = [
            os.getenv(f"PRE_GOAL_{i}")
            for i in range(1, 11)  # Assuming you have up to 10 pre-goals
            if os.getenv(f"PRE_GOAL_{i}") is not None and os.getenv(f"PRE_GOAL_{i}").strip()
        ]

        # Add goals from config_params or .env
        if not env_goals_only:
            ai_goals = [
                str(goal).strip("{}").replace("'", "").replace('"', "")
                if isinstance(goal, dict)
                else str(goal)
                for goal in config_params.get("ai_goals", [])
            ]
        else:
            ai_goals = []

        # Append goals from environment variables
        post_goals = [
            os.getenv(f"POST_GOAL_{i}")
            for i in range(1, 11)  # Assuming you have up to 10 post-goals
            if os.getenv(f"POST_GOAL_{i}") is not None and os.getenv(f"POST_GOAL_{i}").strip()
        ]

        ai_goals = pre_goals + ai_goals + post_goals

        return AIConfig(ai_name, ai_role, ai_goals, api_budget)

    def save(self, config_file: str = SAVE_FILE) -> None:
        """
        Saves the class parameters to the specified file yaml file path as a yaml file.

        Parameters:
            config_file(str): The path to the config yaml file.
              DEFAULT: "../ai_settings.yaml"

        Returns:
            None
        """

        config = {
            "ai_name": self.ai_name,
            "ai_role": self.ai_role,
            "ai_goals": self.ai_goals,
            "api_budget": self.api_budget,
        }
        with open(config_file, "w", encoding="utf-8") as file:
            yaml.dump(config, file, allow_unicode=True)

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

        cfg = Config()
        if prompt_generator is None:
            prompt_generator = build_default_prompt_generator()
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
        full_prompt = f"You are {prompt_generator.name}, {prompt_generator.role}\n{prompt_start}\n\nGOALS:\n\n"
        for i, goal in enumerate(self.ai_goals):
            full_prompt += f"{i+1}. {goal}\n"
        if self.api_budget > 0.0:
            full_prompt += f"\nIt takes money to let you run. Your API budget is ${self.api_budget:.3f}"
        self.prompt_generator = prompt_generator
        full_prompt += f"\n\n{prompt_generator.generate_prompt_string()}"
        return full_prompt
