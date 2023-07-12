# sourcery skip: do-not-use-staticmethod
"""
A module that contains the AIConfig class object that contains the configuration
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

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
        ai_goals: list[str] = [],
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
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.ai_goals = ai_goals
        self.api_budget = api_budget
        # HACK: Existing code wants this to be here. Can't fix quite yet.
        self.prompt_generator = None

    @staticmethod
    def load(ai_settings_file: str = SAVE_FILE) -> "AIConfig":
        """
        Returns class object with parameters (ai_name, ai_role, ai_goals, api_budget)
        loaded from yaml file if yaml file exists, else returns class with no parameters.

        Parameters:
            ai_settings_file (int): The path to the config yaml file.
              DEFAULT: "../ai_settings.yaml"

        Returns:
            cls (object): An instance of given cls object
        """

        try:
            with open(ai_settings_file, encoding="utf-8") as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader) or {}
        except FileNotFoundError:
            config_params = {}

        ai_name = config_params.get("ai_name", "")
        ai_role = config_params.get("ai_role", "")
        ai_goals = [
            str(goal).strip("{}").replace("'", "").replace('"', "")
            if isinstance(goal, dict)
            else str(goal)
            for goal in config_params.get("ai_goals", [])
        ]
        api_budget = config_params.get("api_budget", 0.0)
        return AIConfig(ai_name, ai_role, ai_goals, api_budget)

    def save(self, ai_settings_file: str = SAVE_FILE) -> None:
        """
        Saves the class parameters to the specified file yaml file path as a yaml file.

        Parameters:
            ai_settings_file(str): The path to the config yaml file.
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
        with open(ai_settings_file, "w", encoding="utf-8") as file:
            yaml.dump(config, file, allow_unicode=True)
