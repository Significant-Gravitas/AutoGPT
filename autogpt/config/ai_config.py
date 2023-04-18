"""
A module that contains the AIConfig class object that contains the configuration
"""
from __future__ import annotations

import os
from typing import Type

import yaml


class AIConfig:
    """
    A class object that contains the configuration information for the AI

    Attributes:
        ai_name (str): The name of the AI.
        ai_role (str): The description of the AI's role.
        ai_goals (list): The list of objectives the AI is supposed to complete.
    """

    def __init__(
        self, ai_name: str = "", ai_role: str = "", ai_goals: list | None = None
    ) -> None:
        """
        Initialize a class instance

        Parameters:
            ai_name (str): The name of the AI.
            ai_role (str): The description of the AI's role.
            ai_goals (list): The list of objectives the AI is supposed to complete.
        Returns:
            None
        """
        if ai_goals is None:
            ai_goals = []
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.ai_goals = ai_goals

    # Soon this will go in a folder where it remembers more stuff about the run(s)
    SAVE_FILE = os.path.join(os.path.dirname(__file__), "..", "ai_settings.yaml")

    @staticmethod
    def load(config_file: str = SAVE_FILE) -> AIConfig:
        """
        Returns class object with parameters (ai_name, ai_role, ai_goals) loaded from
          yaml file if yaml file exists,
        else returns class with no parameters.

        Parameters:
           config_file (str): The path to the config yaml file.
             DEFAULT: "../ai_settings.yaml"

        Returns:
            cls (object): An instance of given cls object
        """
        try:
            with open(config_file, encoding="utf-8") as file:
                config_params = yaml.load(file, Loader=yaml.FullLoader)
        except FileNotFoundError:
            config_params = {}

        ai_name = config_params.get("ai_name", "")
        ai_role = config_params.get("ai_role", "")
        ai_goals = config_params.get("ai_goals", [])
        # type: Type[AIConfig]
        return AIConfig(ai_name, ai_role, ai_goals)

    def save(self, config_file: str = SAVE_FILE) -> None:
        """
        Saves the class parameters to the specified file yaml file path as a yaml file.

        Parameters:
            config_file (str): The path to the config yaml file.
              DEFAULT: "../ai_settings.yaml"

        Returns:
            None
        """

        config = {
            "ai_name": self.ai_name,
            "ai_role": self.ai_role,
            "ai_goals": self.ai_goals,
        }
        with open(config_file, "w", encoding="utf-8") as file:
            yaml.dump(config, file, allow_unicode=True)
