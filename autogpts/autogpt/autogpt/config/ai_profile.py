from pathlib import Path

import yaml
from pydantic import BaseModel, Field

DEFAULT_AI_NAME = "AutoGPT"
DEFAULT_AI_ROLE = (
    "a seasoned digital assistant: "
    "capable, intelligent, considerate and assertive. "
    "You have extensive research and development skills, and you don't shy "
    "away from writing some code to solve a problem. "
    "You are pragmatic and make the most out of the tools available to you."
)


class AIProfile(BaseModel):
    """
    Object to hold the AI's personality.

    Attributes:
        ai_name (str): The name of the AI.
        ai_role (str): The description of the AI's role.
        ai_goals (list): The list of objectives the AI is supposed to complete.
        api_budget (float): The maximum dollar value for API calls (0.0 means infinite)
    """

    ai_name: str = DEFAULT_AI_NAME
    ai_role: str = DEFAULT_AI_ROLE
    """`ai_role` should fit in the following format: `You are {ai_name}, {ai_role}`"""
    ai_goals: list[str] = Field(default_factory=list[str])

    @staticmethod
    def load(ai_settings_file: str | Path) -> "AIProfile":
        """
        Returns class object with parameters (ai_name, ai_role, ai_goals, api_budget)
        loaded from yaml file if it exists, else returns class with no parameters.

        Parameters:
            ai_settings_file (Path): The path to the config yaml file.

        Returns:
            cls (object): An instance of given cls object
        """

        try:
            with open(ai_settings_file, encoding="utf-8") as file:
                config_params = yaml.load(file, Loader=yaml.SafeLoader) or {}
        except FileNotFoundError:
            config_params = {}

        ai_name = config_params.get("ai_name", DEFAULT_AI_NAME)
        ai_role = config_params.get("ai_role", DEFAULT_AI_ROLE)
        ai_goals = [
            str(goal).strip("{}").replace("'", "").replace('"', "")
            if isinstance(goal, dict)
            else str(goal)
            for goal in config_params.get("ai_goals", [])
        ]

        return AIProfile(ai_name=ai_name, ai_role=ai_role, ai_goals=ai_goals)

    def save(self, ai_settings_file: str | Path) -> None:
        """
        Saves the class parameters to the specified file yaml file path as a yaml file.

        Parameters:
            ai_settings_file (Path): The path to the config yaml file.

        Returns:
            None
        """

        with open(ai_settings_file, "w", encoding="utf-8") as file:
            yaml.dump(self.dict(), file, allow_unicode=True)
