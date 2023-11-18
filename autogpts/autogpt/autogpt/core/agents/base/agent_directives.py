from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel


from  autogpts.AFAAS.app.sdk import forge_log
logger = forge_log.ForgeLogger(__name__)

if TYPE_CHECKING:
    from .main import BaseAgent


class BaseAgentPromptSettings(ABC):
    @classmethod
    @abstractmethod
    def load_prompt_settings(cls, erase=False, file=""):
        # Get the directory containing the current class file
        base_agent_dir = os.path.dirname(__file__)
        # Construct the path to the YAML file based on __file__
        current_settings_path = os.path.join(base_agent_dir, "prompt_settings.yaml")

        settings = []

        # Load settings from the current directory (based on __file__)
        if os.path.exists(current_settings_path):
            with open(current_settings_path, "r") as file:
                settings.extend(yaml.safe_load(file))

        # Load settings from the specified directory (based on 'file')
        if file:
            specified_settings_path = os.path.join(
                os.path.dirname(file), "prompt_settings.yaml"
            )

            if os.path.exists(specified_settings_path):
                with open(specified_settings_path, "r") as file:
                    specified_settings = yaml.safe_load(file)
                    for item in specified_settings:
                        if item not in settings:
                            settings.append(item)
                        else:
                            # If the item already exists, update it with specified_settings
                            settings_item = settings.index(item)
                            if erase:
                                settings[settings_item] = specified_settings_path
                            else:
                                settings[settings_item].update(specified_settings[item])

        return settings


class BaseAgentDirectives(BaseModel):
    """An object that contains the basic directives for the AI prompt.

    Attributes:
        constraints (list): A list of constraints that the AI should adhere to.
        resources (list): A list of resources that the AI can utilize.
        best_practices (list): A list of best practices that the AI should follow.
    """

    constraints: list[str]
    resources: list[str]
    best_practices: list[str]

    @staticmethod
    def from_file(agent: BaseAgent) -> BaseAgentDirectives:
        pass

        config_params = agent.prompt_settings

        return BaseAgentDirectives(
            constraints=config_params.get("constraints", []),
            resources=config_params.get("resources", []),
            best_practices=config_params.get("best_practices", []),
        )

    def __add__(self, other: "BaseAgentDirectives") -> "BaseAgentDirectives":
        return BaseAgentDirectives(
            resources=self.resources + other.resources,
            constraints=self.constraints + other.constraints,
            best_practices=self.best_practices + other.best_practices,
        ).copy(deep=True)
