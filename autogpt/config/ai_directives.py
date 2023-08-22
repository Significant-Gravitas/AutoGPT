from __future__ import annotations

import logging
from dataclasses import dataclass

import yaml

from autogpt.logs.helpers import request_user_double_check
from autogpt.utils import validate_yaml_file

logger = logging.getLogger(__name__)


@dataclass
class AIDirectives:
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
    def from_file(prompt_settings_file: str) -> AIDirectives:
        (validated, message) = validate_yaml_file(prompt_settings_file)
        if not validated:
            logger.error(message, extra={"title": "FAILED FILE VALIDATION"})
            request_user_double_check()
            exit(1)

        with open(prompt_settings_file, encoding="utf-8") as file:
            config_params = yaml.load(file, Loader=yaml.FullLoader)

        return AIDirectives(
            constraints=config_params.get("constraints", []),
            resources=config_params.get("resources", []),
            best_practices=config_params.get("best_practices", []),
        )
