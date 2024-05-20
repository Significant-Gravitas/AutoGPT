import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from forge.utils.yaml_validator import validate_yaml_file

logger = logging.getLogger(__name__)


class AIDirectives(BaseModel):
    """An object that contains the basic directives for the AI prompt.

    Attributes:
        constraints (list): A list of constraints that the AI should adhere to.
        resources (list): A list of resources that the AI can utilize.
        best_practices (list): A list of best practices that the AI should follow.
    """

    resources: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    best_practices: list[str] = Field(default_factory=list)

    @staticmethod
    def from_file(prompt_settings_file: Path) -> "AIDirectives":
        from forge.logging.helpers import request_user_double_check

        (validated, message) = validate_yaml_file(prompt_settings_file)
        if not validated:
            logger.error(message, extra={"title": "FAILED FILE VALIDATION"})
            request_user_double_check()
            raise RuntimeError(f"File validation failed: {message}")

        with open(prompt_settings_file, encoding="utf-8") as file:
            config_params = yaml.load(file, Loader=yaml.SafeLoader)

        return AIDirectives(
            constraints=config_params.get("constraints", []),
            resources=config_params.get("resources", []),
            best_practices=config_params.get("best_practices", []),
        )

    def __add__(self, other: "AIDirectives") -> "AIDirectives":
        return AIDirectives(
            resources=self.resources + other.resources,
            constraints=self.constraints + other.constraints,
            best_practices=self.best_practices + other.best_practices,
        ).copy(deep=True)
