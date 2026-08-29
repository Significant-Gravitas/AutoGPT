from __future__ import annotations

import logging

from pydantic import BaseModel, Field

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

    def __add__(self, other: AIDirectives) -> AIDirectives:
        return AIDirectives(
            resources=self.resources + other.resources,
            constraints=self.constraints + other.constraints,
            best_practices=self.best_practices + other.best_practices,
        ).model_copy(deep=True)
