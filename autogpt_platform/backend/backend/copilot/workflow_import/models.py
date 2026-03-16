"""Data models for competitor workflow import."""

from enum import Enum
from typing import Any

import pydantic
from pydantic import Field


class CompetitorFormat(str, Enum):
    N8N = "n8n"
    MAKE = "make"
    ZAPIER = "zapier"
    UNKNOWN = "unknown"


class StepDescription(pydantic.BaseModel):
    """A single step/node extracted from a competitor workflow."""

    order: int
    action: str
    service: str
    parameters: dict[str, Any] = {}
    connections_to: list[int] = []


class WorkflowDescription(pydantic.BaseModel):
    """Structured description of a competitor workflow."""

    name: str
    description: str
    steps: list[StepDescription]
    trigger_type: str | None = None
    source_format: CompetitorFormat
    raw_json: dict[str, Any] = Field(default_factory=dict, exclude=True)
