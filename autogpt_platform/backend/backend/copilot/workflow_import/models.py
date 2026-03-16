"""Data models for external workflow import."""

from enum import Enum
from typing import Any

import pydantic
from pydantic import Field


class SourcePlatform(str, Enum):
    N8N = "n8n"
    MAKE = "make"
    ZAPIER = "zapier"
    UNKNOWN = "unknown"


class StepDescription(pydantic.BaseModel):
    """A single step/node extracted from an external workflow."""

    order: int
    action: str
    service: str
    parameters: dict[str, Any] = {}
    connections_to: list[int] = []


class WorkflowDescription(pydantic.BaseModel):
    """Structured description of an external workflow."""

    name: str
    description: str
    steps: list[StepDescription]
    trigger_type: str | None = None
    source_format: SourcePlatform
    raw_json: dict[str, Any] = Field(default_factory=dict, exclude=True)
