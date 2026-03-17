"""Data models for external workflow import."""

from enum import Enum
from typing import Any

import pydantic


class SourcePlatform(str, Enum):
    N8N = "n8n"
    MAKE = "make"
    ZAPIER = "zapier"
    UNKNOWN = "unknown"


class Connection(pydantic.BaseModel):
    """A typed connection between two workflow steps."""

    target_step: int
    connection_type: str  # e.g. "main", "ai_tool", "ai_memory", "ai_languageModel"


class StepDescription(pydantic.BaseModel):
    """A single step/node extracted from an external workflow."""

    order: int
    action: str
    service: str
    parameters: dict[str, Any] = pydantic.Field(default_factory=dict)
    typed_connections: list[Connection] = pydantic.Field(default_factory=list)


class WorkflowDescription(pydantic.BaseModel):
    """Structured description of an external workflow."""

    name: str
    description: str
    steps: list[StepDescription]
    trigger_type: str | None = None
    source_format: SourcePlatform
