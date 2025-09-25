from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ConditionOperator(str, Enum):
    AND = "and"
    OR = "or"


class OptionalBlockConditions(BaseModel):
    """Conditions that determine when a block should be skipped"""

    on_missing_credentials: bool = Field(
        default=False,
        description="Skip block if any required credentials are missing",
    )
    input_flag: Optional[str] = Field(
        default=None,
        description="Name of boolean agent input field that controls skip behavior",
    )
    kv_flag: Optional[str] = Field(
        default=None,
        description="Key-value store flag name that controls skip behavior",
    )
    operator: ConditionOperator = Field(
        default=ConditionOperator.OR,
        description="Logical operator for combining conditions (AND/OR)",
    )


class OptionalBlockConfig(BaseModel):
    """Configuration for making a block optional/skippable"""

    enabled: bool = Field(
        default=False,
        description="Whether this block can be optionally skipped",
    )
    conditions: OptionalBlockConditions = Field(
        default_factory=OptionalBlockConditions,
        description="Conditions that trigger skipping",
    )
    skip_message: Optional[str] = Field(
        default=None,
        description="Custom message to log when block is skipped",
    )


def get_optional_config(node_metadata: dict) -> Optional[OptionalBlockConfig]:
    """Extract optional block configuration from node metadata"""
    if "optional" not in node_metadata:
        return None

    optional_data = node_metadata.get("optional", {})
    if not optional_data:
        return None

    return OptionalBlockConfig(**optional_data)