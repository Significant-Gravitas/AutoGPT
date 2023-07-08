import enum
from typing import Any

from pydantic import BaseModel


class ContentType(str, enum.Enum):
    # TBD what these actually are.
    TEXT = "text"
    CODE = "code"


class Knowledge(BaseModel):
    content: str
    content_type: ContentType
    content_metadata: dict[str, Any]


class AbilityResult(BaseModel):
    """The AbilityResult is a standard response struct for an ability."""

    ability_name: str
    ability_args: dict[str, str]
    success: bool
    message: str
    new_knowledge: Knowledge = None

    def summary(self) -> str:
        kwargs = ", ".join(f"{k}={v}" for k, v in self.ability_args.items())
        return f"{self.ability_name}({kwargs}): {self.message}"
