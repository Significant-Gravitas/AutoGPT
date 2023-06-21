from typing import Any
from pydantic import BaseModel


class Knowledge(BaseModel):
    content: str
    content_type: str
    content_metadata: dict[str, Any]


class AbilityResult(BaseModel):
    """The AbilityResult is a standard response struct for an ability."""
    ability_name: str
    ability_args: dict[str, str]
    success: bool
    message: str
    new_knowledge: Knowledge
