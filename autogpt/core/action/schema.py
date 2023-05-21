from pydantic import BaseModel, Field

from autogpt.core.planning.schema import LanguageModelClassification


class ActionRequirements(BaseModel):
    packages: list[str] = Field(default_factory=list)
    language_model_provider: LanguageModelClassification = None
    memory_provider: bool = False
    workspace: bool = False


class ActionResult(BaseModel):
    success: bool
    message: str
    data: str


class ACTION_ARGUMENTS:
    filename = '"filename": "<filename>"'
