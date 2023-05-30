import enum

from pydantic import BaseModel, Field

from autogpt.core.planning.schema import LanguageModelClassification


class AbilityRequirements(BaseModel):
    packages: list[str] = Field(default_factory=list)
    language_model_provider: LanguageModelClassification = None
    memory_provider: bool = False
    workspace: bool = False


class AbilityResult(BaseModel):
    success: bool
    message: str
    data: str


class AbilityArguments(str, enum.Enum):
    FILENAME = '"filename": "<filename>"'

