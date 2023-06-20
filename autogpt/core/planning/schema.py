import enum

from pydantic import BaseModel, Field

from autogpt.core.ability.schema import AbilityResult
from autogpt.core.resource.model_providers.schema import (
    LanguageModelFunction,
    LanguageModelMessage,
    LanguageModelProviderModelResponse,
)


class LanguageModelClassification(str, enum.Enum):
    """The LanguageModelClassification is a functional description of the model.

    This is used to determine what kind of model to use for a given prompt.
    Sometimes we prefer a faster or cheaper model to accomplish a task when
    possible.

    """

    FAST_MODEL: str = "fast_model"
    SMART_MODEL: str = "smart_model"


class LanguageModelPrompt(BaseModel):
    messages: list[LanguageModelMessage]
    functions: list[LanguageModelFunction] = Field(default_factory=list)

    def __str__(self):
        return "\n\n".join([f"{m.role.value}: {m.content}" for m in self.messages])


class LanguageModelResponse(LanguageModelProviderModelResponse):
    """Standard response struct for a response from a language model."""


class TaskType(str, enum.Enum):
    RESEARCH: str = "research"
    WRITE: str = "write"
    EDIT: str = "edit"
    CODE: str = "code"
    DESIGN: str = "design"
    TEST: str = "test"
    PLAN: str = "plan"


class TaskStatus(str, enum.Enum):
    TODO: str = "todo"
    IN_PROGRESS: str = "in_progress"
    DONE: str = "done"


class Task(BaseModel):
    objective: str
    task_type: str  # TaskType  FIXME: gpt does not obey the enum parameter in its schema
    priority: int
    ready_criteria: list[str]
    acceptance_criteria: list[str]
    cycle_count: int = 0
    status: TaskStatus = TaskStatus.TODO
    parent_task: "Task" = None
    prior_actions: list[AbilityResult] = Field(default_factory=list)
