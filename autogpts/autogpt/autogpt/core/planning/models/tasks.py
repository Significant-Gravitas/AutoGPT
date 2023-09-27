import enum
from typing import Optional

from pydantic import BaseModel, Field

from autogpt.core.ability.schema import AbilityResult


class TaskType(str, enum.Enum):
    RESEARCH: str = "research"
    WRITE: str = "write"
    EDIT: str = "edit"
    CODE: str = "code"
    DESIGN: str = "design"
    TEST: str = "test"
    PLAN: str = "plan"


class TaskStatus(BaseModel):
    name : str 
    description : str

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.name} {self.description}"

class TaskStatusList(str, enum.Enum):
    BACKLOG: TaskStatus = TaskStatus(name= "backlog" ,description= "The task is not ready" )
    READY: TaskStatus = TaskStatus(name= "ready" ,description= "The task  ready" )
    IN_PROGRESS: TaskStatus = TaskStatus(name= "in_progress" ,description= "The being taken care of" )
    DONE : TaskStatus = TaskStatus(name= "done" ,description= "The being achieved" )

    def __eq__(self, other) :
        if isinstance(other, str):
           return self.value.name == other
        else : 
            return super().__eq__(other)



class TaskContext(BaseModel):
    cycle_count: int = 0
    status: TaskStatusList = TaskStatusList.BACKLOG
    parent: "Task" = None
    prior_actions: list[AbilityResult] = Field(default_factory=list)
    memories: list = Field(default_factory=list)
    user_input: list[str] = Field(default_factory=list)
    supplementary_info: list[str] = Field(default_factory=list)
    enough_info: bool = False


class Task(BaseModel):
    responsible_agent_id : str
    objective: str
    type: str  # TaskType  FIXME: gpt does not obey the enum parameter in its schema
    priority: int
    ready_criteria: list[str]
    acceptance_criteria: list[str]
    context: TaskContext = Field(default_factory=TaskContext)


# Need to resolve the circular dependency between Task and TaskContext once both models are defined.
TaskContext.update_forward_refs()
