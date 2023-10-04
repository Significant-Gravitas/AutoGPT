from __future__ import annotations

import enum
from typing import Optional, TYPE_CHECKING
from pydantic import BaseModel, Field

from autogpt.core.tools.schema import ToolResult

if TYPE_CHECKING:
    from autogpt.core.agents.base.main import BaseAgent


class TaskType(str, enum.Enum):
    """
    An enumeration representing the type of tasks available.

    Attributes:
    - RESEARCH: Task type represents research work.
    - WRITE: Task type represents writing.
    - EDIT: Task type represents editing.
    - CODE: Task type represents coding.
    - DESIGN: Task type represents designing.
    - TEST: Task type represents testing.
    - PLAN: Task type represents planning.

    Example:
        >>> task = TaskType.RESEARCH
        >>> print(task)
        TaskType.RESEARCH
    """

    RESEARCH: str = "research"
    WRITE: str = "write"
    EDIT: str = "edit"
    CODE: str = "code"
    DESIGN: str = "design"
    TEST: str = "test"
    PLAN: str = "plan"


class TaskStatus(BaseModel):
    """
    Model representing the status of a task.

    Attributes:
    - name: Name of the task status.
    - description: Description of the task status.

    Example:
        >>> status = TaskStatus(name="in_progress", description="Work ongoing")
        >>> print(status)
        in_progress
    """

    name: str
    description: str

    def __str__(self) -> str:
        """
        Returns the name of the task status.

        Example:
            >>> status = TaskStatus(name="in_progress", description="Work ongoing")
            >>> print(status)
            in_progress
        """
        return self.name

    def __repr__(self) -> str:
        """
        Returns the representation of task status in the format: "name description".

        Example:
            >>> status = TaskStatus(name="in_progress", description="Work ongoing")
            >>> repr(status)
            "in_progress Work ongoing"
        """
        return f"{self.name} {self.description}"


class TaskStatusList(str, enum.Enum):
    """
    An enumeration representing the list of possible task statuses.

    Attributes:
    - BACKLOG: Task is not ready.
    - READY: Task is ready.
    - IN_PROGRESS: Task is currently being taken care of.
    - DONE: Task has been achieved.

    Example:
        >>> task_status = TaskStatusList.BACKLOG
        >>> print(task_status)
        TaskStatusList.BACKLOG
    """

    BACKLOG: TaskStatus = TaskStatus(
        name="backlog", description="The task is not ready"
    )
    READY: TaskStatus = TaskStatus(name="ready", description="The task  ready")
    IN_PROGRESS: TaskStatus = TaskStatus(
        name="in_progress", description="The being taken care of"
    )
    DONE: TaskStatus = TaskStatus(name="done", description="The being achieved")

    def __eq__(self, other):
        """
        Overrides the default equality to check for equality with strings or with other Enum values.

        Args:
        - other: The object to compare with.

        Returns:
        - bool: True if equal, False otherwise.

        Example:
            >>> task_status = TaskStatusList.BACKLOG
            >>> task_status == "backlog"
            True
            >>> task_status == TaskStatusList.READY
            False
        """
        if isinstance(other, str):
            return self.value.name == other
        else:
            return super().__eq__(other)


class TaskContext(BaseModel):
    """
    Model representing the context of a task.

    Attributes:
    - cycle_count: Number of cycles (default is 0).
    - status: Status of the task (default is BACKLOG).
    - parent: Parent task (default is None).
    - prior_actions: List of prior actions (default is empty list).
    - memories: List of memories related to the task (default is empty list).
    - user_input: List of user inputs related to the task (default is empty list).
    - supplementary_info: Additional information about the task (default is empty list).
    - enough_info: Flag indicating if enough information is available (default is False).

    Example:
        >>> context = TaskContext(cycle_count=5, status=TaskStatusList.IN_PROGRESS)
        >>> print(context.cycle_count)
        5
    """

    cycle_count: int = 0
    status: TaskStatusList = TaskStatusList.BACKLOG
    parent: "Task" = None
    prior_actions: list[ToolResult] = Field(default_factory=list)
    memories: list = Field(default_factory=list)
    user_input: list[str] = Field(default_factory=list)
    supplementary_info: list[str] = Field(default_factory=list)
    enough_info: bool = False


class Task(BaseModel):
    """
    Model representing a task.

    Attributes:
    - responsible_agent_id: ID of the responsible agent (default is None).
    - objective: Objective of the task.
    - type: Type of the task (corresponds to TaskType, but due to certain issues, it's defined as str).
    - priority: Priority of the task.
    - ready_criteria: List of criteria to consider the task as ready.
    - acceptance_criteria: List of criteria to accept the task.
    - context: Context of the task (default is a new TaskContext).

    Example:
        >>> task = Task(objective="Write a report", type="write", priority=2, ready_criteria=["Gather info"], acceptance_criteria=["Approved by manager"])
        >>> print(task.objective)
        "Write a report"
    """

    responsible_agent_id: Optional[str] = Field(default="")
    objective: Optional[str]
    type: Optional[str]  # TaskType  FIXME: gpt does not obey the enum parameter in its schema
    priority: Optional[int]
    ready_criteria: Optional[list[str]]
    acceptance_criteria: Optional[list[str]]
    context: TaskContext = Field(default_factory=TaskContext)
    subtasks: Optional[list[Task]]

    def dump(self, depth=0) -> dict:
        if depth < 0:
            raise ValueError("Depth must be a non-negative integer.")

        # Initialize the return dictionary
        return_dict = self.dict()

        # Recursively process subtasks up to the specified depth
        if depth > 0 and self.subtasks:
            return_dict["subtasks"] = [subtask.dump(depth=depth - 1) for subtask in self.subtasks]

        return return_dict


# Need to resolve the circular dependency between Task and TaskContext once both models are defined.
Task.update_forward_refs()
# Need to resolve the circular dependency between Task and TaskContext once both models are defined.
TaskContext.update_forward_refs()
