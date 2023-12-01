from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

from autogpts.autogpt.autogpt.core.agents import BaseAgent

from ...sdk.forge_log import ForgeLogger
# from .plan import Plan
from .base import BaseTask
from .meta import TaskStatusList

# from autogpts.autogpt.autogpt.core.configuration import AFAASModel
# from autogpts.autogpt.autogpt.core.tools.schema import ToolResult



LOG = ForgeLogger(__name__)

if TYPE_CHECKING:
    from .stack import TaskStack


class Task(BaseTask):
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

    ###
    ### GENERAL properties
    ###

    task_id: str = Field(default_factory=lambda: Task.generate_uuid())
    task_parent: BaseTask
    _task_parent_id: str

    # task_predecessors: Optional[list[Task]]  = []
    # _task_predecessors_id: Optional[list[str]]  = []
    _task_predecessors: Optional[TaskStack] = None

    @property
    def task_predecessors(self) -> TaskStack:
        if self._task_predecessors is None:
            from .stack import TaskStack

            self._task_predecessors = TaskStack(parent_task=self)
        return self._task_predecessors

    # task_successors: Optional[list[Task]]  = []
    # _task_successors_id: Optional[list[str]] = []
    _task_successors: Optional[TaskStack] = None

    @property
    def task_successors(self) -> TaskStack:
        if self._task_successors is None:
            from .stack import TaskStack

            self._task_successors = TaskStack(parent_task=self)
        return self._task_successors

    state: Optional[TaskStatusList] = Field(default=TaskStatusList.BACKLOG.value)

    task_text_output: Optional[str]
    """ Placeholder : The agent summary of his own doing while performing the task"""

    ###
    ### Task Management properties
    ###
    task_history: Optional[list[dict]]

    command: Optional[str] = Field(default_factory=lambda: Task.default_command())
    arguments: Optional[dict] = Field(default={})

    class Config(BaseTask.Config):
        default_exclude = set(BaseTask.Config.default_exclude) | {
            # If commented create an infinite loop
            "task_parent",
            "task_predecessors",
            "task_successors",
        }

    @property
    def plan_id(self) -> str:
        return self.agent.plan.plan_id

    @staticmethod
    def generate_uuid():
        return "T" + str(uuid.uuid4())

    @classmethod
    def get_task_from_db(cls, task_id: str, agent: BaseAgent) -> Task:
        memory = agent._memory
        task_table = memory.get_table("tasks")
        task = task_table.get(task_id)
        return cls(**task, agent=agent)

    @classmethod
    def create_in_db(cls, task: Task, agent: BaseAgent):
        memory = agent._memory
        task_table = memory.get_table("tasks")
        task_table.add(value=task, id=task.task_id)

    def save_in_db(self):
        from autogpts.autogpt.autogpt.core.memory.table import AbstractTable

        memory = self.agent._memory
        task_table: AbstractTable = memory.get_table("tasks")
        task_table.update(
            value=self,
            task_id=self.task_id,
            plan_id=self.plan_id,
        )

    def __setattr__(self, key, value):
        # Set attribute as normal
        super().__setattr__(key, value)
        # If the key is a model field, mark the instance as modified
        if key in self.__fields__:
            self.agent.plan._register_task_as_modified(task_id=self.task_id)

    def find_task_path(self) -> list[Task]:
        """
        Finds the path from this task to the root.
        """
        path = [self]
        current_task: BaseTask = self

        while (
            hasattr(current_task, "task_parent")
            and current_task.task_parent is not None
        ):
            path.append(current_task.task_parent)
            current_task = current_task.task_parent

        return path

    def get_path_structure(self) -> str:
        path_to_task = self.find_task_path()
        indented_structure = ""

        for i, task in enumerate(path_to_task):
            indented_structure += "  " * i + "-> " + task.task_goal + "\n"

        return indented_structure


# Need to resolve the circular dependency between Task and TaskContext once both models are defined.
Task.update_forward_refs()
