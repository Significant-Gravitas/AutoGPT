from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field, validator

from AFAAS.interfaces.agent import BaseAgent

from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.interfaces.task.base import AbstractBaseTask
from AFAAS.interfaces.task.meta import TaskStatusList
from AFAAS.prompts.common import AFAAS_SMART_RAG_Strategy

LOG = AFAASLogger(name=__name__)

if TYPE_CHECKING:
    from AFAAS.interfaces.task.stack import TaskStack


class Task(AbstractBaseTask):
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
    plan_id: str = Field()

    command: Optional[str] = Field(default_factory=lambda: Task.default_command())
    arguments: Optional[dict] = Field(default={})

    _task_parent_id: str = Field()

    @property
    def task_parent(self) -> AbstractBaseTask:
        LOG.trace(
            f"{self.debug_formated_str(True)} {self.__class__.__name__}.task_parent({self._task_parent_id})"
        )
        try:
            # Lazy load the parent task
            return self.agent.plan.get_task(self._task_parent_id)
        except KeyError:
            raise ValueError(f"No parent task found with ID {self._task_parent_id}")

    @task_parent.setter
    def task_parent(self, task: AbstractBaseTask):
        LOG.trace(f"{self.__class__.__name__}.task_parent.setter()")
        if not isinstance(task, Task):
            raise ValueError("task_parent must be an instance of Task")
        self._task_parent_id = task.task_id

    _task_predecessors: Optional[TaskStack] #= Field(default=None)
    _task_successors: Optional[TaskStack] #= Field(default=None)
    @property
    def task_predecessors(self) -> TaskStack:
        if self._task_predecessors is None:
            from AFAAS.interfaces.task.stack import TaskStack

            self._task_predecessors = TaskStack(
                parent_task=self, description="Predecessors"
            )
        return self._task_predecessors

    @property
    def task_successors(self) -> TaskStack:
        if self._task_successors is None:
            from AFAAS.interfaces.task.stack import TaskStack

            self._task_successors = TaskStack(
                parent_task=self, description="Successors"
            )
        return self._task_successors

    state: Optional[TaskStatusList] = Field(default=TaskStatusList.BACKLOG)

    @validator("state", pre=True, always=True)
    def set_state(cls, new_state, values):
        task_id = values.get("task_id")
        if task_id and new_state:
            LOG.debug(f"Setting state of task {task_id} to {new_state}")
            # Assuming LOG and agent are defined and accessible
            agent = values.get("agent")
            if agent:
                agent.plan._registry_update_task_status_in_list(
                    task_id=task_id, status=new_state
                )
        else:
            LOG.error(f"Task {task_id} has state is None")
        return new_state

    def __setattr__(self, key, value):
        # Set attribute as normal
        super().__setattr__(key, value)
        # If the key is a model field, mark the instance as modified
        if key in self.__fields__:
            self.agent.plan._register_task_as_modified(task_id=self.task_id)

        if key == "state":
            self.agent.plan._registry_update_task_status_in_list(
                task_id=self.task_id, status=value
            )

    task_text_output: Optional[str]
    """ Placeholder : The agent summary of his own doing while performing the task"""
    task_text_output_as_uml: Optional[str]
    """ Placeholder : The agent summary of his own doing while performing the task as a UML diagram"""

    class Config(AbstractBaseTask.Config):
        default_exclude = set(AbstractBaseTask.Config.default_exclude) | {
            # If commented create an infinite loop
            "task_parent",
            "task_predecessors",
            "task_successors",
        }

    def __init__(self, **data):
        LOG.trace(f"Entering {self.__class__.__name__}.__init__() : {data['task_goal']}")
        super().__init__(**data)
        LOG.trace(f"Quitting {self.__class__.__name__}.__init__() : {self.task_goal}")

    @property
    def plan_id(self) -> str:
        return self.agent.plan.plan_id

    @staticmethod
    def generate_uuid():
        return "T" + str(uuid.uuid4())

    def is_ready(self) -> bool:
        if (
            len(self.task_predecessors.get_active_tasks_from_stack()) == 0
            and len(self.subtasks.get_active_tasks_from_stack()) == 0
            and (
                self.state == TaskStatusList.BACKLOG
                or self.state == TaskStatusList.READY
            )
        ):
            # NOTE: This remove subtasks stored in the plan as they should not be required anymore
            for task_id in self.subtasks:
                self.agent.plan.unregister_loaded_task(task_id=task_id)

            # NOTE: Normaly the task should already be ready .
            # NOTE: Create two different states for ready & ready with active subtasks
            if self.state != TaskStatusList.READY:
                LOG.error(
                    f"Task {self.debug_formated_str()} is ready but not in the ready state. This should not happen."
                )
                self.state = TaskStatusList.READY

            return True

        return False

    def add_predecessor(self, task: Task):
        """
        Adds a predecessor to this task (also automatically adds this task as a successor to the given predecessor task).

        Args:
            task (Task): The task to be added as a predecessor.

        Warning:
            This method should not be used in conjunction with `add_successor` or `_add_predecessor` on the same task objects,
            as it can lead to a recursive loop.
        """
        self.task_predecessors.add(task)
        task._add_successor(self)

    def add_successor(self, task: Task):
        """
        Adds a successor to this task (also automatically adds this task as a successor to the given predecessor task).

        Args:
            task (Task): The task to be added as a successor.

        Warning:
            This method should not be used in conjunction with `add_predecessor` on the same task objects,
            as it can lead to a recursive loop.
        """
        self.task_successors.add(task)
        task._add_predecessor(self)

    def _add_successor(self, task: Task):
        """
        DO NOT USE : This method should only be used within `Task.add_predecessor()`
        """
        self.task_successors.add(task)

    def _add_predecessor(self, task: Task):
        """
        DO NOT USE : This method should only be used within `Task.add_successors()`
        """
        self.task_predecessors.add(task)

    #############################################################################################
    #############################################################################################
    #############################################################################################
    # region Task DB Management
    #############################################################################################
    #############################################################################################
    #############################################################################################

    @classmethod
    def get_task_from_db(cls, task_id: str, agent: BaseAgent) -> Task:
        memory = agent.memory
        task_table = memory.get_table("tasks")
        task = task_table.get(task_id=task_id, plan_id=agent.plan.plan_id)
        return cls(**task, agent=agent)

    @classmethod
    def create_in_db(cls, task: Task, agent: BaseAgent):
        memory = agent.memory
        task_table = memory.get_table("tasks")
        task_table.add(value=task, id=task.task_id)

    def save_in_db(self):
        from AFAAS.core.db.table import AbstractTable

        memory = self.agent.memory
        task_table: AbstractTable = memory.get_table("tasks")
        task_table.update(
            value=self,
            task_id=self.task_id,
            plan_id=self.plan_id,
        )

    # endregion

    def get_task_path(self, task_to_root=False, include_self=False) -> list[Task]:
        """
        Finds the path from the root to the task ( not including the task itself by default)
        If task_to_root is True, the path will be from the task to the root.
        If include_self is True, the task will be included in the path.
        """
        path: list[Task] = []
        if include_self:
            path.append(self)

        current_task: Task = self

        while (
            hasattr(current_task, "task_parent")
            and current_task.task_parent is not None
        ):
            path.append(current_task.task_parent)
            current_task = current_task.task_parent

        if not task_to_root:
            path.reverse()

        return path

    def get_formated_task_path(self) -> str:
        path_to_task = self.get_task_path()
        indented_structure = ""

        for i, task in enumerate(path_to_task):
            indented_structure += "  " * i + "-> " + task.debug_formated_str() + "\n"

        return indented_structure

    def get_sibblings(self) -> list[Task]:
        """
        Finds the sibblings of this task.
        """
        if self.task_parent is None:
            return []

        return self.task_parent.subtasks.get_all_tasks_from_stack()

    def __hash__(self):
        return hash(self.task_id)

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.task_id == other.task_id
        return False

    async def prepare_rag(
        self,
        predecessors: bool = True,
        successors: bool = False,
        history: int = 10,
        sibblings=True,
        path=True,
        similar_tasks: int = 0,
        avoid_redondancy: bool = False,
    ):
        return
        plan_history: list[Task] = []
        if history > 0:
            plan_history = self.agent.plan.get_last_achieved_tasks(count=history)
        task_predecessors: list[Task] = []
        if predecessors:
            task_predecessors = self.task_predecessors.get_all_tasks_from_stack()

        history_and_predecessors = set(plan_history) | set(task_predecessors)

        task_path: list[Task] = []
        if path:
            if avoid_redondancy:
                task_path = list(set(self.get_task_path()) - history_and_predecessors)
            else:
                task_path = self.get_task_path()

        task_sibblings: list[Task] = []
        if sibblings:
            if avoid_redondancy:
                task_sibblings = list(
                    set(self.get_sibblings()) - history_and_predecessors
                )
            else:
                task_sibblings = self.get_sibblings()

        # TODO: Build it in a Pipeline for Autocorrection
        task_history = list(history_and_predecessors)
        task_history.sort(key=lambda task: task.modified_at)
        rv: str = await self.agent.execute_strategy(
            strategy_name=AFAAS_SMART_RAG_Strategy.STRATEGY_NAME,
            task=self,
            task_history=task_history,
            task_sibblings=task_sibblings,
            task_path=task_path,
            related_tasks=None,
        )

        self.task_context = rv.parsed_result[0]["command_args"]["resume"]
        self.long_description = rv.parsed_result[0]["command_args"]["long_description"]
        return rv


# Need to resolve the circular dependency between Task and TaskContext once both models are defined.
Task.update_forward_refs()
