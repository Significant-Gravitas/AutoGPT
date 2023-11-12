from __future__ import annotations

import enum
import importlib
import pkgutil
import random
import string
import uuid
from logging import Logger
from typing import TYPE_CHECKING, Optional, Union

from pydantic import BaseModel, Field

from autogpts.autogpt.autogpt.core.configuration import AFAASModel
from autogpts.autogpt.autogpt.core.tools.schema import ToolResult
logger = Logger(name=__name__)

if TYPE_CHECKING:
    from autogpts.autogpt.autogpt.core.agents import BaseAgent

    from .plan import Plan
    from .tasks import Task


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


class TaskStatus(AFAASModel):
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


class TaskContext(AFAASModel):
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


class BaseTask(AFAASModel):
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
    task_id: str 

    task_goal: str
    """ The title / Name of the task """

    task_context: Optional[str]
    """ Placeholder : Context given by RAG & other elements """

    long_decription: Optional[str] 
    """ Placeholder : A longer description of the task than `task_goal` """
    
    task_text_output : Optional[str] 
    """ Placeholder : The agent summary of his own doing while performing the task"""

    ###
    ### Task Management properties
    ###
    task_history: Optional[list[dict]]
    subtasks: Optional[list[Task]]
    acceptance_criteria: Optional[list[str]]

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def default_command() -> str: 
        try : 
            import autogpts.autogpt.autogpt.core.agents.routing
            return "afaas_routing"
        except :
            return "afaas_make_initial_plan"

    def add_tasks(self, tasks=list[Task], position: int = None):
        if position is not None:
            for tasks in tasks:
                self.subtask.insert(tasks, position)
        else:
            for tasks in tasks:
                self.subtask.append(tasks)


    def __getitem__(self, index: Union[int, str]):
        """
        Get a task from the plan by index or slice notation. This method is an alias for `get_task`.

        Args:
            index (Union[int, str]): The index or slice notation to retrieve a task.

        Returns:
            Task or List[Task]: The task or list of tasks specified by the index or slice.

        Examples:
            >>> plan = Plan([Task("Task 1"), Task("Task 2")])
            >>> plan[0]
            Task(task_goal='Task 1')
            >>> plan[1:]
            [Task(task_goal='Task 2')]

        Raises:
            IndexError: If the index is out of range.
            ValueError: If the index type is invalid.
        """
        return self.get_task_with_index(index)

    def get_task_with_index(self, index: Union[int, str]):
        """
        Get a task from the plan by index or slice notation.

        Args:
            index (Union[int, str]): The index or slice notation to retrieve a task.

        Returns:
            Task or List[Task]: The task or list of tasks specified by the index or slice.

        Examples:
            >>> plan = Plan([Task("Task 1"), Task("Task 2")])
            >>> plan.get_task(0)
            Task(task_goal='Task 1')
            >>> plan.get_task(':')
            [Task(task_goal='Task 1'), Task(task_goal='Task 2')]

        Raises:
            IndexError: If the index is out of range.
            ValueError: If the index type is invalid.
        """
        if isinstance(index, int):
            # Handle positive integers and negative integers
            if -len(self.subtask) <= index < len(self.subtask):
                return self.subtask[index]
            else:
                raise IndexError("Index out of range")
        elif isinstance(index, str) and index.startswith(":") and index[1:].isdigit():
            # Handle notation like ":-1"
            start = 0 if index[1:] == "" else int(index[1:])
            return self.subtask[start:]
        else:
            raise ValueError("Invalid index type")

    ###
    ### FIXME : To test
    ###
    def remove_task(self, task_id: str):
        logger.error("""FUNCTION NOT WORKING :
                     1. We now manage multiple predecessor
                     2. Tasks should not be deleted but managed by state""")
        # 1. Set all task_predecessor_id to null if they reference the task to be removed
        def clear_predecessors(task: Task):
            if task.task_predecessor_id == task_id:
                task.task_predecessor_id = None
            for subtask in task.subtasks or []:
                clear_predecessors(subtask)

        # 2. Remove leaves with status "DONE" if ALL their siblings have this status
        def should_remove_siblings(
            task: Task, task_parent: Optional[Task] = None
        ) -> bool:
            # If it's a leaf and has a parent
            if not task.subtasks and task_parent:
                all_done = all(st.status == "DONE" for st in task_parent.subtasks)
                if all_done:
                    # Delete the Task objects
                    for st in task_parent.subtasks:
                        del st
                    task_parent.subtasks = None  # or []
                return all_done
            # elif task.subtasks:
            #     for st in task.subtasks:
            #         should_remove_siblings(st, task)
            return False

        for task in self.subtask:
            should_remove_siblings(task)
            clear_predecessors(task)

    def get_ready_leaf_tasks(self) -> list[Task]:
        """
        Get tasks that have status "READY", no subtasks, and no task_predecessor_id.

        Returns:
            List[Task]: A list of tasks meeting the specified criteria.
        """
        ready_tasks = []

        def check_task(task: Task):
            if (
                task.status == "READY"
                and not task.subtasks
                and not task.task_predecessor_id
            ):
                ready_tasks.append(task)

            # Check subtasks recursively
            for subtask in task.subtasks or []:
                check_task(subtask)

        # Start checking from the root tasks in the plan
        for task in self.subtask:
            check_task(task)

        return ready_tasks

    def get_first_ready_task(self) -> Optional[Task]:
        """
        Get the first task that has status "READY", no subtasks, and no task_predecessor_id.

        Returns:
            Task or None: The first task meeting the specified criteria or None if no such task is found.
        """

        def check_task(task: Task) -> Optional[Task]:
            if (
                task.status == "READY"
                and not task.subtasks
                and not task.task_predecessor_id
            ):
                return task

            # Check subtasks recursively
            for subtask in task.subtasks or []:
                found_task = check_task(subtask)
                if (
                    found_task
                ):  # If a task is found in the subtasks, return it immediately
                    return found_task
            return None

        # Start checking from the root tasks in the plan
        for task in self.subtask:
            found_task = check_task(task)
            if found_task:
                return found_task

        return None
    
    @staticmethod
    def debug_parse_task(plan: dict) -> str:
        parsed_response = f"Agent Plan:\n"
        for i, task in enumerate(plan.subtask):
            task : Task
            parsed_response += f"{i+1}. {task.task_id} - {task.task_goal}\n"
            parsed_response += f"Description {task.description}\n"
            # parsed_response += f"Task type: {task.type}  "
            # parsed_response += f"Priority: {task.priority}\n"
            parsed_response += f"Predecessors:\n"
            for j, predecessor in enumerate(task.task_predecessors):
                 parsed_response += f"    {j+1}. {predecessor}\n"
            parsed_response += f"Successors:\n"
            for j, succesors in enumerate(task.task_succesors):
                 parsed_response += f"    {j+1}. {succesors}\n"
            parsed_response += f"Acceptance Criteria:\n"
            for j, criteria in enumerate(task.acceptance_criteria):
                parsed_response += f"    {j+1}. {criteria}\n"
            parsed_response += "\n"

        return parsed_response
    
    def dump(self, depth=0) -> dict:
        if depth < 0:
            raise ValueError("Depth must be a non-negative integer.")

        # Initialize the return dictionary
        return_dict = self.dict()

        # Recursively process subtasks up to the specified depth
        if depth > 0 and self.subtasks:
            return_dict["subtasks"] = [
                subtask.dump(depth=depth - 1) for subtask in self.subtasks
            ]

        return return_dict

    def find_task(self, task_id: str):
        """
        Recursively searches for a task with the given task_id in the tree of tasks.
        """
        # Check current task
        if self.task_id == task_id:
            return self

        # If there are subtasks, recursively check them
        if self.subtasks:
            for subtask in self.subtasks:
                found_task = subtask.find_task(task_id = task_id)
                if found_task:
                    return found_task
        return None

    def find_task_path_with_id(self, search_task_id: str):
        """
        Recursively searches for a task with the given task_id and its parent tasks.
        Returns the parent task and all child tasks on the path to the desired task.
        """

        logger.warning("Deprecated : Recommended function is Task.find_task_path()")

        if self.task_id == search_task_id:
            return self

        if self.subtasks:
            for subtask in self.subtasks:
                found_task = subtask.find_task_path_with_id(search_task_id)
                if found_task:
                    return [self] + [found_task]
        return None

    def find_task_path(self):
        """
        Finds the path from this task to the root.
        """
        path = [self]
        current_task = self

        while current_task.task_parent is not None:
            path.append(current_task.task_parent)
            current_task = current_task.task_parent

        return path

    def get_path_structure(self, task) -> str:
        path_to_task = self.find_task_path(task)
        indented_structure = ""

        for i, task in enumerate(path_to_task):
            indented_structure += "  " * i + "-> " + task.name + "\n"

        return indented_structure
