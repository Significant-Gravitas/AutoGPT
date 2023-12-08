from __future__ import annotations
import abc
from typing import Optional, List, Union
from pydantic import BaseModel, Field

class AbstractBaseTask(abc.ABC, BaseModel):
    """
    Abstract base class for tasks.

    This class defines the interface and common behaviors for all task types.
    """

    @abc.abstractmethod
    def add_task(self, task: "AbstractBaseTask"):
        """
        Add a subtask to this task.

        Args:
            task (AbstractBaseTask): The subtask to add.
        """
        pass

    @abc.abstractmethod
    def remove_task(self, task_id: str):
        """
        Remove a subtask by its task ID.

        Args:
            task_id (str): The ID of the task to remove.
        """
        pass

    @abc.abstractmethod
    def get_subtasks(self) -> List[AbstractBaseTask]:
        """
        Get a list of subtasks for this task.

        Returns:
            List[AbstractBaseTask]: A list of subtasks.
        """
        pass

    @abc.abstractmethod
    def execute(self):
        """
        Execute the task. This method should be implemented to define what the task does.
        """
        pass

    def __str__(self):
        return f"AbstractBaseTask(task_id={self.task_id}, task_goal={self.task_goal})"


class AbstractTask(AbstractBaseTask):
    """
    Abstract base class for tasks.
    """

    @abc.abstractmethod
    def is_ready(self) -> bool:
        """
        Checks if the task is ready to be executed.
        """
        pass

    @abc.abstractmethod
    def add_predecessor(self, task: "AbstractTask"):
        """
        Adds a predecessor to this task.
        """
        pass

    @abc.abstractmethod
    def add_successor(self, task: "AbstractTask"):
        """
        Adds a successor to this task.
        """
        pass

    @abc.abstractmethod
    def get_task_from_db(cls, task_id: str, agent: AbstractAgent) -> "AbstractTask":
        """
        Retrieves a task from the database.
        """
        pass

    @abc.abstractmethod
    def save_in_db(self):
        """
        Saves the task to the database.
        """
        pass

    @abc.abstractmethod
    def find_task_path(self) -> List["AbstractTask"]:
        """
        Finds the path from this task to the root.
        """
        pass

    @abc.abstractmethod
    def get_path_structure(self) -> str:
        """
        Gets the structured path of the task in a readable format.
        """
        pass


class AbstractPlan( AbstractTask):
    """
    Abstract base class for plans, which are collections of tasks.
    """

    @abc.abstractmethod
    def get_task(self, task_id: str) -> AbstractBaseTask:
        """
        Retrieves a task from the plan based on its ID.

        Args:
            task_id (str): The ID of the task to retrieve.

        Returns:
            AbstractTask: The task corresponding to the given ID.
        """
        pass

    @abc.abstractmethod
    def get_next_task(self, task: AbstractTask = None) -> AbstractTask:
        """
        Retrieves the next task in the plan based on the current task.

        Args:
            task (Task): The current task.

        Returns:
            AbstractTask: The next task in the plan.
        """
        pass

    @abc.abstractmethod
    def get_ready_tasks(self) -> List[AbstractTask]:
        """
        Retrieves all tasks that are ready for execution.

        Returns:
            List[AbstractTask]: A list of tasks that are ready.
        """
        pass

    @abc.abstractmethod
    def get_active_tasks(self) -> List[AbstractTask]:
        """
        Retrieves all active tasks in the plan.

        Returns:
            List[AbstractTask]: A list of active tasks.
        """
        pass

    @abc.abstractmethod
    def set_task_status(self, task: AbstractTask, status: "TaskStatusList"):
        """
        Sets the status of a task.

        Args:
            task (Task): The task whose status needs to be updated.
            status (TaskStatusList): The new status for the task.
        """
        pass

    @abc.abstractmethod
    def save(self):
        """
        Saves the current state of the plan to the database.
        """
        pass