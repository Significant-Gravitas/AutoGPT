from __future__ import annotations

import enum

from AFAAS.interfaces.agent import AbstractAgent
from AFAAS.interfaces.configuration import AFAASModel

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
            return self.value == other
        else:
            return super().__eq__(other)
