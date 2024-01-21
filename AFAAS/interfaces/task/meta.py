from __future__ import annotations

import enum

from AFAAS.configs.schema import AFAASModel


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
    BACKLOG: TaskStatus = TaskStatus(
        name="backlog", description="The task is not ready"
    )
    READY: TaskStatus = TaskStatus(name="ready", description="The task  ready")
    IN_PROGRESS_WITH_SUBTASKS: TaskStatus = TaskStatus(
        name="ready_with_subtasks",
        description="subtasks of this task are being processed",
    )
    # IN_PROGRESS: TaskStatus = TaskStatus( name="in_progress", description="The being taken care of" )
    DONE: TaskStatus = TaskStatus(name="done", description="The being achieved")

    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other
        else:
            return super().__eq__(other)
