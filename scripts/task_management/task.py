from task_management.task_state import TaskState
from typing import List, Optional

class Task:
    def __init__(self, id: int, name: str, state: TaskState = TaskState.NOT_STARTED, parent_task: Optional['Task'] = None):
        self.id = id
        self.name = name
        self.state = state
        self.parent_task = parent_task
        self.subtasks = []  # type: List[Task]
        self.is_leaf_task = True

    def add_subtask(self, subtask: 'Task') -> None:
        """Add a subtask to the current task."""
        self.subtasks.append(subtask)
        self.is_leaf_task = False

    def update_state(self, state: TaskState) -> None:
        """Update the state of the task."""
        self.state = state

    def get_subtasks(self) -> List['Task']:
        """Get the list of subtasks for a task."""
        return self.subtasks

    def get_status(self) -> TaskState:
        """Get the status of the task."""
        return self.state

    def is_leaf(self) -> bool:
        """Check if the task is an end-of-chain task."""
        return self.is_leaf_task