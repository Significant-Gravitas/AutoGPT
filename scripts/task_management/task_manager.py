from task_management.task import Task
from task_management.task_state import TaskState
from config import Singleton

from typing import List, Optional, Union, Dict

# TODO (HunterBunter): Temporaily set this as a singleton, but ideally we should establish a
# better architecture as singleton is an anti pattern.
class TaskManager(metaclass=Singleton):
    def __init__(self):
        self.initial_tasks = []  # type: List[Task]
        self.tasks = {}  # type: Dict[int, Task]
        self.current_task = None  # type: Optional[Task]

    def create_task(self, name: str, state: TaskState = TaskState.NOT_STARTED, parent_task_id: Optional[int] = None) -> Task:
        """
        Create a new task and add it to the tasks list.

        :param name: The name of the task.
        :param state: The initial state of the task.
        :param parent_task_id: The id of the parent task, if any.
        :return: The created task.
        """
        task_id = len(self.tasks) + 1
        parent_task = None
        if parent_task_id is not None:
            parent_task = self.get_task(parent_task_id)
            if parent_task is None:
                raise ValueError(f"Parent task with id {parent_task_id} not found.")

        new_task = Task(task_id, name, state, parent_task)
        self.tasks[task_id] = new_task

        if parent_task is None:
            self.initial_tasks.append(new_task)
        else:
            parent_task.add_subtask(new_task)

        return new_task

    def get_task(self, task_id: int) -> Union[Task, None]:
        """
        Retrieve a task by its id.

        :param task_id: The id of the task.
        :return: The task with the given id or None if not found.
        """
        return self.tasks.get(task_id)

    def update_task_state(self, task_id: int, state: TaskState) -> bool:
        """
        Update the state of a task.

        :param task_id: The id of the task.
        :param state: The new state of the task.
        :return: True if the task was updated successfully, False otherwise.
        """
        task = self.get_task(task_id)
        if task:
            task.update_state(state)
            return True
        else:
            return False

    def get_task_status(self, task_id: int) -> Union[TaskState, None]:
        """
        Get the status of a task.

        :param task_id: The id of the task.
        :return: The status of the task or None if not found.
        """
        task = self.get_task(task_id)
        if task:
            return task.get_status()
        else:
            return None

    def set_current_task(self, task_id: int) -> bool:
        """
        Set the current task by its id.

        :param task_id: The id of the task.
        :return: True if the task was set successfully, False otherwise.
        """
        task = self.get_task(task_id)
        if task:
            self.current_task = task
            return True
        else:
            return False

    def get_task_hierarchy(self) -> str:
        """
        Get a string representation of the task hierarchy of the current_task,
        showing ancestors, siblings, and children.

        :return: A string representation of the current_task hierarchy.

        Example return:
        Ancestors: 1: Parent Task (IN_PROGRESS) -> 3: Subtask A (COMPLETED)
        Current task: 4: Subtask B (IN_PROGRESS)
        Siblings: 3: Subtask A (COMPLETED), 5: Subtask C (NOT_STARTED)
        Children: 6: Subtask B1 (NOT_STARTED), 7: Subtask B2 (NOT_STARTED)
        """
        if not self.current_task:
            return "No current task"

        def task_repr(task: Task) -> str:
            return f"{task.id}: {task.name} ({task.state.name})"

        # Ancestors
        ancestors = []
        parent = self.current_task.parent_task
        while parent:
            ancestors.append(task_repr(parent))
            parent = parent.parent_task

        ancestors_str = " -> ".join(reversed(ancestors))

        # Siblings
        if self.current_task.parent_task:
            siblings = [task_repr(sibling) for sibling in self.current_task.parent_task.subtasks if sibling.id != self.current_task.id]
        else:
            siblings = [task_repr(sibling) for sibling in self.initial_tasks if sibling.id != self.current_task.id]

        siblings_str = ", ".join(siblings)

        # Children
        children = [task_repr(child) for child in self.current_task.subtasks]

        children_str = ", ".join(children)

        hierarchy_str = f"Ancestors: {ancestors_str}\nCurrent task: {task_repr(self.current_task)}\nSiblings: {siblings_str}\nChildren: {children_str}"
        return hierarchy_str

       