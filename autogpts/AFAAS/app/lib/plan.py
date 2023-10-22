from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Union, Optional, List , TYPE_CHECKING
from logging import Logger
import uuid

from .tasks import Task

logger = Logger(name=__name__)

from autogpts.autogpt.autogpt.core.configuration import (
    AFAASModel,
)

class Plan(AFAASModel):
    """
    Represents a plan consisting of a list of tasks.
    """
    plan_id : str = 'PL' + str(uuid.uuid4())
    # def _get_tasks_from_db(self):
    #     return Task.get_from_db(self.plan_id)
    # tasks: list[Task] =  Field(default_factory=_get_tasks_from_db)
    tasks: list[Task] = []

    def add_tasks(self , tasks = list[Task], position : int = None) : 
        if position is not None :
            for tasks in tasks : 
                self.tasks.insert(tasks, position)
        else: 
            for tasks in tasks : 
                self.tasks.append(tasks)

    # def add_task(self , task = list[Task], position : int = None) : 
    #     if position is not None :
    #         self.tasks.insert(task, position)
    #     else : 
    #         self.tasks.append(task)

    def dump(self, depth=0) -> dict:
        """
        Dump the plan and its tasks into a dictionary up to a specified depth.

        Args:
            depth (int): The depth up to which tasks should be included in the dictionary.
                         If depth is 0, only the plan itself is included.

        Examples:
            >>> plan = Plan([Task("Task 1"), Task("Task 2")])
            >>> plan.dump()
            {'tasks': [{'name': 'Task 1'}, {'name': 'Task 2'}]}

        Returns:
            dict: A dictionary representation of the plan and its tasks.
        Raises:
            ValueError: If depth is a negative integer.
        """
        if depth < 0:
            raise ValueError("Depth must be a non-negative integer.")

        # Initialize the return dictionary
        return_dict = self.dict()

        # Recursively process tasks up to the specified depth
        if depth > 0:
            return_dict["tasks"] = [task.dump(depth=depth - 1) for task in self.tasks]

        return return_dict

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
            Task(name='Task 1')
            >>> plan[1:]
            [Task(name='Task 2')]

        Raises:
            IndexError: If the index is out of range.
            ValueError: If the index type is invalid.
        """
        return self.get_task(index)

    def get_task(self, index: Union[int, str]):
        """
        Get a task from the plan by index or slice notation.

        Args:
            index (Union[int, str]): The index or slice notation to retrieve a task.

        Returns:
            Task or List[Task]: The task or list of tasks specified by the index or slice.

        Examples:
            >>> plan = Plan([Task("Task 1"), Task("Task 2")])
            >>> plan.get_task(0)
            Task(name='Task 1')
            >>> plan.get_task(':')
            [Task(name='Task 1'), Task(name='Task 2')]

        Raises:
            IndexError: If the index is out of range.
            ValueError: If the index type is invalid.
        """
        if isinstance(index, int):
            # Handle positive integers and negative integers
            if -len(self.tasks) <= index < len(self.tasks):
                return self.tasks[index]
            else:
                raise IndexError("Index out of range")
        elif isinstance(index, str) and index.startswith(":") and index[1:].isdigit():
            # Handle notation like ":-1"
            start = 0 if index[1:] == "" else int(index[1:])
            return self.tasks[start:]
        else:
            raise ValueError("Invalid index type")

    def find_task(self, task_id: str) -> Optional[Task]:
        """
        Find a task with the given task_id in the list of tasks.
        """
        for task in self.tasks:
            found_task = task.find_task(task_id)
            if found_task:
                return found_task
        return None

    def find_task_path_with_id(self, task_id: str) -> Optional[List[Task]]:
        """
        Find the path to a task with the given task_id in the list of tasks.
        """
        logger.warning("Deprecated : Recommended function is Task.find_task_path()")
        for task in self.tasks:
            path = task.find_task_path_with_id(task_id)
            if path:
                return path
        return None

    ###
    ### NOTE : To test
    ###
    def remove_task(self, task_id: str):
        # 1. Set all task_predecessor_id to null if they reference the task to be removed
        def clear_predecessors(task: Task):
            if task.task_predecessor_id == task_id:
                task.task_predecessor_id = None
            for subtask in task.subtasks or []:
                clear_predecessors(subtask)

        # 2. Remove leaves with status "DONE" if ALL their siblings have this status
        def should_remove_siblings(
            task: Task, parent_task: Optional[Task] = None
        ) -> bool:
            # If it's a leaf and has a parent
            if not task.subtasks and parent_task:
                all_done = all(st.status == "DONE" for st in parent_task.subtasks)
                if all_done:
                    # Delete the Task objects
                    for st in parent_task.subtasks:
                        del st
                    parent_task.subtasks = None  # or []
                return all_done
            # elif task.subtasks:
            #     for st in task.subtasks:
            #         should_remove_siblings(st, task)
            return False

        for task in self.tasks:
            should_remove_siblings(task)
            clear_predecessors(task)

    def get_ready_leaf_tasks(self) -> List[Task]:
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
        for task in self.tasks:
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
        for task in self.tasks:
            found_task = check_task(task)
            if found_task:
                return found_task

        return None

    def generate_pitch(self, task=None):
        if task is None:
            task = self.get_first_ready_task()

        # Extract the task's siblings and path
        siblings = [
            sib
            for sib in self.tasks
            if sib.task_parent_id == task.task_parent_id and sib != task
        ]
        path_to_task = task.find_task_path()

        # Build the pitch
        pitch = """
        # INSTRUCTION
        Your goal is to find the best-suited command in order to achieve the following task: {task_description}

        # CONTEXT
        The high-level plan designed to achieve our goal is: 
        {high_level_plan}
        
        We are working on the task "{task_name}" that consists in: {task_command}. This task is located in:
        {path_structure}
        """.format(
            task_description=task.description,
            high_level_plan="\n".join(
                [
                    "{}: {}".format(t.name, t.description)
                    for t in self.tasks
                    if not t.task_parent_id
                ]
            ),
            task_name=task.name,
            task_command=task.command,  # assuming each task has a 'command' attribute
            path_structure="\n".join(["->".join(p.name for p in path_to_task)]),
        )

        return pitch

    @staticmethod
    def parse_agent_plan(plan: dict) -> str:
        parsed_response = f"Agent Plan:\n"
        for i, task in enumerate(plan.tasks):
            parsed_response += f"{i+1}. {task.name}\n"
            parsed_response += f"Description {task.description}\n"
            parsed_response += f"Task type: {task.type}  "
            parsed_response += f"Priority: {task.priority}\n"
            parsed_response += f"Ready Criteria:\n"
            for j, criteria in enumerate(task.ready_criteria):
                parsed_response += f"    {j+1}. {criteria}\n"
            parsed_response += f"Acceptance Criteria:\n"
            for j, criteria in enumerate(task.acceptance_criteria):
                parsed_response += f"    {j+1}. {criteria}\n"
            parsed_response += "\n"

        return parsed_response

    async def save():
        pass


# # 1. Find the first ready task
# first_ready_task = plan.get_first_ready_task()

# # 2. Retrieve the task and its siblings
# parent_task_id = first_ready_task.task_parent_id
# siblings = []
# if parent_task_id:
#     parent_task = plan.find_task(parent_task_id)
#     siblings = parent_task.subtasks

# # 3. Retrieve the list of tasks on the path
# path_to_task = plan.find_task_path(first_ready_task.task_id)
