from pydantic import BaseModel
from autogpt.core.agents.simple.lib.models.tasks import Task
from typing import Union


class Plan(BaseModel):
    """
    Represents a plan consisting of a list of tasks.
    """

    tasks: list[Task] = []

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
            return_dict["tasks"] = [task.dump(depth = depth - 1) for task in self.tasks]

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
