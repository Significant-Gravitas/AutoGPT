from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional, Union, get_args

from pydantic import BaseModel, Field, validator

from AFAAS.interfaces.agent import AbstractAgent
from AFAAS.configs import AFAASModel
from AFAAS.lib.sdk.logger import AFAASLogger

# from AFAAS.core.tools.schema import ToolResult
LOG = AFAASLogger(name=__name__)

if TYPE_CHECKING:
    from .stack import TaskStack


class AbstractBaseTask(abc.ABC, AFAASModel):
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

    class Config(AFAASModel.Config):
        # This is a list of Field to Exclude during serialization
        default_exclude = set(AFAASModel.Config.default_exclude) | {"subtasks", "agent"}
        json_encoders = AFAASModel.Config.json_encoders | {}

    ###
    ### GENERAL properties
    ###
    if TYPE_CHECKING:
        from AFAAS.interfaces.agent import BaseAgent

    agent: AbstractAgent = Field(exclude=True)

    @property
    def agent_id(self):
        return self.agent.agent_id

    task_id: str

    task_goal: str
    """ The title / Name of the task """

    task_context: Optional[str]
    """ Placeholder : Context given by RAG & other elements """

    long_description: Optional[str]

    ###
    ### Task Management properties
    ###
    task_history: Optional[list[dict]]

    acceptance_criteria: Optional[list[str]] = []

    ###
    ### Dynamic properties
    ###
    _subtasks: Optional[TaskStack] = Field()

    @property
    def subtasks(self) -> TaskStack:
        if self._subtasks is None:
            from .stack import TaskStack

            self._subtasks = TaskStack(parent_task=self, description="Subtasks")
        return self._subtasks
   
    # @validator('_subtasks', pre=True, always=True)
    # def set_subtasks(cls, v, values, **kwargs):
    #     if isinstance(v, dict) and 'task_ids' in v:
    #         # Initialize TaskStack with task_ids and other necessary parameters
    #         return TaskStack(task_ids=v['task_ids'], parent_task=values['self'], description="Subtasks")
    #     elif v is None:
    #         # Initialize an empty TaskStack or handle it as per your requirements
    #         return TaskStack(parent_task=values['self'], description="Subtasks")
    #     else:
    #         # Handle other cases or raise an error
    #         raise ValueError("Invalid value for _subtasks")

    _default_command: str = None

    @classmethod
    def default_command(cls) -> str:
        if cls._default_command is not None:
            return cls._default_command

        try:
            import AFAAS.core.agents.routing

            cls._default_command = "afaas_routing"
        except:
            cls._default_command = "afaas_make_initial_plan"

        return cls._default_command
    
    def __init__(self, **data):
        LOG.trace(f"{self.__class__.__name__}.__init__()")
        super().__init__(**data)
        if '_subtasks' in data and isinstance(data['_subtasks'], list):
            from AFAAS.interfaces.task.stack import TaskStack
            self._subtasks = TaskStack(parent_task = self, _task_ids = data['_subtasks'])

    def dict_memory(self, **kwargs) -> dict:
        d = super().dict(**kwargs)

        # Iterate over each attribute of the dict
        for field, field_info in self.__fields__.items():
            field_value = getattr(self, field)

            if field_value is not None:
                field_type = field_info.outer_type_

                # Direct check for BaseTask instances
                if isinstance(field_value, AbstractBaseTask):
                    d[field] = field_value.task_id

                # Check for lists of BaseTask instances
                if isinstance(field_value, list) and issubclass(
                    get_args(field_type)[0], AbstractBaseTask
                ):
                    # Replace the list of BaseTask instances with a list of their task_ids
                    d[field] = [v.task_id for v in field_value]

        return self._apply_custom_encoders(data=d)

    def add_task(self, task: "AbstractBaseTask"):
        LOG.debug(f"Adding task {self.debug_formated_str()} to {task.debug_formated_str()}")
        self.subtasks.add(task=task)
        self.agent.plan._register_new_task(task=task)

    def add_tasks(self, tasks: list["AbstractBaseTask"]):
        LOG.debug(f"Adding {len(tasks)} tasks to {self.debug_formated_str()}")
        for task in tasks:
            self.add_task(task=task)

    def __getitem__(self, index: Union[int, str]):
        """
        Get a task from the plan by index or slice notation. This method is an alias for `get_task`.

        Args:
            index (Union[int, str]): The index or slice notation to retrieve a task.

        Returns:
            Task or List [BaseTask]: The task or list of tasks specified by the index or slice.

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
            Task or List [BaseTask]: The task or list of tasks specified by the index or slice.

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
            if -len(self.subtasks) <= index < len(self.subtasks):
                return self.subtasks[index]
            else:
                raise IndexError("Index out of range")
        elif isinstance(index, str) and index.startswith(":") and index[1:].isdigit():
            # Handle notation like ":-1"
            start = 0 if index[1:] == "" else int(index[1:])
            return self.subtasks[start:]
        else:
            raise ValueError("Invalid index type")

    ###
    ### FIXME : To test
    ###
    def remove_task(self, task_id: str):
        LOG.error(
            """FUNCTION NOT WORKING :
                     1. We now manage multiple predecessor
                     2. Tasks should not be deleted but managed by state"""
        )

        # 1. Set all task_predecessors_id to null if they reference the task to be removed
        def clear_predecessors(task: AbstractBaseTask):
            if task_id in task.task_predecessors_id:
                task.task_predecessors_id.remove(task_id)
            for subtask in task.subtasks.get_all_tasks_from_stack() or []:
                clear_predecessors(task=subtask)

        # 2. Remove leaves with status "DONE" if ALL their siblings have this status
        def should_remove_siblings(
            task: AbstractBaseTask, task_parent: Optional[AbstractBaseTask] = None
        ) -> bool:
            # If it's a leaf and has a parent
            if not task.subtasks and task_parent:
                all_done = all(st.status == "DONE" for st in task_parent.subtasks.get_done_tasks_from_stack())
                if all_done:
                    # Delete the Task objects
                    for st in task_parent.subtasks.get_all_tasks_from_stack():
                        del st
                    task_parent.subtasks = None  # or []
                return all_done
            # elif task.subtasks:
            #     for st in task.subtasks:
            #         should_remove_siblings(st, task)
            return False

        for task in self.subtasks.get_all_tasks_from_stack():
            should_remove_siblings(task=task)
            clear_predecessors(task=task)

    def find_ready_tasks(self) -> list[AbstractBaseTask]:
        """
        Get tasks that have status "READY", no subtasks, and no task_predecessors_id.

        Returns:
            List [BaseTask]: A list of tasks meeting the specified criteria.
        """
        LOG.notice(
            "Deprecated : Recommended functions are:\n"
            + "- Plan.get_ready_tasks()\n"
            + "- Task.get_first_ready_task()\n"
            + "- Plan.get_next_task()\n"
        )
        ready_tasks = []

        def check_task(task: AbstractBaseTask):
            if (
                task.is_ready()
            ):
                ready_tasks.append(task)

            # Check subtasks recursively
            for subtask in task.subtasks.get_all_tasks_from_stack():
                check_task(task=subtask)

        # Start checking from the root tasks in the plan
        for task in self.subtasks.get_all_tasks_from_stack():
            check_task(task=task)

        return ready_tasks

    def find_first_ready_task(self) -> Optional[AbstractBaseTask]:
        """
        Get the first task that has status "READY", no subtasks, and no task_predecessors_id.

        Returns:
            Task or None: The first task meeting the specified criteria or None if no such task is found.
        """

        LOG.notice(
            "Deprecated : Recommended functions are:\n"
            + "- Plan.get_ready_tasks()\n"
            + "- Plan.get_first_ready_task()\n"
            + "- Plan.get_next_task()\n"
        )

        def check_task(task: AbstractBaseTask) -> Optional[AbstractBaseTask]:
            if (
                task.is_ready()
            ):
                return task

            # Check subtasks recursively
            for subtask in task.subtasks.get_all_tasks_from_stack() or []:
                found_task = check_task(task=subtask)
                if (
                    found_task is not None
                ):  # If a task is found in the subtasks, return it immediately
                    return found_task
            return None

        # Start checking from the root tasks in the plan
        for task in self.subtasks.get_all_tasks_from_stack() :
            found_task = check_task(task=task)
            if found_task:
                return found_task

        return None
        

    def find_ready_branch(self) -> list[AbstractBaseTask]:
        ready_tasks = []

        def check_task(task: AbstractBaseTask, found_ready: bool) -> bool:
            nonlocal ready_tasks
            if task.is_ready():
                ready_tasks.append(task)
                return True

            # If a ready task has already been found in this branch, check only siblings
            if found_ready:
                return False

            # Check subtasks
            for subtask in task.subtasks:
                if check_task(subtask, found_ready):
                    found_ready = True

            return found_ready

        # Start checking from the root tasks in the plan
        for task in self.subtasks:
            if check_task(self.agent.plan.get_task(task), False):
                break  # Break after finding the first ready task and its siblings

        return ready_tasks
        

    def find_task(self, task_id: str):
        """
        Recursively searches for a task with the given task_id in the tree of tasks.
        """
        LOG.warning("Deprecated : Recommended function is Plan.get_task()")
        # Check current task
        if self.task_id == task_id:
            return self

        # If there are subtasks, recursively check them
        if self.subtasks:
            for subtask in self.subtasks:
                found_task = subtask.find_task(task_id=task_id)
                if found_task:
                    return found_task
        return None

    def find_task_path_with_id(self, search_task_id: str):
        """
        Recursively searches for a task with the given task_id and its parent tasks.
        Returns the parent task and all child tasks on the path to the desired task.
        """

        LOG.warning("Deprecated : Recommended function is Task.get_task_path()")

        if self.task_id == search_task_id:
            return self

        if self.subtasks:
            for subtask in self.subtasks:
                found_task = subtask.find_task_path_with_id(search_task_id = search_task_id)
                if found_task:
                    return [self] + [found_task]
        return None

    #
    @abc.abstractmethod
    def create_in_db(self, agent: BaseAgent):
        ...

    def debug_formated_str(self, status = False) -> str:
        status = f"({self.state})" if status else ""
        return f"`{LOG.italic(self.task_goal)}` ({LOG.bold(self.task_id)})" + status
    

    @staticmethod
    def debug_info_parse_task(task: AbstractBaseTask) -> str:
        from .task import Task

        parsed_response = f"Task {task.debug_formated_str()} :\n"
        task: Task
        for i, task in enumerate(task.subtasks.get_all_tasks_from_stack()):
            parsed_response += f"{i+1}. {task.debug_formated_str()}\n"
            parsed_response += f"Description {task.long_description}\n"
            # parsed_response += f"Task type: {task.type}  "
            # parsed_response += f"Priority: {task.priority}\n"
            parsed_response += f"Predecessors:\n"
            for j, predecessor in enumerate(task.task_predecessors.get_all_tasks_from_stack()):
                parsed_response += f"    {j+1}. {predecessor}\n"
            parsed_response += f"Successors:\n"
            for j, succesors in enumerate(task.task_successors.get_all_tasks_from_stack()):
                parsed_response += f"    {j+1}. {succesors}\n"
            parsed_response += f"Acceptance Criteria:\n"
            for j, criteria in enumerate(task.acceptance_criteria):
                parsed_response += f"    {j+1}. {criteria}\n"
            if(LOG.level < LOG.DEBUG):
                parsed_response += f"Task context: {task.task_context}\n"
                parsed_response += f"Status: {task.state}\n"
                parsed_response += f"Task output: {task.task_text_output}\n"
                parsed_response += f"Task history: {task.task_text_output_as_uml}\n"
            parsed_response += "\n"

        return parsed_response

    def debug_dump(self, depth=0) -> dict:
        if depth < 0:
            raise ValueError("Depth must be a non-negative integer.")

        # Initialize the return dictionary
        return_dict = self.dict()

        # Recursively process subtasks up to the specified depth
        if depth > 0 and self.subtasks:
            return_dict["subtasks"] = [
                subtask.dump(depth=depth - 1) for subtask in self.subtasks.get_all_tasks_from_stack()
            ]

        return return_dict
    

    def debug_dump_str(self, depth : int =0 , iteration : int = 0) -> str:
        if depth < 0:
            raise ValueError("Depth must be a non-negative integer.")

        # Initialize the return dictionary
        return_str = self.debug_formated_str() + "\n"

        # Recursively process subtasks up to the specified depth
        if depth > 0 and len(self.subtasks) > 0:
            for i, subtask in enumerate(self.subtasks.get_all_tasks_from_stack()) :
                return_str += "  "* iteration + f"{i+1}."+ subtask.debug_dump_str(depth = depth - 1, iteration = iteration + 1)  + "\n"

        return return_str

# Need to resolve the circular dependency between Task and TaskContext once both models are defined.
AbstractBaseTask.update_forward_refs()
