from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Optional, Union, get_args

from pydantic import Field

from AFAAS.configs.schema import AFAASModel
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.lib.sdk.logger import AFAASLogger

from .meta import TaskStatusList

# from AFAAS.interfaces.tools.schema import ToolResult
LOG = AFAASLogger(name=__name__)

if TYPE_CHECKING:
    from .stack import TaskStack
    from .task import AbstractTask


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
        default_exclude = set(AFAASModel.Config.default_exclude) | {
            "subtasks",
            "agent",
            "_loaded_tasks_dict",
        }
        json_encoders = AFAASModel.Config.json_encoders | {}

    ###
    ### GENERAL properties
    ###

    agent: BaseAgent = Field(exclude=True)

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
    def default_tool(cls) -> str:
        if cls._default_command is not None:
            return cls._default_command

        try:
            pass

            cls._default_command = "afaas_routing"
        except:
            cls._default_command = "afaas_make_initial_plan"

        return cls._default_command

    def __init__(self, **data):
        LOG.trace(f"{self.__class__.__name__}.__init__()")
        super().__init__(**data)

        # FIXME: Make it dynamic as in AFAAS.lib.message_common
        from AFAAS.interfaces.task.stack import TaskStack

        if "_task_predecessors" in data and isinstance(
            data["_task_predecessors"], list
        ):
            self._task_predecessors = TaskStack(
                parent_task=self, _task_ids=data["_task_predecessors"]
            )
        else:
            self._task_predecessors = TaskStack(parent_task=self, _task_ids=[])
        if "_task_successors" in data and isinstance(data["_task_successors"], list):
            self._task_successors = TaskStack(
                parent_task=self, _task_ids=data["_task_successors"]
            )
        else:
            self._task_successors = TaskStack(parent_task=self, _task_ids=[])
        if "_subtasks" in data and isinstance(data["_subtasks"], list):
            self._subtasks = TaskStack(parent_task=self, _task_ids=data["_subtasks"])
        else:
            self._subtasks = TaskStack(parent_task=self, _task_ids=[])

    def dict_db(self, **kwargs) -> dict:
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
        from AFAAS.lib.task.plan import Plan

        if isinstance(self, Plan):
            LOG.debug(F"Adding task { task.debug_formated_str(status=True) } to plan")
        elif self.state not in (
            TaskStatusList.READY,
            TaskStatusList.IN_PROGRESS_WITH_SUBTASKS,
        ):
            raise Exception(
                f"Can't add task {task.debug_formated_str(status=True)} to {self.debug_formated_str(status=True)}."
            )
        LOG.debug(
            f"Adding task {task.debug_formated_str()} as subtask of {self.task_id}"
        )
        task._task_parent_id = self.task_id
        task._task_parent = self
        self.subtasks.add(task=task)
        self.agent.plan._register_new_task(task=task)

    def add_tasks(self, tasks: list["AbstractBaseTask"]):
        LOG.debug(f"Adding {len(tasks)} tasks to {self.debug_formated_str()}")
        for task in tasks:
            self.add_task(task=task)

    def __getitem__(self, index: Union[int, str]):
        return self.get_task_with_index(index)

    def get_task_with_index(self, index: Union[int, str]):
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
    async def remove_task(self, task_id: str):
        LOG.error(
            """FUNCTION NOT WORKING :
                     1. We now manage multiple predecessor
                     2. Tasks should not be deleted but managed by state"""
        )

        # 1. Set all task_predecessors_id to null if they reference the task to be removed
        async def clear_predecessors(task: AbstractBaseTask):
            if task_id in task.task_predecessors_id:
                task.task_predecessors_id.remove(task_id)
            for subtask in await task.subtasks.get_all_tasks_from_stack() or []:
                await clear_predecessors(task=subtask)

        # 2. Remove leaves with status "DONE" if ALL their siblings have this status
        async def should_remove_siblings(
            task: AbstractBaseTask, task_parent: Optional[AbstractBaseTask] = None
        ) -> bool:
            # If it's a leaf and has a parent
            if not task.subtasks and task_parent:
                all_done = all(
                    st.state == TaskStatusList.DONE
                    for st in await task_parent.subtasks.get_done_tasks_from_stack()
                )
                if all_done:
                    # Delete the Task objects
                    for st in await task_parent.subtasks.get_all_tasks_from_stack():
                        del st
                    task_parent.subtasks = None  # or []
                return all_done
            # elif task.subtasks:
            #     for st in task.subtasks:
            #         should_remove_siblings(st, task)
            return False

        for task in await self.subtasks.get_all_tasks_from_stack():
            await should_remove_siblings(task=task)
            await clear_predecessors(task=task)

    async def find_ready_tasks(self) -> list[AbstractBaseTask]:
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

        async def check_task(task: AbstractTask):
            if await task.is_ready():
                ready_tasks.append(task)

            if task.state != TaskStatusList.IN_PROGRESS_WITH_SUBTASKS:
                return

            # Check subtasks recursively
            for subtask in await task.subtasks.get_all_tasks_from_stack():
                await check_task(task=subtask)

        # Start checking from the root tasks in the plan
        for task in await self.subtasks.get_all_tasks_from_stack():
            await check_task(task=task)

        return ready_tasks

    async def find_first_ready_task(self) -> Optional[AbstractBaseTask]:
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

        async def check_task(task: AbstractTask) -> Optional[AbstractTask]:
            if await task.is_ready():
                return task

            if task.state != TaskStatusList.IN_PROGRESS_WITH_SUBTASKS:
                return None

            # Check subtasks recursively
            for subtask in await task.subtasks.get_all_tasks_from_stack() or []:
                found_task = await check_task(task=subtask)
                if (
                    found_task is not None
                ):  # If a task is found in the subtasks, return it immediately
                    return found_task
            return None

        # Start checking from the root tasks in the plan
        for task in await self.subtasks.get_all_tasks_from_stack():
            found_task = await check_task(task=task)
            if found_task:
                return found_task

        return None

    async def find_ready_subbranch(
        self, origin: AbstractTask = None
    ) -> list[AbstractBaseTask]:
        ready_tasks = []
        found_ready = False

        async def check_task(task: AbstractTask) -> None:
            nonlocal ready_tasks, found_ready

            if await task.is_ready():
                ready_tasks.append(task)
                found_ready = True

            # If a ready task has already been found in this branch, stop diving into subtasks
            if not found_ready:
                for subtask_id in task.subtasks:
                    await check_task(await self.agent.plan.get_task(subtask_id))

        # Start checking from the root tasks in the plan
        for task_id in self.subtasks:
            if task_id != origin.task_id:
                await check_task(await self.agent.plan.get_task(task_id))
        return ready_tasks

    # def find_task(self, task_id: str):
    #     """
    #     Recursively searches for a task with the given task_id in the tree of tasks.
    #     """
    #     LOG.warning("Deprecated : Recommended function is Plan.get_task()")
    #     # Check current task
    #     if self.task_id == task_id:
    #         return self

    #     # If there are subtasks, recursively check them
    #     if self.subtasks:
    #         for subtask in self.subtasks:
    #             found_task = subtask.find_task(task_id=task_id)
    #             if found_task:
    #                 return found_task
    #     return None

    # def find_task_path_with_id(self, search_task_id: str):
    #     """
    #     Recursively searches for a task with the given task_id and its parent tasks.
    #     Returns the parent task and all child tasks on the path to the desired task.
    #     """

    #     LOG.warning("Deprecated : Recommended function is Task.get_task_path()")

    #     if self.task_id == search_task_id:
    #         return self

    #     if self.subtasks:
    #         for subtask in self.subtasks:
    #             found_task = subtask.find_task_path_with_id(
    #                 search_task_id=search_task_id
    #             )
    #             if found_task:
    #                 return [self] + [found_task]
    #     return None

    #
    @abc.abstractmethod
    async def db_create(self, agent: BaseAgent):
        ...

    def __str__(self):
        return f"{self.task_goal} (id : {self.task_id})"

    def debug_formated_str(self, status=False) -> str:
        status = f"({self.state})" if status else ""
        return f"`{LOG.italic(self.task_goal)}` ({LOG.bold(self.task_id)})" + status

    @staticmethod
    async def debug_info_parse_task(task: AbstractTask) -> str:
        from .task import AbstractTask

        parsed_response = f"Task {task.debug_formated_str()} :\n"
        task: AbstractTask
        for i, task in enumerate(await task.subtasks.get_all_tasks_from_stack()):
            parsed_response += f"{i+1}. {task.debug_formated_str()}\n"
            parsed_response += f"Description {task.long_description}\n"
            parsed_response += f"Predecessors:\n"
            for j, predecessor in enumerate(
                await task.task_predecessors.get_all_tasks_from_stack()
            ):
                parsed_response += f"    {j+1}. {predecessor}\n"
            parsed_response += f"Successors:\n"
            for j, succesors in enumerate(
                await task.task_successors.get_all_tasks_from_stack()
            ):
                parsed_response += f"    {j+1}. {succesors}\n"
            parsed_response += f"Acceptance Criteria:\n"
            for j, criteria in enumerate(task.acceptance_criteria):
                parsed_response += f"    {j+1}. {criteria}\n"
            if LOG.level < LOG.DEBUG:
                parsed_response += f"Task context: {task.task_context}\n"
                parsed_response += f"Status: {task.state}\n"
                parsed_response += f"Task output: {task.task_text_output}\n"
                parsed_response += f"Task history: {task.task_text_output_as_uml}\n"
            parsed_response += "\n"

        return parsed_response

    async def debug_dump(self, depth=0) -> dict:
        if depth < 0:
            raise ValueError("Depth must be a non-negative integer.")

        # Initialize the return dictionary
        return_dict = self.dict()

        # Recursively process subtasks up to the specified depth
        if depth > 0 and self.subtasks:
            return_dict["subtasks"] = [
                subtask.dump(depth=depth - 1)
                for subtask in await self.subtasks.get_all_tasks_from_stack()
            ]

        return return_dict

    async def debug_dump_str(self, depth: int = 0, iteration: int = 0) -> str:
        if depth < 0:
            raise ValueError("Depth must be a non-negative integer.")

        # Initialize the return dictionary
        return_str = self.debug_formated_str() + "\n"

        # Recursively process subtasks up to the specified depth
        if depth > 0 and len(self.subtasks) > 0:
            for i, subtask in enumerate(await self.subtasks.get_all_tasks_from_stack()):
                return_str += (
                    "  " * iteration
                    + f"{i+1}."
                    + await subtask.debug_dump_str(
                        depth=depth - 1, iteration=iteration + 1
                    )
                    + "\n"
                )

        return return_str


# Need to resolve the circular dependency between Task and TaskContext once both models are defined.
AbstractBaseTask.update_forward_refs()
