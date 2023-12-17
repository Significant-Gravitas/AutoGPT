from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Generator
from pydantic import Field

#from AFAAS.interfaces.agent import AbstractAgent
from AFAAS.configs import AFAASModel
from AFAAS.interfaces.task.base import AbstractBaseTask
from AFAAS.interfaces.task.plan import AbstractPlan
from AFAAS.interfaces.task.task import AbstractTask
from AFAAS.lib.sdk.logger import AFAASLogger
LOG = AFAASLogger(name=__name__)

class TaskStack(AFAASModel):
    parent_task: AbstractBaseTask = Field(..., exclude=True)
    _task_ids: list[str] = Field(default=[])

    def __init__(self, **data: Any):
        super().__init__(**data)
        if not '_task_ids' in data.keys():
            self._task_ids = []


    def dict(self, *args, **kwargs) -> list[str]:
        return self._task_ids

    def json(self, *args, **kwargs):
        return json.dumps(self.dict())

    def __len__(self):
        return len(self._task_ids)

    def __iter__(self):
        LOG.error('Iterating over TaskStack')
        return iter(self._task_ids)
    
    @classmethod
    def __get_validators__(cls) -> Generator:
        LOG.trace(f"{cls.__name__}.__get_validators__()")
        yield cls.validate

    @classmethod
    def validate(cls, v: Any) -> "TaskStack":
        LOG.trace(f"{cls.__name__}.validate()")
        if isinstance(v, cls):
            return v
        elif isinstance(v, dict):
            # Assuming the dictionary contains the necessary data to create a TaskStack
            return cls(**v)  # Adjust this line as needed based on how TaskStack is initialized
        else:
            raise TypeError(f"Expected TaskStack or dict, received {type(v)}")


    def add(self, task: AbstractTask):
        """
        Add a task. Can also mark it as ready.
        """
        LOG.debug(f"Adding task {LOG.italic(task.debug_formated_str())})in a stack of task {LOG.italic(self.parent_task.debug_formated_str())}")
        LOG.trace(self._task_ids)

        self._task_ids.append(task.task_id)
        parent_is_plan : bool = isinstance(self.parent_task, AbstractPlan)
        if parent_is_plan:
            plan: AbstractPlan = self.parent_task
        else:
            plan: AbstractPlan = self.parent_task.agent.plan
            plan._register_task_as_modified(task_id=self.parent_task.task_id)

        if(self.parent_task.subtasks == self) : 
            # FIXME: Evaluate what is the best way to evaluate predecessors
            LOG.info(f"Added task ``{LOG.italic(task.task_goal)}`` as subtask of task ``{LOG.italic(self.parent_task.task_goal)}``")
            # LOG.trace((f"As is subtask do not inherit from parent predecessors, 3 options are considered :\n"
            #             + f"- Always add all predecessors of parent task to subtask predecessors\n"
            #             + f"- Smartly/Dynamicaly add all predecessors of parent task to subtask predecessors\n"
            #             + f"- Consider parent predecessor when evaluatin `Task.is_ready()`\n"))
            from .meta import TaskStatusList
            if( not parent_is_plan 
               and self.parent_task.state != TaskStatusList.READY ) :
                LOG.warning(f"Added subtask should only be added if parent_task is READY. Current state of {self.parent_task.debug_formated_str()} is {self.parent_task.state}")
            

    def get_task(self, task_id) -> AbstractBaseTask:
        """
        Get a specific task.
        """
        return self.parent_task.agent.plan.get_task(task_id)

    def get_all_tasks_from_stack(self) -> list[AbstractTask]:
        """
        Get all tasks. If only_ready is True, return only ready tasks.
        """
        return [
            self.parent_task.agent.plan.get_task(task_id) for task_id in self._task_ids
        ]
    

    def get_all_task_ids_from_stack(self) -> list[AbstractTask]:
        """
        Get all tasks. If only_ready is True, return only ready tasks.
        """
        return [
            task_id for task_id in self._task_ids
        ]

    def get_ready_tasks_from_stack(self) -> list[AbstractTask]:
        """
        Get all ready tasks.
        """
        ready_task_ids_set = set(self.parent_task.agent.plan.get_ready_tasks())

        common_task_ids = ready_task_ids_set.intersection(self._task_ids)

        return [
            self.parent_task.agent.plan.get_task(task_id) for task_id in common_task_ids
        ]
    

    def get_done_tasks_from_stack(self) -> list[AbstractTask]:
        """
        Get all ready tasks.
        """
        done_task_ids_set = set(self.parent_task.agent.plan.get_done_tasks())

        common_task_ids = done_task_ids_set.intersection(self._task_ids)

        return [
            self.parent_task.agent.plan.get_task(task_id) for task_id in common_task_ids
        ]

    def get_active_tasks_from_stack(self) -> list[AbstractTask]:
        """
        Get all active tasks.
        """
        active_task_ids_set = set(self.parent_task.agent.plan.get_active_tasks())

        common_task_ids = active_task_ids_set.intersection(self._task_ids)

        return [
            self.parent_task.agent.plan.get_task(task_id) for task_id in common_task_ids
        ]
    
    def __repr__(self):
        return f"Stack Type : {self.description}\n" + super().__repr__()
    
    def __str__(self):
        return self._task_ids.__str__()


# Additional methods can be added as needed
TaskStack.update_forward_refs()
