
from __future__ import annotations

from autogpts.autogpt.autogpt.core.configuration import AFAASModel
from autogpts.autogpt.autogpt.core.agents import AbstractAgent    
from .base import BaseTask

class TaskStack(AFAASModel):
    parent_task : BaseTask
    _task_ids : list[str] 

    def __init__(self):
        self._task_ids : list[str] = []  # List for all task IDs
    
    def __len__(self):
        return len(self._task_ids)
    
    def add(self, task: BaseTask):
        """
        Add a task. Can also mark it as ready.
        """
        self._task_ids.append(task.task_id)
        self.parent_task.add_task(task)

    def get_task(self, task_id)->BaseTask:
        """
        Get a specific task.
        """
        return self.parent_task.agent.plan.get_task(task_id)

    def get_all_tasks(self)-> list[BaseTask]:
        """
        Get all tasks. If only_ready is True, return only ready tasks.
        """
        return [self.parent_task.agent.plan.get_task(task_id) for task_id in self._task_ids]

    def get_ready_tasks(self)-> list[BaseTask]:
        """
        Get all ready tasks.
        """
        ready_task_ids_set = set(self.parent_task.agent.plan.get_ready_tasks())

        common_task_ids = ready_task_ids_set.intersection(self._task_ids)

        return [self.parent_task.agent.plan.get_task(task_id) for task_id in common_task_ids]


    # Additional methods can be added as needed
