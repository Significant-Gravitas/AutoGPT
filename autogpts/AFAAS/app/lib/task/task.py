from __future__ import annotations

import enum
import importlib
import pkgutil
import random
import string
import uuid
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, Field

# from autogpts.autogpt.autogpt.core.configuration import AFAASModel
# from autogpts.autogpt.autogpt.core.tools.schema import ToolResult

# if TYPE_CHECKING:
#     from autogpts.autogpt.autogpt.core.agents import BaseAgent

    #from .plan import Plan
from .base import BaseTask, TaskStatusList




class Task(BaseTask):
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

    ###
    ### GENERAL properties
    ###
    task_id: str = Field(
        default_factory=lambda: Task.generate_uuid()
    )  
    task_parent: Optional[Task] = []
    task_parent_id: Optional[str]  = []
    task_predecessors: Optional[list[Task]]  = []  
    task_predecessors_id: Optional[list[str]]  = []
    task_successors: Optional[list[Task]]  = []   
    task_successors_id: Optional[list[str]] = []
    
    state: Optional[TaskStatusList] = Field(default=TaskStatusList.BACKLOG.value)

    task_text_output : Optional[str] 
    """ Placeholder : The agent summary of his own doing while performing the task"""

    ###
    ### Task Management properties
    ###
    task_history: Optional[list[dict]]

    command: Optional[str] = Field(default_factory=lambda: Task.default_command())
    arguments: Optional[dict] = Field(default={})

    class Config(BaseTask.Config):
        pass
        
    @staticmethod
    def generate_uuid() :
        return "T" + str(uuid.uuid4())
    
 




# Need to resolve the circular dependency between Task and TaskContext once both models are defined.
Task.update_forward_refs()

