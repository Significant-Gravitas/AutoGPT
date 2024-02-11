from __future__ import annotations

import uuid
from abc import abstractmethod
from typing import Optional, TYPE_CHECKING

from pydantic import Field

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.task.base import AbstractBaseTask
from AFAAS.interfaces.task.meta import TaskStatusList
from AFAAS.interfaces.task.stack import TaskStack


class AbstractTask(AbstractBaseTask):
    @abstractmethod
    def __init__(self, **data):
        super().__init__(**data)

    # Properties
    @property
    @abstractmethod
    def task_id(self) -> str: ...

    # @property
    # @abstractmethod
    # def plan_id(self) -> str:
    #     ...

    @abstractmethod
    async def task_parent(self) -> AbstractBaseTask: ...

    @property
    @abstractmethod
    def task_predecessors(self) -> TaskStack: ...

    @property
    @abstractmethod
    def task_successors(self) -> TaskStack: ...

    state: Optional[TaskStatusList] = Field(default=TaskStatusList.BACKLOG)

    rag_history_txt: Optional[str] = None
    """description of previous step built by rag"""
    rag_related_task_txt: Optional[str] = None
    """description of related task obtained (most likely from a vector search)"""
    task_workflow: Optional[str] = None
    """Workfrom of the task (cf: class Workflow)"""
    rag_uml: Optional[list[str]] = None
    """Experimental : Attempt to gather UML represenation of previous steps"""

    task_text_output: Optional[str] = None

    task_text_output_as_uml: Optional[str] = None

    # Methods
    @staticmethod
    def generate_uuid():
        return "T" + str(uuid.uuid4())

    @abstractmethod
    async def is_ready(self) -> bool: ...

    @abstractmethod
    def add_predecessor(self, task: "AbstractTask"): ...

    @abstractmethod
    def add_successor(self, task: "AbstractTask"): ...

    @classmethod
    @abstractmethod
    async def get_task_from_db(
        cls, task_id: str, agent: BaseAgent
    ) -> "AbstractTask": ...

    @classmethod
    @abstractmethod
    async def db_create(cls, task: "AbstractTask", agent: BaseAgent): ...

    @abstractmethod
    async def db_save(self): ...

    @abstractmethod
    async def get_task_path(
        self, task_to_root: bool = False, include_self: bool = False
    ) -> list["AbstractTask"]: ...

    @abstractmethod
    async def get_formated_task_path(self) -> str: ...

    @abstractmethod
    async def get_siblings(self) -> list["AbstractTask"]: ...

    @abstractmethod
    def __hash__(self): ...

    @abstractmethod
    def __eq__(self, other): ...

    def str_with__status__(self):
        return f"{self.task_goal} (id : {self.task_id} / status : {self.state})"

    @abstractmethod
    async def task_preprossessing(
        self,
        predecessors: bool = True,
        successors: bool = False,
        history: int = 10,
        sibblings: bool = True,
        path: bool = True,
        similar_tasks: int = 0,
        avoid_redondancy: bool = False,
    ): ...

    @abstractmethod
    async def task_execute(self): ...

    @abstractmethod
    async def task_postprocessing(self): ...

    @abstractmethod
    async def retry(self): ...

    @abstractmethod
    async def memorize_output(self):
        ...

    memory : dict = {}
    """ Task is the unit of work in the AFAAS system. It passes from one system to another and memory is the way to keep track of the data that has been processed."""
