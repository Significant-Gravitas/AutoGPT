from __future__ import annotations

from abc import abstractmethod
from typing import Optional , TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.task.base import AbstractBaseTask
from AFAAS.interfaces.task.meta import TaskStatusList
from AFAAS.interfaces.task.task import AbstractTask


class AbstractPlan(AbstractBaseTask):
    # Properties
    @property
    @abstractmethod
    def task_id(self) -> str: ...

    @property
    @abstractmethod
    def plan_id(self) -> str: ...

    def parent_task(self) -> AbstractBaseTask:
        return None

    # Public Methods
    @staticmethod
    @abstractmethod
    def generate_uuid() -> str: ...

    @abstractmethod
    def get_all_tasks_ids(self) -> list[str]: ...

    @abstractmethod
    def get_all_done_tasks_ids(self) -> list[str]: ...

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[AbstractTask]: ...

    @abstractmethod
    def get_next_task(
        self, task: Optional[AbstractTask] = None
    ) -> Optional[AbstractTask]: ...

    @abstractmethod
    async def get_ready_tasks(
        self, task_ids_set: Optional[list[str]] = None
    ) -> list[AbstractTask]: ...

    @abstractmethod
    async def get_active_tasks(
        self, task_ids_set: Optional[list[str]] = None
    ) -> list[AbstractTask]: ...

    @abstractmethod
    async def get_first_ready_tasks(
        self, task_ids_set: Optional[list[str]] = None
    ) -> AbstractTask: ...

    @abstractmethod
    async def get_last_achieved_tasks(self, count: int = 1) -> list[AbstractTask]: ...

    @abstractmethod
    def unregister_loaded_task(self, task_id: str) -> AbstractTask: ...

    @classmethod
    @abstractmethod
    async def db_create(cls, agent: BaseAgent) -> "AbstractPlan": ...

    @abstractmethod
    def db_save(self): ...

    @classmethod
    @abstractmethod
    async def get_plan_from_db(
        cls, plan_id: str, agent: BaseAgent
    ) -> "AbstractPlan": ...

    # @abstractmethod
    # def generate_pitch(self, task: Optional[AbstractTask] = None) -> str:
    #     ...

    @abstractmethod
    def _registry_update_task_status_in_list(
        self, task_id: AbstractTask, status: TaskStatusList
    ): ...

    @abstractmethod
    def _register_task_as_modified(self, task_id: str) -> None: ...

    @abstractmethod
    def _register_new_task(self, task: AbstractTask) -> None: ...

    @abstractmethod
    def __len__(self) -> int:
        """The total number of tasks in the plan."""
        ...

    @abstractmethod
    def set_as_priority(self, task: AbstractTask) -> None:
        """Move an Task that is ready among the next tasks to be executed. For example : If a tasks fails and is to be executed again, it can be moved to the top of the list of tasks to be executed."""
        ...
