from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional

from pydantic import Field, validator

from AFAAS.interfaces.adapters import AbstractChatModelResponse
from AFAAS.interfaces.adapters.embeddings.wrapper import SearchFilter, DocumentType,  Filter, FilterType
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.task.base import AbstractBaseTask
from AFAAS.interfaces.task.meta import TaskStatusList
from AFAAS.interfaces.task.plan import AbstractPlan
from AFAAS.interfaces.task.task import AbstractTask
from AFAAS.lib.sdk.logger import AFAASLogger, logging
from AFAAS.prompts.common.afaas_task_post_rag_update import (
    AfaasPostRagTaskUpdateStrategy,
)

from AFAAS.interfaces.adapters.embeddings.wrapper import SearchFilter, DocumentType,  Filter, FilterType
from AFAAS.prompts.common.afaas_task_rag_step2_history import AfaasTaskRagStep2Strategy
from AFAAS.prompts.common.afaas_task_rag_step3_related import AfaasTaskRagStep3Strategy

LOG = AFAASLogger(name=__name__)

from AFAAS.interfaces.task.stack import TaskStack


class Task(AbstractTask):
    def __init__(self, **data):
        super().__init__(**data)
        LOG.trace(
            f"Quitting {self.__class__.__name__}.__init__() : {data['task_goal']}"
        )
        self._task_parent_loading = False
        # self._task_parent_loaded = asyncio.Event()
        self._task_parent_future = asyncio.Future()
        self.plan_id = self.agent.plan.plan_id

    def __setattr__(self, key, value):
        # Set attribute as normal
        super().__setattr__(key, value)
        # If the key is a model field, mark the instance as modified
        if key in self.__fields__:
            self.agent.plan._register_task_as_modified(task_id=self.task_id)

        if key == "state":
            self.agent.plan._registry_update_task_status_in_list(
                task_id=self.task_id, status=value
            )

    class Config(AbstractBaseTask.Config):
        default_exclude = set(AbstractBaseTask.Config.default_exclude) | {
            # If commented create an infinite loop
            "task_parent",
            "task_predecessors",
            "task_successors",
            "_task_parent_future",
            "_task_parent_loading",
            "_task_parent",
        }

    ###
    ### GENERAL properties
    ###
    task_id: str = Field(default_factory=lambda: Task.generate_uuid())

    plan_id: Optional[str] = Field()

    _task_parent_id: str = Field(...)
    _task_parent: Optional[Task] = None

    async def task_parent(self) -> AbstractBaseTask:
        # LOG.trace(
        #     f"{self.debug_formated_str(status = True)} {self.__class__.__name__}.task_parent()({self._task_parent_id})"
        # )
        try:
            # Lazy load the parent task
            return await self.agent.plan.get_task(self._task_parent_id)
        except KeyError:
            raise ValueError(f"No parent task found with ID {self._task_parent_id}")

    _task_predecessors: Optional[TaskStack]  # = Field(default=None)
    _task_successors: Optional[TaskStack]  # = Field(default=None)

    @property
    def task_predecessors(self) -> TaskStack:
        if self._task_predecessors is None:
            from AFAAS.interfaces.task.stack import TaskStack

            self._task_predecessors = TaskStack(
                parent_task=self, description="Predecessors"
            )
        return self._task_predecessors

    @property
    def task_successors(self) -> TaskStack:
        if self._task_successors is None:
            from AFAAS.interfaces.task.stack import TaskStack

            self._task_successors = TaskStack(
                parent_task=self, description="Successors"
            )
        return self._task_successors

    @validator("state", pre=True)
    def set_state(cls, new_state, values):
        task_id = values.get("task_id")
        if task_id and new_state:
            LOG.debug(f"Setting state of task {task_id} to {new_state}")
            # Assuming LOG and agent are defined and accessible
            agent: BaseAgent = values.get("agent")
            if agent:
                agent.plan._registry_update_task_status_in_list(
                    task_id=task_id, status=new_state
                )
        else:
            LOG.error(f"Task {task_id} state is None")
        return new_state

    command: Optional[str] = Field(default_factory=lambda: Task.default_tool())
    arguments: Optional[dict] = Field(default={})

    task_text_output: Optional[str]
    """ The agent summary of his own doing while performing the task"""
    task_text_output_as_uml: Optional[str]
    """ The agent summary of his own doing while performing the task as a UML diagram"""

    async def is_ready(self) -> bool:
        if (
            (self.state == TaskStatusList.BACKLOG or self.state == TaskStatusList.READY)
            and len(await self.task_predecessors.get_active_tasks_from_stack()) == 0
            and len(await self.subtasks.get_active_tasks_from_stack()) == 0
        ):
            # NOTE: This remove subtasks stored in the plan as they should not be required anymore
            for task_id in self.subtasks:
                self.agent.plan.unregister_loaded_task(task_id=task_id)

            # NOTE: Normaly the task should already be ready .
            # NOTE: Create two different states for ready & ready with active subtasks
            if self.state != TaskStatusList.READY:
                LOG.error(
                    f"Task {self.debug_formated_str()} is ready but not in the ready state. This should not happen."
                )
                self.state = TaskStatusList.READY

            return True

        return False

    async def close_task(self):
        self.state = TaskStatusList.DONE

        LOG.info(f"Terminating Task : {self.debug_formated_str()}")
        # TODO: MOVE to the validator state for robustness
        if (
            len(
                set(await self.get_siblings_ids(excluse_self=False))
                - set(self.agent.plan.get_all_done_tasks_ids())
            )
            == 0
        ):
            parent = await self.task_parent()
            if not isinstance(parent, AbstractPlan):
                # FIXME:  Make resumé of Parent
                parent.state = TaskStatusList.DONE

    def add_predecessor(self, task: Task):
        """
        Adds a predecessor to this task (also automatically adds this task as a successor to the given predecessor task).

        Args:
            task (Task): The task to be added as a predecessor.

        Warning:
            This method should not be used in conjunction with `add_successor` or `_add_predecessor` on the same task objects,
            as it can lead to a recursive loop.
        """
        self.task_predecessors.add(task)
        task._add_successor(self)

    def add_successor(self, task: Task):
        """
        Adds a successor to this task (also automatically adds this task as a successor to the given predecessor task).

        Args:
            task (Task): The task to be added as a successor.

        Warning:
            This method should not be used in conjunction with `add_predecessor` on the same task objects,
            as it can lead to a recursive loop.
        """
        self.task_successors.add(task)
        task._add_predecessor(self)

    def _add_successor(self, task: Task):
        """
        DO NOT USE : This method should only be used within `Task.add_predecessor()`
        """
        self.task_successors.add(task)

    def _add_predecessor(self, task: Task):
        """
        DO NOT USE : This method should only be used within `Task.add_successors()`
        """
        self.task_predecessors.add(task)

    #############################################################################################
    #############################################################################################
    #############################################################################################
    # region Task DB Management
    #############################################################################################
    #############################################################################################
    #############################################################################################

    @classmethod
    async def get_task_from_db(cls, task_id: str, agent: BaseAgent) -> Task:
        db = agent.db
        task_table = await db.get_table("tasks")
        task = await task_table.get(task_id=task_id, plan_id=agent.plan.plan_id)
        return cls(**task, agent=agent)

    @classmethod
    async def db_create(cls, task: Task, agent: BaseAgent):
        db = agent.db
        task_table = await db.get_table("tasks")
        await task_table.add(value=task, id=task.task_id)

    async def db_save(self):
        from AFAAS.interfaces.db.db_table import AbstractTable

        db = self.agent.db
        task_table: AbstractTable = await db.get_table("tasks")
        await task_table.update(
            value=self,
            task_id=self.task_id,
            plan_id=self.plan_id,
        )

    # endregion

    async def get_task_path(self, task_to_root=False, include_self=False) -> list[Task]:
        """
        Finds the path from the root to the task ( not including the task itself by default)
        If task_to_root is True, the path will be from the task to the root.
        If include_self is True, the task will be included in the path.
        """
        path: list[Task] = []
        if include_self:
            path.append(self)

        current_task: Task = self

        while (
            hasattr(current_task, "task_parent")
            and await current_task.task_parent() is not None
        ):
            path.append(await current_task.task_parent())
            current_task = await current_task.task_parent()

        if not task_to_root:
            path.reverse()

        return path

    async def get_formated_task_path(self) -> str:
        path_to_task = await self.get_task_path()
        indented_structure = ""

        for i, task in enumerate(path_to_task):
            indented_structure += "  " * i + "-> " + task.debug_formated_str() + "\n"

        return indented_structure

    async def get_siblings(self, excluse_self=True) -> list[Task]:
        """
        Finds the sibblings of this task.
        """
        parent_task = await self.task_parent()
        if parent_task is None:
            return []

        # Get all siblings including the task itself.
        all_siblings = await parent_task.subtasks.get_all_tasks_from_stack()
        if excluse_self:
            return [task for task in all_siblings if task.task_id != self.task_id]

        return all_siblings

    async def get_siblings_ids(self, excluse_self=True) -> list[Task]:
        parent_task = await self.task_parent()
        if parent_task is None:
            return []

        # Get all sibling IDs including the ID of the task itself.
        all_sibling_ids = parent_task.subtasks.get_all_task_ids_from_stack()
        if excluse_self:
            # Exclude the ID of the task itself from the list of sibling IDs.
            return [task_id for task_id in all_sibling_ids if task_id != self.task_id]

        return all_sibling_ids

    def __hash__(self):
        return hash(self.task_id)

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.task_id == other.task_id
        return False

    def __copy__(self):
        import copy
        from AFAAS.interfaces.task.base import AFAASModel
        cls = self.__class__
        clone = cls(**self.dict(), agent = self.agent)
        # for attr in self.__dict__:
        #     original_value = getattr(self, attr)

        #     if isinstance(original_value, (AbstractBaseTask, BaseAgent)):
        #         # Keep reference for AbstractBaseTask and BaseAgent types
        #         setattr(clone, attr, original_value)
        #     elif isinstance(original_value, TaskStack):
        #         continue
        #     elif isinstance(original_value, asyncio.Future):
        #         setattr(clone, attr, original_value)
        #     else:
        #         # Copy all other attributes
        #         setattr(clone, attr, copy.copy(original_value))
        clone.agent = self.agent
        clone._task_parent = self._task_parent

        return clone

    def __deepcopy__(self, memo) : 
        import copy
        LOG.warning(f"You should not use deepcopy on Task objects. Use Task.clone() instead")
        return copy.deepcopy(self)

    async def clone(self , with_predecessor = False) -> Task:
        import copy
        clone = copy.copy(self)


        import datetime
        clone.created_at = datetime.datetime.now()
        clone.task_id = Task.generate_uuid()

        clone.state = TaskStatusList.BACKLOG
        clone.task_text_output = None
        clone.task_text_output_as_uml = None
        clone._task_successors = []
        for successor in await self.task_successors.get_all_tasks_from_stack():
            successor.add_predecessor(clone)
        if with_predecessor :
            for predecessor in await self.task_predecessors.get_all_tasks_from_stack():
                predecessor.add_successor(clone)
        return clone

    async def retry(self) -> Task:
        """ Clone a task and adds it as its immediate successor"""
        LOG.warning("Task.retry() is an experimental method")
        clone = self.clone()
        self.add_successor(clone)
        return clone

    async def prepare_rag(
        self,
        predecessors: bool = True,
        successors: bool = False,
        history: int = 10,
        sibblings=True,
        path=True,
        nb_similar_tasks: int = 100,
        avoid_sibbling_predecessors_redundancy: bool = False,
    ):
        plan_history: list[Task] = []
        if history > 0:
            plan_history = await self.agent.plan.get_last_achieved_tasks(count=history)

        # 2a. Get the predecessors of the task
        task_predecessors: list[Task] = []
        if predecessors:
            task_predecessors = await self.task_predecessors.get_all_tasks_from_stack()

        # 2b. Get the successors of the task
        task_successors: list[Task] = []
        if successors:
            task_successors = await self.task_successors.get_all_tasks_from_stack()

        # 3. Remove predecessors from history to avoid redondancy
        history_and_predecessors = set(plan_history) | set(task_predecessors)

        # 4. Get the path to the task and remove it from history to avoid redondancy
        task_path: list[Task] = []
        if path:
            task_path = await self.get_task_path()

        # 5. Get the sibblings of the task and remove them from history to avoid redondancy
        task_sibblings: list[Task] = []
        if sibblings:
            # sibblings_tmp = await self.get_siblings()
            if avoid_sibbling_predecessors_redundancy:
                task_sibblings = (
                    set(await self.get_siblings()) - history_and_predecessors
                )  # - set([self])
            else:
                task_sibblings = set(await self.get_siblings())  # - set([self])

        # 6. Get the similar tasks , if at least n (similar_tasks) have been treated so we only look for similarity in complexe cases
        related_tasks: list[Task] = []
        if len(self.agent.plan.get_all_done_tasks_ids()) > nb_similar_tasks:
            task_embedding = await self.agent.embedding_model.aembed_query(
                text=self.long_description
            )
            # FIXME: Create an adapter or open a issue on Langchain Github : https://github.com/langchain-ai/langchain to harmonize the AP

            related_tasks_documents = await self.agent.vectorstores.get_related_documents(
                                        embedding =  task_embedding ,
                                        nb_results = 10 ,
                                        document_type = DocumentType.TASK,
                                        search_filters= SearchFilter(filters = {
                                            'agent_id' : Filter( 
                                                filter_type=FilterType.EQUAL,
                                                value=self.agent.agent_id,
                                            ) 
                                        }
                                        )
            )

            LOG.debug(related_tasks_documents)
            ## 1. Make Task Object
            for task in related_tasks_documents:
                related_tasks.append(
                    await self.agent.plan.get_task(task.metadata["task_id"])
                )

            ## 2. Make a set of related tasks and remove current tasks, sibblings, history and predecessors
            related_tasks = list(
                set(related_tasks) - task_sibblings - history_and_predecessors
            )

            if LOG.level <= logging.DEBUG:
                input("Press Enter to continue...")

        task_sibblings = list(task_sibblings)
        task_history = list(history_and_predecessors)
        task_history.sort(key=lambda task: task.modified_at)

        # TODO: Build it in a Pipeline for Autocorrection

        # NOTE: (Deactivated because Unnecessary ) RAG : 1. Summarize Path => No need as we keep it as is
        # self.rag_path_txt = task_path  + task_sibblings

        # RAG : 2. Summarize History
        self.rag_uml = []
        if len(task_history) > 0:
            rv: AbstractChatModelResponse = await self.agent.execute_strategy(
                strategy_name=AfaasTaskRagStep2Strategy.STRATEGY_NAME,
                task=self,
                task_path=task_path,
                task_history=task_history,
                task_followup=task_successors,
                task_sibblings=task_sibblings,
                related_tasks=related_tasks,
            )
            self.rag_history_txt = rv.parsed_result[0]["command_args"]["paragraph"]
            self.rag_uml = rv.parsed_result[0]["command_args"].get("uml_diagrams", "")

            # RAG : 3. Summarize Followup
            if len(task_successors) > 0 or len(related_tasks) > 0:
                rv: AbstractChatModelResponse = await self.agent.execute_strategy(
                    strategy_name=AfaasTaskRagStep3Strategy.STRATEGY_NAME,
                    task=self,
                    task_path=task_path,
                    task_history=task_history,
                    task_followup=task_successors,
                    task_sibblings=task_sibblings,
                    related_tasks=related_tasks,
                )
                self.rag_related_task_txt = rv.parsed_result[0]["command_args"][
                    "paragraph"
                ]
                self.rag_uml += rv.parsed_result[0]["command_args"].get(
                    "uml_diagrams", ""
                )

            # RAG : 4. Post-Rag task update
            rv: AbstractChatModelResponse = await self.agent.execute_strategy(
                strategy_name=AfaasPostRagTaskUpdateStrategy.STRATEGY_NAME,
                task=self,
                task_path=task_path,
            )
            self.long_description = rv.parsed_result[0]["command_args"][
                "long_description"
            ]
            # FIXME: ONY for ruting & planning ?
            self.task_workflow = rv.parsed_result[0]["command_args"]["task_workflow"]


# Need to resolve the circular dependency between Task and TaskContext once both models are defined.
Task.update_forward_refs()
