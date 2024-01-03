from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from pydantic import Field, validator

from AFAAS.interfaces.adapters import AbstractChatModelResponse
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.task.base import AbstractBaseTask
from AFAAS.interfaces.task.meta import TaskStatusList
from AFAAS.interfaces.task.task import AbstractTask
from AFAAS.lib.sdk.logger import AFAASLogger, logging
from AFAAS.prompts.common.afaas_task_post_rag_update import (
    AfaasPostRagTaskUpdateStrategy,
)
from AFAAS.prompts.common.afaas_task_rag_step2_history import AfaasTaskRagStep2Strategy
from AFAAS.prompts.common.afaas_task_rag_step3_related import AfaasTaskRagStep3Strategy

LOG = AFAASLogger(name=__name__)

if TYPE_CHECKING:
    from AFAAS.interfaces.task.stack import TaskStack


class Task(AbstractTask):
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
    task_id: str = Field(default_factory=lambda: Task.generate_uuid())
    plan_id: str = Field()

    command: Optional[str] = Field(default_factory=lambda: Task.default_command())
    arguments: Optional[dict] = Field(default={})

    _task_parent_id: str = Field()

    @property
    def task_parent(self) -> AbstractBaseTask:
        LOG.trace(
            f"{self.debug_formated_str(True)} {self.__class__.__name__}.task_parent({self._task_parent_id})"
        )
        try:
            # Lazy load the parent task
            return self.agent.plan.get_task(self._task_parent_id)
        except KeyError:
            raise ValueError(f"No parent task found with ID {self._task_parent_id}")

    @task_parent.setter
    def task_parent(self, task: AbstractBaseTask):
        LOG.trace(f"{self.__class__.__name__}.task_parent.setter()")
        if not isinstance(task, Task):
            raise ValueError("task_parent must be an instance of Task")
        self._task_parent_id = task.task_id

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

    @validator("state", pre=True, always=True)
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
            LOG.error(f"Task {task_id} has state is None")
        return new_state

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

    task_text_output: Optional[str]
    """ Placeholder : The agent summary of his own doing while performing the task"""
    task_text_output_as_uml: Optional[str]
    """ Placeholder : The agent summary of his own doing while performing the task as a UML diagram"""

    class Config(AbstractBaseTask.Config):
        default_exclude = set(AbstractBaseTask.Config.default_exclude) | {
            # If commented create an infinite loop
            "task_parent",
            "task_predecessors",
            "task_successors",
        }

    def __init__(self, **data):
        LOG.trace(
            f"Entering {self.__class__.__name__}.__init__() : {data['task_goal']}"
        )
        super().__init__(**data)
        LOG.trace(
            f"Quitting {self.__class__.__name__}.__init__() : {data['task_goal']}"
        )

    @property
    def plan_id(self) -> str:
        return self.agent.plan.plan_id

    def is_ready(self) -> bool:
        if (
            len(self.task_predecessors.get_active_tasks_from_stack()) == 0
            and len(self.subtasks.get_active_tasks_from_stack()) == 0
            and (
                self.state == TaskStatusList.BACKLOG
                or self.state == TaskStatusList.READY
            )
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
    def get_task_from_db(cls, task_id: str, agent: BaseAgent) -> Task:
        memory = agent.memory
        task_table = memory.get_table("tasks")
        task = task_table.get(task_id=task_id, plan_id=agent.plan.plan_id)
        return cls(**task, agent=agent)

    @classmethod
    def create_in_db(cls, task: Task, agent: BaseAgent):
        memory = agent.memory
        task_table = memory.get_table("tasks")
        task_table.add(value=task, id=task.task_id)

    def save_in_db(self):
        from AFAAS.interfaces.db.db_table import AbstractTable

        memory = self.agent.memory
        task_table: AbstractTable = memory.get_table("tasks")
        task_table.update(
            value=self,
            task_id=self.task_id,
            plan_id=self.plan_id,
        )

    # endregion

    def get_task_path(self, task_to_root=False, include_self=False) -> list[Task]:
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
            and current_task.task_parent is not None
        ):
            path.append(current_task.task_parent)
            current_task = current_task.task_parent

        if not task_to_root:
            path.reverse()

        return path

    def get_formated_task_path(self) -> str:
        path_to_task = self.get_task_path()
        indented_structure = ""

        for i, task in enumerate(path_to_task):
            indented_structure += "  " * i + "-> " + task.debug_formated_str() + "\n"

        return indented_structure

    def get_sibblings(self) -> list[Task]:
        """
        Finds the sibblings of this task.
        """
        if self.task_parent is None:
            return []

        return self.task_parent.subtasks.get_all_tasks_from_stack()

    def __hash__(self):
        return hash(self.task_id)

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.task_id == other.task_id
        return False

    async def prepare_rag(
        self,
        predecessors: bool = True,
        successors: bool = False,
        history: int = 10,
        sibblings=True,
        path=True,
        similar_tasks: int = 100,
        avoid_sibbling_predecessors_redundancy: bool = False,
    ):
        plan_history: list[Task] = []
        if history > 0:
            plan_history = self.agent.plan.get_last_achieved_tasks(count=history)

        # 2a. Get the predecessors of the task
        task_predecessors: list[Task] = []
        if predecessors:
            task_predecessors = self.task_predecessors.get_all_tasks_from_stack()

        # 2b. Get the successors of the task
        task_successors: list[Task] = []
        if successors:
            task_successors = self.task_successors.get_all_tasks_from_stack()

        # 3. Remove predecessors from history to avoid redondancy
        history_and_predecessors = set(plan_history) | set(task_predecessors)

        # 4. Get the path to the task and remove it from history to avoid redondancy
        task_path: list[Task] = []
        if path:
            task_path = self.get_task_path()

        # 5. Get the sibblings of the task and remove them from history to avoid redondancy
        task_sibblings: list[Task] = []
        if sibblings:
            if avoid_sibbling_predecessors_redundancy:
                task_sibblings = (
                    set(self.get_sibblings()) - history_and_predecessors
                )  # - set([self])
            else:
                task_sibblings = set(self.get_sibblings())  # - set([self])

        # 6. Get the similar tasks , if at least n (similar_tasks) have been treated so we only look for similarity in complexe cases
        related_tasks: list[Task] = []
        if (
            len(self.agent.plan.get_all_done_tasks_ids()) > similar_tasks
        ):
            task_embedding = await self.agent.embedding_model.aembed_query(
                text=self.long_description
            )
            try:
                # FIXME: Create an adapter or open a issue on Langchain Github : https://github.com/langchain-ai/langchain to harmonize the AP
                related_tasks_documents = (
                    await self.agent.vectorstore.asimilarity_search_by_vector(
                        task_embedding,
                        k=similar_tasks,
                        include_metadata=True,
                        filter={"plan_id": {"$eq": self.plan_id}},
                    )
                )
            except Exception:
                related_tasks_documents = (
                    await self.agent.vectorstore.asimilarity_search_by_vector(
                        task_embedding,
                        k=10,
                        include_metadata=True,
                        filter=[{"metadata.plan_id": {"$eq": self.plan_id}}],
                    )
                )

            LOG.debug(related_tasks_documents)
            ## 1. Make Task Object
            for task in related_tasks_documents:
                related_tasks.append(self.agent.plan.get_task(task.metadata["task_id"]))

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
            self.rag_uml = rv.parsed_result[0]["command_args"].get("uml_diagrams" , '')

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
                self.rag_uml += rv.parsed_result[0]["command_args"]["uml_diagrams"]

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
