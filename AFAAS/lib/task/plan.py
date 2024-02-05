from __future__ import annotations

import asyncio
import threading
import uuid
from typing import ClassVar

from pydantic import Field

# from AFAAS.core.db import
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.task.meta import TaskStatusList
from AFAAS.interfaces.task.plan import AbstractBaseTask, AbstractPlan
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.task.task import Task
from AFAAS.interfaces.workflow import FastTrackedWorkflow

LOG = AFAASLogger(name=__name__)


class Plan(AbstractPlan):
    class Config(AbstractPlan.Config):
        # This is a list of Field to Exclude during serialization
        default_exclude = set(AbstractPlan.Config.default_exclude) | {
            "initialized",
            "lock",
            "_loaded_tasks_dict",
        }
        json_encoders = AbstractPlan.Config.json_encoders | {}

    _instance: ClassVar[dict[Plan]] = {}
    lock: ClassVar[threading.Lock] = threading.Lock()
    initialized: ClassVar[bool] = False

    def __new__(cls, *args, **kwargs):
        if kwargs.get("agent", None) is not None:
            agent_id = kwargs.get("agent").agent_id
            with cls.lock:
                if agent_id in cls._instance:
                    return cls._instance[agent_id]

                instance = super(Plan, cls).__new__(cls)
                cls._instance[agent_id] = instance
                return instance
        else:
            return super(Plan, cls).__new__(cls)

    def __init__(self, *args, **kwargs):
        with self.lock:
            if self.initialized:
                return

            # Initialize the instance if needed
            super().__init__(**kwargs)
            Plan._instance[kwargs["agent"].agent_id] = self
            self.agent.plan: Plan = self
            self.initialized = True
            # self.reset_instance()
            self._modified_tasks_ids = []
            self._new_tasks_ids = []
            self._loaded_tasks_dict = {}
            self._all_task_ids = kwargs.get("_all_task_ids", [])
            self._ready_task_ids = kwargs.get("_ready_task_ids", [])
            self._done_task_ids = kwargs.get("_done_task_ids", [])
            # self.task_goal = kwargs.get('task_goal' , self.agent.agent_goal)

    @classmethod
    async def _load(cls, plan_id: str, agent: BaseAgent, **kwargs):
        instance = cls(plan_id=plan_id, agent=agent, **kwargs)

        # db = agent.db
        # task_table = await db.get_table("tasks")

        # from AFAAS.interfaces.db.table.nosql.base import AbstractTable
        # filter = AbstractTable.FilterDict(
        #     {
        #         "plan_id": [
        #             AbstractTable.FilterItem(
        #                 value=str(instance.plan_id),
        #                 operator=AbstractTable.Operators.EQUAL_TO,
        #             )
        #         ],
        #     }
        # )
        # all_tasks_from_db_dict = await task_table.list(filter=filter)

        # # Process tasks
        # all_tasks_ids = []
        # for task_as_dict in all_tasks_from_db_dict:
        #     #NOTE: Safegard as Pytest as create unexpected situation
        #     if task_as_dict['task_id'] in instance._all_task_ids :
        #         raise Exception(f"Error {task_as_dict['task_id']} already exist in {instance._all_task_ids}")

        #     task = Task(**task_as_dict, agent=agent)
        #     instance._register_task(task=task)

        #     if task.state == TaskStatusList.READY:
        #         instance._registry_update_task_status_in_list(
        #             task_id=task.task_id, status=TaskStatusList.READY
        #         )
        #     elif task.state == TaskStatusList.DONE:
        #         instance._registry_update_task_status_in_list(
        #             task_id=task.task_id, status=TaskStatusList.DONE
        #         )
        #     all_tasks_ids.append(
        #         task_as_dict['task_id']
        #         )

        return instance

    @classmethod
    async def get_plan_from_db(cls, plan_id: str, agent: BaseAgent) -> Plan:
        from AFAAS.core.db.table.nosql.agent import AgentsTable
        from AFAAS.interfaces.db.db import AbstractMemory

        plan_table: AgentsTable = await agent.db.get_table("plans")
        plan_dict = await plan_table.get(plan_id=plan_id, agent_id=agent.agent_id)

        if len(plan_dict) == 0:
            raise Exception(
                f"Plan {plan_id} not found in the database for agent {agent.agent_id}"
            )
        # TODO:v0.0.x : get_plan_from_db & load
        return await cls._load(**plan_dict, agent=agent)

    task_id: str = Field(default_factory=lambda: Plan.generate_uuid())

    @staticmethod
    def generate_uuid():
        return "PL" + str(uuid.uuid4())

    @property
    def plan_id(self):
        return self.task_id

    #############################################################################################
    #############################################################################################
    #############################################################################################
    # region Graph Traversal ( cf : region VSCode Feature)
    #############################################################################################
    #############################################################################################
    #############################################################################################

    async def get_task(self, task_id: str) -> AbstractBaseTask:
        """
        Get a task from the plan
        """
        task: Task = None
        if task_id == self.plan_id:
            return self
        if task_id in self._all_task_ids:
            task = self._loaded_tasks_dict.get(task_id)
            if task is None:
                task = await Task.get_task_from_db(task_id=task_id, agent=self.agent)
                self._loaded_tasks_dict[task_id] = task
            return task
        else:
            raise Exception(
                f"Task {task_id} not found in plan {self.plan_id} [{str(self._all_task_ids)}]"
            )

    async def _check_subtasks_or_successors(self, task: Task) -> Task:
        if task is None:
            return None

        rv: list[Task] = []
        subtask_ids = task.subtasks.get_all_task_ids_from_stack()
        successor_ids = task.task_successors.get_all_task_ids_from_stack()
        for subtask_id in subtask_ids + successor_ids:
            subtask = await self.get_task(subtask_id)
            if await subtask.is_ready():
                rv.append(subtask)

        if len(rv) > 0:
            return rv[0]
        return None

    async def _check_siblings(
        self, task: Task, _origin_task: Task, visited_tasks: set
    ) -> Task:
        siblings = await task.get_siblings()
        for sibling in siblings:
            if sibling.task_id not in visited_tasks:
                ready_sibling = await self.get_next_task(
                    sibling,
                    _origin_task,
                    check_outer=False,
                    visited_tasks=visited_tasks,
                )
                if ready_sibling:
                    return ready_sibling

        return None

    async def get_next_task(
        self,
        task: Task = None,
        _origin_task: Task = None,
        check_outer: bool = True,
        visited_tasks: set = None,
    ) -> Task:
        LOG.trace(
            f"get_next_task() : Current tasks that are ready are  Task : {self._ready_task_ids}"
        )
        LOG.debug(f"Getting next task from plan {self.debug_formated_str()}")

        if visited_tasks is None:
            visited_tasks = set()
        if task is not None:
            visited_tasks.add(task.task_id)

        if len(self._ready_task_ids) > 0:
            return await self.get_task(self._ready_task_ids[0])

        if task is None:
            LOG.notice(
                f"No Task has been provided, we will try to find the first ready task"
            )
            tasks = await self.find_ready_subbranch()
            if tasks:
                return tasks[0]

        if task is not None:
            if _origin_task == task:
                return None
            if _origin_task is not None and await task.is_ready():
                return await self.get_task(self._ready_task_ids[0])

            # Check for ready subtasks or successors
            ready_task = await self._check_subtasks_or_successors(task)
            if ready_task:
                return ready_task

            ready_sibling = await self._check_siblings(
                task=task, _origin_task=_origin_task, visited_tasks=visited_tasks
            )
            if ready_sibling:
                return ready_sibling

        # Recursively check parent task
        if check_outer:
            return await self._find_outer_next_task(
                task=task, _origin_task=_origin_task
            )

        return None

    async def _find_outer_next_task(
        self, task: Task, _origin_task: Task = None
    ) -> Task:
        LOG.warning(
            f"We are browsing the tree from leaf to root. This use case is not yet supported. This functionality is in alpha version."
        )

        if hasattr(task, "_task_parent_id") and task._task_parent_id is None:
            if not isinstance(task, Plan):
                LOG.error(f"Task {task.formated_str()} is not a plan and has no parent")
            return None
        elif not hasattr(task, "_task_parent_id"):
            return None

        parent_task = await task.task_parent() if task else None
        if parent_task is None or task == _origin_task:
            if parent_task is None:
                LOG.critical(
                    f"Task {task.debug_formated_str(status=True)} has no parent"
                )
            return None

        # Check ready tasks in the parent's subbranch
        ready_tasks = await parent_task.find_ready_subbranch(origin=task)
        if ready_tasks:
            return ready_tasks[0]

        # Recursively check the parent's siblings and ancestors
        return await self._find_outer_next_task(
            task=parent_task, _origin_task=_origin_task
        )

    async def get_first_ready_tasks(self, task_ids_set: list[str] = None) -> Task:
        """
        Get the first ready tasks from Plan._ready_task_ids
        """
        LOG.debug(f"Getting first ready tasks from plan {self.plan_id}")
        return await self.get_task(self._ready_task_ids[0])

    async def get_last_achieved_tasks(self, count: int = 1) -> list[Task]:
        """
        Get the n last achieved tasks from Plan._done_task_ids
        """
        LOG.debug(f"Getting last achieved tasks from plan {self.plan_id}")
        return [
            await self.get_task(task_id) for task_id in self._done_task_ids[-count:]
        ]

    def get_all_tasks_ids(self) -> list[str]:
        """
        Get all the tasks ids from the plan
        """
        return self._all_task_ids

    def get_ready_tasks_ids(self, task_ids_set: list[str] = None) -> list[Task]:
        """
        Get the all ready tasks from Plan._ready_task_ids
        """
        LOG.debug(f"Getting ready tasks from plan {self.plan_id}")

        ready_task_ids = set(self._ready_task_ids)
        if (task_ids_set is not None) and (len(task_ids_set) > 0):
            ready_task_ids = list(ready_task_ids.intersection(set(task_ids_set)))

        return ready_task_ids

    async def get_ready_tasks(self, task_ids_set: list[str] = None) -> list[Task]:
        return [
            await self.get_task(task_id=task_id)
            for task_id in self.get_ready_tasks_ids(task_ids_set=task_ids_set)
        ]

    def get_active_tasks_ids(self, task_ids_set: list[str] = None) -> list[Task]:
        """
        Active tasks are tasks not in Plan._done_task_ids but in Plan._all_task_ids
        """
        LOG.debug(f"Getting active tasks from plan {self.plan_id}")
        all_task_ids_set = set(self._all_task_ids)
        done_task_ids_set = set(self._done_task_ids)
        active_task_ids = all_task_ids_set - done_task_ids_set  # Set difference

        if (task_ids_set is not None) and (len(task_ids_set) > 0):
            active_task_ids = active_task_ids.intersection(set(task_ids_set))

        return active_task_ids

    async def get_active_tasks(self, task_ids_set: list[str] = None) -> list[Task]:
        return [
            await self.get_task(task_id)
            for task_id in self.get_active_tasks_ids(task_ids_set=task_ids_set)
        ]

    def get_all_done_tasks_ids(self) -> list[str]:
        """
        Get all the tasks ids from the plan
        """
        return self._done_task_ids

    async def get_all_done_tasks(self) -> list[Task]:
        """
        Get all the tasks ids from the plan
        """
        return [await self.get_task(task_id) for task_id in self._done_task_ids]

    #############################################################################################
    #############################################################################################
    #############################################################################################
    # endregion
    # region Task Registry ( cf : region VSCode Feature)
    #############################################################################################
    ########################_#####################################################################
    #############################################################################################
    def unregister_loaded_task(self, task_id: str) -> Task:
        """
        Remove a task from the Plan._loaded_tasks_dict and free db
        """
        return self._loaded_tasks_dict.pop(task_id)

    def _registry_update_task_status_in_list(
        self, status: TaskStatusList, task_id: str
    ):
        """
        Update the status of a task in the task list.

        Args:
            task_id (Task): The ID of the task to update.
            status (TaskStatusList): The new status of the task.

        Raises:
            ValueError: If the status is not a valid TaskStatusList value.

        """

        LOG.debug(f"Updating task {task_id} status to {status}")
        if status == TaskStatusList.READY:
            if task_id not in self._ready_task_ids:
                self._ready_task_ids.append(task_id)
        elif status == TaskStatusList.IN_PROGRESS_WITH_SUBTASKS:
            if task_id in self._ready_task_ids:
                self._ready_task_ids.remove(task_id)
        elif status == TaskStatusList.DONE:
            if task_id in self._ready_task_ids:
                self._ready_task_ids.remove(task_id)
            if task_id not in self._done_task_ids:
                self._done_task_ids.append(task_id)

            # if len(set(task.get_siblings_ids()) - set(task.agent.plan.get_all_done_tasks_ids())) == 0 :
            #     loop = asyncio.get_event_loop()
            #     parent = loop.run_until_complete(task.task_parent())
            #     if (not isinstance(parent, Plan)):
            #         #FIXME:  Make resumé of Parent
            #         parent.state = TaskStatusList.DONE

    def _register_new_task(self, task: Task):
        """
        Registers a new task.

        Args:
            task (Task): The task object to be registered.

        Returns:
            None
        """
        LOG.debug(
            (
                f"Start registering new task {task.task_goal}\n"
                + f"- Step 1 : Register the task in the plan\n"
                + f"- Step 2 : Register the task as new in the Lazy Saving Stack\n"
            )
        )
        self._register_task(task=task)
        self._loaded_tasks_dict[task.task_id] = task
        self._register_task_as_new(task_id=task.task_id)

    def _register_new_tasks(self, tasks: list[Task]):
        """
        Register a list of tasks in the plan
        """
        LOG.debug(f"Registering {len(tasks)} new tasks")
        for task in tasks:
            self._register_new_task(task=task)

    def _register_task(self, task: Task):
        """
        Register a Task in the index of Task (Plan._all_task_ids).
        If the Task is READY, the Task is Added to the index of ready tasks (Plan._ready_task_ids)
        """
        LOG.debug(f"Registering task {task.debug_formated_str()} in the index of Task")
        self._all_task_ids.append(task.task_id)
        # if task.state == TaskStatusList.READY:
        #     self._ready_task_ids.append(task.task_id)

    def _register_tasks(self, tasks: list[Task]):
        """
        Register a list of tasks in the plan
        """
        LOG.debug(f"Registering {len(tasks)} tasks")
        for task in tasks:
            self._register_task(task=task)

    def _register_task_as_modified(self, task_id: str):
        """
        Register a task as modified in the index of modified Task (Plan._modified_tasks_ids)
        """
        LOG.debug(f"Task {task_id} is registered as modified in the Lazy Loading List")
        if task_id not in self._modified_tasks_ids:
            self._modified_tasks_ids.append(task_id)

    def _register_task_as_new(self, task_id: str):
        """
        Register a task as new in the index of new Task (Plan._new_tasks_ids)
        """
        LOG.debug(f"Task {task_id} is registered as new in the Lazy Loading List")
        if task_id not in self._new_tasks_ids:
            self._new_tasks_ids.append(task_id)

    #############################################################################################
    #############################################################################################
    #############################################################################################
    # endregion
    # region Database access ( cf : region VSCode Feature)
    #############################################################################################
    #############################################################################################
    #############################################################################################

    def create_initial_tasks(self, status: TaskStatusList):
        LOG.debug(f"Creating initial task for plan {self.plan_id}")
        initial_task_list = []

        ###
        # Step 2 : Prepend usercontext
        ###
        # FIXME: DEACTIVATED FOR TEST PURPOSE
        if False:
            try:
                pass

                refine_user_context_task = Task(
                    agent=self.agent,
                    plan_id=self.plan_id,
                    _task_parent_id=self.plan_id,
                    # task_parent=self,
                    state=status,
                    responsible_agent_id=None,
                    task_goal="Refine a user requirements for better exploitation by Agents",
                    command="afaas_refine_user_context",
                    long_description="This tasks will consists in interacting with the user in order to get a more detailed, precise, complete and exploitable set of requirements",
                    acceptance_criteria=[
                        "The user has clearly and undoubtly stated his willingness to quit the process"
                    ],
                    arguments={},
                    task_workflow = FastTrackedWorkflow.name
                )
                initial_task_list += [refine_user_context_task] 
            except:
                pass

        initial_task = Task(
            agent=self.agent,
            plan_id=self.plan_id,
            # task_parent=self,
            state=status,
            _task_parent_id=self.plan_id,
            responsible_agent_id=None,
            task_goal=self.task_goal,
            command=Task.default_tool(),
            long_description="This is the initial task of the plan, no task has been performed yet and this tasks will consist in splitting the goal into subtasks",
            arguments={"note_to_agent_length": 400},
            acceptance_criteria=["A plan has been made to achieve the specific task"],
            task_workflow = FastTrackedWorkflow.name
        )
        # self._current_task = initial_task  # .task_id
        initial_task_list += [initial_task]

        self.add_tasks(tasks=initial_task_list)

    async def db_create(self):
        agent = self.agent
        if agent:
            db = agent.db
            self.agent_id = agent.agent_id
            plan_table = await db.get_table("plans")
            await plan_table.add(value=self, id=self.plan_id)

    async def db_save(self):
        ###
        # Step 1 : Lazy saving : Update Existing Tasks
        ###
        for task_id in self._modified_tasks_ids:
            # Safeguard to avoid saving new tasks
            if task_id not in self._new_tasks_ids:
                task: Task = self._loaded_tasks_dict.get(task_id)
                if task:
                    LOG.db_log(f"Saving task {task.task_goal}")
                    await task.db_save()  # Save the task

        ###
        # Step 2 : Lazy saving : Create New Tasks
        ###
        for task_id in self._new_tasks_ids:
            task: Task = self._loaded_tasks_dict.get(task_id)
            if task:
                LOG.db_log(f"Creating task {task.task_goal}")
                await task.db_create(task=task, agent=self.agent)  # Save the task

        # Reinitalize the lists
        self._modified_tasks_ids = []
        self._new_tasks_ids = []

        ###
        # Step 3 : Save the Plan
        ###
        agent = self.agent
        if agent:
            db = agent.db
            plan_table = await db.get_table("plans")
            await plan_table.update(
                plan_id=self.plan_id, agent_id=self.agent.agent_id, value=self
            )

    def __hash__(self):
        return hash(self.plan_id)

    def __len__(self):
        return len(self._all_task_ids)

    def set_as_priority(self, task: Task):
        if (
            task.state is not TaskStatusList.READY
            or task.task_id not in self._ready_task_ids
        ):
            raise Exception(
                f"Task {task.task_id} is not ready ; state : {task.state} ; ready tasks : {self._ready_task_ids}"
            )

        self._ready_task_ids.remove(task.task_id)
        self._ready_task_ids.insert(0, task.task_id)


Plan.update_forward_refs()
