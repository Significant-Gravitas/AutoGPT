from __future__ import annotations

import uuid
from logging import Logger
from typing import TYPE_CHECKING, List, Optional, Union

from pydantic import BaseModel, Field

from .basetask import BaseTask
from .tasks import Task, TaskStatusList

logger = Logger(name=__name__)

from autogpts.autogpt.autogpt.core.configuration import AFAASModel
from autogpts.autogpt.autogpt.core.agents import BaseAgent

class Plan(BaseTask):
    """
    Represents a plan consisting of a list of tasks.
    """
    task_id: str = Field(
        default_factory=lambda: Plan.generate_uuid(),
        alias = "plan_id"
    ) 

    @staticmethod
    def generate_uuid() :
        return "PL" + str(uuid.uuid4())

    subtask: list[Task] = []

    def add_tasks(self, tasks=list[Task], position: int = None):
        if position is not None:
            for tasks in tasks:
                self.subtask.insert(tasks, position)
        else:
            for tasks in tasks:
                self.subtask.append(tasks)

    def dump(self, depth=0) -> dict:
        """
        Dump the plan and its tasks into a dictionary up to a specified depth.

        Args:
            depth (int): The depth up to which tasks should be included in the dictionary.
                         If depth is 0, only the plan itself is included.

        Examples:
            >>> plan = Plan([Task("Task 1"), Task("Task 2")])
            >>> plan.dump()
            {'tasks': [{'task_goal': 'Task 1'}, {'task_goal': 'Task 2'}]}

        Returns:
            dict: A dictionary representation of the plan and its tasks.
        Raises:
            ValueError: If depth is a negative integer.
        """
        if depth < 0:
            raise ValueError("Depth must be a non-negative integer.")

        # Initialize the return dictionary
        return_dict = self.dict()

        # Recursively process tasks up to the specified depth
        if depth > 0:
            return_dict["tasks"] = [task.dump(depth=depth - 1) for task in self.subtask]

        return return_dict

    def __getitem__(self, index: Union[int, str]):
        """
        Get a task from the plan by index or slice notation. This method is an alias for `get_task`.

        Args:
            index (Union[int, str]): The index or slice notation to retrieve a task.

        Returns:
            Task or List[Task]: The task or list of tasks specified by the index or slice.

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
        return self.get_task(index)

    def get_task(self, index: Union[int, str]):
        """
        Get a task from the plan by index or slice notation.

        Args:
            index (Union[int, str]): The index or slice notation to retrieve a task.

        Returns:
            Task or List[Task]: The task or list of tasks specified by the index or slice.

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
            if -len(self.subtask) <= index < len(self.subtask):
                return self.subtask[index]
            else:
                raise IndexError("Index out of range")
        elif isinstance(index, str) and index.startswith(":") and index[1:].isdigit():
            # Handle notation like ":-1"
            start = 0 if index[1:] == "" else int(index[1:])
            return self.subtask[start:]
        else:
            raise ValueError("Invalid index type")

    def find_task_path_with_id(self, task_id: str) -> Optional[List[Task]]:
        """
        Find the path to a task with the given task_id in the list of tasks.
        """
        logger.warning("Deprecated : Recommended function is Task.find_task_path()")
        for task in self.subtask:
            path = task.find_task_path_with_id(task_id)
            if path:
                return path
        return None

    ###
    ### FIXME : To test
    ###
    def remove_task(self, task_id: str):
        logger.error("""FUNCTION NOT WORKING :
                     1. We now manage multiple predecessor
                     2. Tasks should not be deleted but managed by state""")
        # 1. Set all task_predecessor_id to null if they reference the task to be removed
        def clear_predecessors(task: Task):
            if task.task_predecessor_id == task_id:
                task.task_predecessor_id = None
            for subtask in task.subtasks or []:
                clear_predecessors(subtask)

        # 2. Remove leaves with status "DONE" if ALL their siblings have this status
        def should_remove_siblings(
            task: Task, task_parent: Optional[Task] = None
        ) -> bool:
            # If it's a leaf and has a parent
            if not task.subtasks and task_parent:
                all_done = all(st.status == "DONE" for st in task_parent.subtasks)
                if all_done:
                    # Delete the Task objects
                    for st in task_parent.subtasks:
                        del st
                    task_parent.subtasks = None  # or []
                return all_done
            # elif task.subtasks:
            #     for st in task.subtasks:
            #         should_remove_siblings(st, task)
            return False

        for task in self.subtask:
            should_remove_siblings(task)
            clear_predecessors(task)

    def get_ready_leaf_tasks(self) -> List[Task]:
        """
        Get tasks that have status "READY", no subtasks, and no task_predecessor_id.

        Returns:
            List[Task]: A list of tasks meeting the specified criteria.
        """
        ready_tasks = []

        def check_task(task: Task):
            if (
                task.status == "READY"
                and not task.subtasks
                and not task.task_predecessor_id
            ):
                ready_tasks.append(task)

            # Check subtasks recursively
            for subtask in task.subtasks or []:
                check_task(subtask)

        # Start checking from the root tasks in the plan
        for task in self.subtask:
            check_task(task)

        return ready_tasks

    def get_first_ready_task(self) -> Optional[Task]:
        """
        Get the first task that has status "READY", no subtasks, and no task_predecessor_id.

        Returns:
            Task or None: The first task meeting the specified criteria or None if no such task is found.
        """

        def check_task(task: Task) -> Optional[Task]:
            if (
                task.status == "READY"
                and not task.subtasks
                and not task.task_predecessor_id
            ):
                return task

            # Check subtasks recursively
            for subtask in task.subtasks or []:
                found_task = check_task(subtask)
                if (
                    found_task
                ):  # If a task is found in the subtasks, return it immediately
                    return found_task
            return None

        # Start checking from the root tasks in the plan
        for task in self.subtask:
            found_task = check_task(task)
            if found_task:
                return found_task

        return None

    def generate_pitch(self, task=None):
        if task is None:
            task = self.get_first_ready_task()

        # Extract the task's siblings and path
        siblings = [
            sib
            for sib in self.subtask
            if sib.task_parent_id == task.task_parent_id and sib != task
        ]
        path_to_task = task.find_task_path()

        # Build the pitch
        pitch = """
        # INSTRUCTION
        Your goal is to find the best-suited command in order to achieve the following task: {task_description}

        # CONTEXT
        The high-level plan designed to achieve our goal is: 
        {high_level_plan}
        
        We are working on the task "{task_name}" that consists in: {task_command}. This task is located in:
        {path_structure}
        """.format(
            task_description=task.description,
            high_level_plan="\n".join(
                [
                    "{}: {}".format(t.task_goal, t.description)
                    for t in self.subtask
                    if not t.task_parent_id
                ]
            ),
            task_name=task.task_goal,
            task_command=task.command,  # assuming each task has a 'command' attribute
            path_structure="\n".join(["->".join(p.task_goal for p in path_to_task)]),
        )

        return pitch

    @staticmethod
    def parse_agent_plan(plan: dict) -> str:
        parsed_response = f"Agent Plan:\n"
        for i, task in enumerate(plan.subtask):
            task : Task
            parsed_response += f"{i+1}. {task.task_id} - {task.task_goal}\n"
            parsed_response += f"Description {task.description}\n"
            # parsed_response += f"Task type: {task.type}  "
            # parsed_response += f"Priority: {task.priority}\n"
            parsed_response += f"Predecessors:\n"
            for j, predecessor in enumerate(task.task_predecessors):
                 parsed_response += f"    {j+1}. {predecessor}\n"
            parsed_response += f"Successors:\n"
            for j, succesors in enumerate(task.task_succesors):
                 parsed_response += f"    {j+1}. {succesors}\n"
            parsed_response += f"Acceptance Criteria:\n"
            for j, criteria in enumerate(task.acceptance_criteria):
                parsed_response += f"    {j+1}. {criteria}\n"
            parsed_response += "\n"

        return parsed_response

    async def save():
        pass


# # 1. Find the first ready task
# first_ready_task = plan.get_first_ready_task()

# # 2. Retrieve the task and its siblings
# task_parent_id = first_ready_task.task_parent_id
# siblings = []
# if task_parent_id:
#     task_parent = plan.find_task(task_parent_id)
#     siblings = task_parent.subtasks

# # 3. Retrieve the list of tasks on the path
# path_to_task = plan.find_task_path(first_ready_task.task_id)
    @classmethod
    def create_plan(cls, agent : BaseAgent):
        memory = agent._memory
        plan_table = memory.get_table("plans")
        plan = cls(agent_id = agent.agent_id, task_goal = agent.agent_goal_sentence)
        plan._create_initial_tasks(agent = agent)

        plan_table.add(plan, id=plan.plan_id)
        return plan
        
    def _create_initial_tasks(self):
        try : 
            import autogpts.autogpt.autogpt.core.agents.routing
            initial_task = Task(
                task_parent_id=None,
                task_predecessor_id=None,
                responsible_agent_id=None,
                # task_goal="Define an agent approach to tackle a tasks",
                task_goal= self.task_goal,
                command="afaas_routing",
                arguments = {'note_to_agent_length' : 400},
                acceptance_criteria=[
                    "A plan has been made to achieve the specific task"
                ],
                state=TaskStatusList.READY,
            )
        except:
            initial_task = Task(
                task_parent_id=None,
                task_predecessor_id=None,
                responsible_agent_id=None,
                task_goal= self.task_goal,
                command="afaas_make_initial_plan",
                arguments={},
                acceptance_criteria=[
                    "Contextual information related to the task has been provided"
                ],
                state=TaskStatusList.READY,
            )
        # self._current_task = initial_task  # .task_id
        initial_task_list = [initial_task]

        ###
        ### Step 2 : Prepend usercontext
        ###
        try : 
            import autogpts.autogpt.autogpt.core.agents.usercontext
            refine_user_context_task = Task(
                # task_parent = self.plan() ,
                task_parent_id=None,
                task_predecessor_id=None,
                responsible_agent_id=None,
                task_goal="Refine a user requirements for better exploitation by Agents",
                command="afaas_refine_user_context",
                arguments={},
                state=TaskStatusList.READY,
            )
            initial_task_list = [refine_user_context_task] + initial_task_list
            # self._current_task = refine_user_context_task  # .task_id
        except:
            pass

        self.add_tasks(tasks= initial_task_list)

    # def _get_tasks_from_db(self):
    #     return Task.get_from_db(self.plan_id)
    # tasks: list[Task] =  Field(default_factory=_get_tasks_from_db)