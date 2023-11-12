from __future__ import annotations

import uuid
from logging import Logger
from typing import TYPE_CHECKING, List, Optional, Union

from pydantic import BaseModel, Field

from .base import BaseTask
from .task import Task, TaskStatusList

logger = Logger(name=__name__)

from autogpts.autogpt.autogpt.core.configuration import AFAASModel
from autogpts.autogpt.autogpt.core.agents import BaseAgent

class Plan(BaseTask):
    """
    Represents a plan consisting of a list of tasks.
    """
    # class Config(BaseTask):
    #     allow_population_by_field_name = True

    task_id: str = Field(
        default_factory=lambda: Plan.generate_uuid(),
        alias = "plan_id"
    ) 

    @staticmethod
    def generate_uuid() :
        return "PL" + str(uuid.uuid4())
    
    @property
    def plan_id(self):
        return self.task_id

    def generate_pitch(self, task=None):
        if task is None:
            task = self.get_first_ready_task()

        # Extract the task's siblings and path
        siblings = [
            sib
            for sib in self.subtasks
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
                    for t in self.subtasks
                    if not t.task_parent_id
                ]
            ),
            task_name=task.task_goal,
            task_command=task.command,  # assuming each task has a 'command' attribute
            path_structure="\n".join(["->".join(p.task_goal for p in path_to_task)]),
        )

        return pitch


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
        plan._create_initial_tasks()

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
        # FIXME: DEACTIVATED FOR TEST PURPOSE
        if False : 
            try : 
                import autogpts.autogpt.autogpt.core.agents.usercontext
                refine_user_context_task = Task(
                    # task_parent = self.plan() ,
                    task_parent_id=None,
                    task_predecessor_id=None,
                    responsible_agent_id=None,
                    task_goal="Refine a user requirements for better exploitation by Agents",
                    command="afaas_refine_user_context",
                    acceptance_criteria=[
                        "The user has clearly and undoubtly stated his willingness to quit the process"
                    ],
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