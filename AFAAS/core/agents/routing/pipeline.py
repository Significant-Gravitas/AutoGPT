from __future__ import annotations
from pydantic import Field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine, Type
from AFAAS.interfaces.job import JobInterface
from AFAAS.interfaces.prompts import AbstractPromptStrategy
from AFAAS.prompts.routing import (
    EvaluateSelectStrategy,
    RoutingStrategy,
    RoutingStrategyFunctionNames,
    SelectPlanningStrategy,
    SelectPlanningStrategyFunctionNames,
)

from AFAAS.lib.task.task import Task, TaskStatusList

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.core.agents.routing.main import RoutingAgent

from AFAAS.core.tools.tool_decorator import SAFE_MODE, tool
from AFAAS.interfaces.tools.tool import AFAASBaseTool
from AFAAS.prompts.routing import RoutingStrategyConfiguration


from AFAAS.interfaces.job import JobInterface
from AFAAS.interfaces.pipeline import Pipeline
from AFAAS.interfaces.task.task import AbstractTask
from AFAAS.lib.sdk.logger import AFAASLogger

def routing_post_processing(
        pipeline: Pipeline,
        command_name: str,
        command_args: dict,
        assistant_reply_dict: Any,
    ):
        Pipeline.default_post_processing(
            pipeline=pipeline,
            command_name=command_name,
            command_args=command_args,
            assistant_reply_dict=assistant_reply_dict,
        )
        if command_args["strategy"] == RoutingStrategyFunctionNames.EVALUATE_AND_SELECT:
            evaluate_context_job = EvaluateContextJob()
            pipeline.jobs.append(evaluate_context_job)

async def generate_new_tasks(
    pipeline: Pipeline,
    command_name: str,
    command_args: Any,
    assistant_reply_dict: Any,
):
    agent: BaseAgent = pipeline._agent
    pipeline_task: AbstractTask = pipeline._task
    pipeline_task.task_text_output = assistant_reply_dict
    llm_task_list = command_args["task_list"]

    if len(llm_task_list) == 1:
        command: str = "afaas_select_tool"
    else:
        command: str = Task.default_tool()

    tasks: dict[str, AbstractTask] = {}
    for task_data in llm_task_list:
        task_id: str = task_data["task_id"]

        # Set up basic task properties
        task_data["agent_id"] = agent.agent_id
        task_data["plan_id"] = agent.plan.plan_id
        task_data["agent"] = agent
        task_data["_task_parent_id"] = pipeline_task.task_id
        task_data["_task_parent"] = pipeline_task
        task_data["command"] = command

        # Store predecessors and then remove the field
        predecessors = task_data.pop("predecessors", [])

        # Remove task_id from task_data before creating the task
        del task_data["task_id"]

        # Create a new AbstractTask instance
        new_task = Task(**task_data)
        tasks[task_id] = new_task

        # Set up predecessors for the new task
        for predecessor_id in predecessors:
            if predecessor_id in tasks:
                # Existing task in the dictionary
                predecessor_task = tasks[predecessor_id]
            else:
                # Retrieve the task from planner_agent's plan
                predecessor_task = await agent.plan.get_task(predecessor_id)

            if predecessor_task is not None:
                new_task.add_predecessor(predecessor_task)

    # NOTE: In both case we add tasks to the parent tasks of the planning task
    # parent_task = await pipeline_task.task_parent()
    # parent_task.add_tasks(tasks=tasks.values())

    # NOTE: IF TASKS are added as subtasks then
    pipeline_task.add_tasks(tasks=tasks.values())
    pipeline_task.state = TaskStatusList.IN_PROGRESS_WITH_SUBTASKS

    return tasks


class PlanningJob(JobInterface):
    strategy : Type[AbstractPromptStrategy]= SelectPlanningStrategy
    strategy_kwargs = {}
    response_post_process : Callable = generate_new_tasks
    autocorrection = False


class RoutingJob(JobInterface):
    strategy : Type[AbstractPromptStrategy]= RoutingStrategy
    strategy_kwargs: dict
    response_post_process: Callable = routing_post_processing
    autocorrection = False


class EvaluateContextJob(JobInterface):
    strategy : Type[AbstractPromptStrategy]= EvaluateSelectStrategy
    strategy_kwargs = {}
    response_post_process : Callable = Pipeline.default_post_processing
    autocorrection = True

class RoutingPipeline(Pipeline) : 
    def __init__(self, task: AbstractTask, agent: BaseAgent, note_to_agent_length : int ) -> None:
        super().__init__(task=task, agent=agent)

        self.add_job(job=RoutingJob(strategy_kwargs = {'note_to_agent_length' : note_to_agent_length }) )
        self.add_job(job=PlanningJob())
