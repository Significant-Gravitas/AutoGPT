from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine

from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.job import JobInterface
from AFAAS.interfaces.pipeline import Pipeline
from AFAAS.interfaces.task.task import AbstractTask
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.task.task import Task, TaskStatusList

LOG = AFAASLogger(name=__name__)


from AFAAS.interfaces.agent.loop import BaseLoop

if TYPE_CHECKING:
    from ..planner.main import PlannerAgent

from AFAAS.prompts.routing import (
    EvaluateSelectStrategy,
    RoutingStrategy,
    RoutingStrategyFunctionNames,
    SelectPlanningStrategy,
    SelectPlanningStrategyFunctionNames,
)


class RoutingLoop(BaseLoop):
    class LoophooksDict(BaseLoop.LoophooksDict):
        pass

    def __init__(self) -> None:
        super().__init__()
        self._active = False
        self._is_running = False

    async def run(
        self,
        agent: BaseAgent,
        hooks: LoophooksDict,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ) -> dict:
        planner_agent: BaseAgent = self._agent.parrent_agent
        pipeline_task: AbstractTask = self._agent.current_task
        task_history: str = pipeline_task.task_history
        task_context: str = pipeline_task.task_context
        task_goal: str = pipeline_task.long_description
        command_name: str
        command_args: dict
        assistant_reply_dict = None

        # Step for Routing
        routing_job = JobInterface(
            strategy=RoutingStrategy,
            strategy_kwargs={
                "note_to_agent_length": agent._settings.note_to_agent_length
            },
            response_post_process=RoutingLoop.routing_post_processing,
            autocorrection=False,
        )
        # Step for Planning
        planning_job = JobInterface(
            strategy=SelectPlanningStrategy,
            strategy_kwargs={},  # Add required kwargs
            response_post_process=self.generate_new_tasks,
            autocorrection=False,
        )
        pipeline = Pipeline(task=pipeline_task, agent=self._agent)  # Send Routing Agent
        pipeline.add_job(job=routing_job)
        pipeline.add_job(job=planning_job)
        return await pipeline.execute()

    @staticmethod
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
            task_data["agent_id"] = pipeline._agent.agent_id
            task_data["plan_id"] = pipeline._agent.parrent_agent.plan.plan_id
            task_data["agent"] = pipeline_task.agent
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

    @staticmethod
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
            evaluate_context_job = JobInterface(
                strategy=EvaluateSelectStrategy,
                strategy_kwargs={},  # Add required kwargs
                response_post_process=Pipeline.default_post_processing,
                autocorrection=True,
            )
            pipeline.jobs.append(evaluate_context_job)

    def __repr__(self):
        """Return a string representation of the RoutingLoop.

        Returns:
            str: A string representation of the RoutingLoop.
        """
        return "RoutingLoop()"
