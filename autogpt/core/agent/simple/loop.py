import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from autogpt.core.ability import (
    AbilityRegistrySettings,
    AbilityResult,
    SimpleAbilityRegistry,
)
from autogpt.core.agent.base import Agent
from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.memory import MemorySettings, SimpleMemory
from autogpt.core.planning import PlannerSettings, SimplePlanner, Task, TaskStatus
from autogpt.core.plugin.simple import (
    PluginLocation,
    PluginStorageFormat,
    SimplePluginService,
)
from autogpt.core.resource.model_providers import OpenAIProvider, OpenAISettings
from autogpt.core.workspace.simple import SimpleWorkspace, WorkspaceSettings
from autogpt.core.agent.simple.models import AgentConfiguration, AgentSettings, AgentSystems, AgentSystemSettings


class SimpleLoop():

    def __init__(
        self,
        settings: AgentSystemSettings,
        logger: logging.Logger,
        ability_registry: SimpleAbilityRegistry,
        memory: SimpleMemory,
        openai_provider: OpenAIProvider,
        planning: SimplePlanner,
        workspace: SimpleWorkspace,
    ):
        self._configuration = settings.configuration
        self._logger = logger
        self._ability_registry = ability_registry
        self._memory = memory
        # FIXME: Need some work to make this work as a dict of providers
        #  Getting the construction of the config to work is a bit tricky
        self._openai_provider = openai_provider
        self._planning = planning
        self._workspace = workspace
        self._task_queue = []
        self._completed_tasks = []
        self._current_task = None
        self._next_ability = None


    async def build_initial_plan(self) -> dict:
        plan = await self._planning.make_initial_plan(
            agent_name=self._configuration.name,
            agent_role=self._configuration.role,
            agent_goals=self._configuration.goals,
            abilities=self._ability_registry.list_abilities(),
        )
        tasks = [Task.parse_obj(task) for task in plan.content["task_list"]]

        # TODO: Should probably do a step to evaluate the quality of the generated tasks,
        #  and ensure that they have actionable ready and acceptance criteria

        self._task_queue.extend(tasks)
        self._task_queue.sort(key=lambda t: t.priority, reverse=True)
        self._task_queue[-1].context.status = TaskStatus.READY
        return plan.content

    async def determine_next_ability(self, *args, **kwargs):
        if not self._task_queue:
            return {"response": "I don't have any tasks to work on right now."}

        self._configuration.cycle_count += 1
        task = self._task_queue.pop()
        self._logger.info(f"Working on task: {task}")

        task = await self._evaluate_task_and_add_context(task)
        next_ability = await self._choose_next_ability(
            task,
            self._ability_registry.dump_abilities(),
        )
        self._current_task = task
        self._next_ability = next_ability.content
        return self._current_task, self._next_ability

    async def execute_next_ability(self, user_input: str, *args, **kwargs):
        if user_input == "y":
            ability = self._ability_registry.get_ability(
                self._next_ability["next_ability"]
            )
            ability_response = await ability(**self._next_ability["ability_arguments"])
            await self._update_tasks_and_memory(ability_response)
            if self._current_task.context.status == TaskStatus.DONE:
                self._completed_tasks.append(self._current_task)
            else:
                self._task_queue.append(self._current_task)
            self._current_task = None
            self._next_ability = None

            return ability_response.dict()
        else:
            raise NotImplementedError

    async def _evaluate_task_and_add_context(self, task: Task) -> Task:
        """Evaluate the task and add context to it."""
        if task.context.status == TaskStatus.IN_PROGRESS:
            # Nothing to do here
            return task
        else:
            self._logger.debug(f"Evaluating task {task} and adding relevant context.")
            # TODO: Look up relevant memories (need working memory system)
            # TODO: Evaluate whether there is enough information to start the task (language model call).
            task.context.enough_info = True
            task.context.status = TaskStatus.IN_PROGRESS
            return task

    async def _choose_next_ability(self, task: Task, ability_schema: list[dict]):
        """Choose the next ability to use for the task."""
        self._logger.debug(f"Choosing next ability for task {task}.")
        if task.context.cycle_count > self._configuration.max_task_cycle_count:
            # Don't hit the LLM, just set the next action as "breakdown_task" with an appropriate reason
            raise NotImplementedError
        elif not task.context.enough_info:
            # Don't ask the LLM, just set the next action as "breakdown_task" with an appropriate reason
            raise NotImplementedError
        else:
            next_ability = await self._planning.determine_next_ability(
                task, ability_schema
            )
            return next_ability

    async def _update_tasks_and_memory(self, ability_result: AbilityResult):
        self._current_task.context.cycle_count += 1
        self._current_task.context.prior_actions.append(ability_result)
        # TODO: Summarize new knowledge
        # TODO: store knowledge and summaries in memory and in relevant tasks
        # TODO: evaluate whether the task is complete

    def __repr__(self):
        return "SimpleLoop()"


    async def run(self) :

        from autogpt.core.runner.client_lib.parser import (
            parse_ability_result,
            parse_agent_plan,
            parse_next_ability,
        )
        import click
        
        plan = await self.build_initial_plan()
        print(parse_agent_plan(plan))


        while True:
            current_task, next_ability = await self.determine_next_ability(plan)
            print(parse_next_ability(current_task, next_ability))
            user_input = click.prompt(
                "Should the agent proceed with this ability?",
                default="y",
            )
            ability_result = await self.execute_next_ability(user_input)
            print(parse_ability_result(ability_result))


def _prune_empty_dicts(d: dict) -> dict:
    """
    Prune branches from a nested dictionary if the branch only contains empty dictionaries at the leaves.

    Args:
        d: The dictionary to prune.

    Returns:
        The pruned dictionary.
    """
    pruned = {}
    for key, value in d.items():
        if isinstance(value, dict):
            pruned_value = _prune_empty_dicts(value)
            if (
                pruned_value
            ):  # if the pruned dictionary is not empty, add it to the result
                pruned[key] = pruned_value
        else:
            pruned[key] = value
    return pruned
