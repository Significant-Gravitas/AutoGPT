import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from forge.llm.providers import CompletionModelFunction, OpenAIProvider, OpenAISettings
from forge.models.config import Configurable, SystemConfiguration, SystemSettings
from pydantic import BaseModel

from autogpt.core.ability import (
    AbilityRegistrySettings,
    AbilityResult,
    SimpleAbilityRegistry,
)
from autogpt.core.agent.base import Agent
from autogpt.core.memory import MemorySettings, SimpleMemory
from autogpt.core.planning import PlannerSettings, SimplePlanner, Task, TaskStatus
from autogpt.core.plugin.simple import (
    PluginLocation,
    PluginStorageFormat,
    SimplePluginService,
)
from autogpt.core.workspace.simple import SimpleWorkspace, WorkspaceSettings


class AgentSystems(SystemConfiguration):
    ability_registry: PluginLocation
    memory: PluginLocation
    openai_provider: PluginLocation
    planning: PluginLocation
    workspace: PluginLocation


class AgentConfiguration(SystemConfiguration):
    cycle_count: int
    max_task_cycle_count: int
    creation_time: str
    name: str
    role: str
    goals: list[str]
    systems: AgentSystems


class AgentSystemSettings(SystemSettings):
    configuration: AgentConfiguration


class AgentSettings(BaseModel):
    agent: AgentSystemSettings
    ability_registry: AbilityRegistrySettings
    memory: MemorySettings
    openai_provider: OpenAISettings
    planning: PlannerSettings
    workspace: WorkspaceSettings

    def update_agent_name_and_goals(self, agent_goals: dict) -> None:
        self.agent.configuration.name = agent_goals["agent_name"]
        self.agent.configuration.role = agent_goals["agent_role"]
        self.agent.configuration.goals = agent_goals["agent_goals"]


class SimpleAgent(Agent, Configurable):
    default_settings = AgentSystemSettings(
        name="simple_agent",
        description="A simple agent.",
        configuration=AgentConfiguration(
            name="Entrepreneur-GPT",
            role=(
                "An AI designed to autonomously develop and run businesses with "
                "the sole goal of increasing your net worth."
            ),
            goals=[
                "Increase net worth",
                "Grow Twitter Account",
                "Develop and manage multiple businesses autonomously",
            ],
            cycle_count=0,
            max_task_cycle_count=3,
            creation_time="",
            systems=AgentSystems(
                ability_registry=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.ability.SimpleAbilityRegistry",
                ),
                memory=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.memory.SimpleMemory",
                ),
                openai_provider=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route=("forge.llm.model_providers.OpenAIProvider"),
                ),
                planning=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.planning.SimplePlanner",
                ),
                workspace=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.workspace.SimpleWorkspace",
                ),
            ),
        ),
    )

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

    @classmethod
    def from_workspace(
        cls,
        workspace_path: Path,
        logger: logging.Logger,
    ) -> "SimpleAgent":
        agent_settings = SimpleWorkspace.load_agent_settings(workspace_path)
        agent_args = {}

        agent_args["settings"] = agent_settings.agent
        agent_args["logger"] = logger
        agent_args["workspace"] = cls._get_system_instance(
            "workspace",
            agent_settings,
            logger,
        )
        agent_args["openai_provider"] = cls._get_system_instance(
            "openai_provider",
            agent_settings,
            logger,
        )
        agent_args["planning"] = cls._get_system_instance(
            "planning",
            agent_settings,
            logger,
            model_providers={"openai": agent_args["openai_provider"]},
        )
        agent_args["memory"] = cls._get_system_instance(
            "memory",
            agent_settings,
            logger,
            workspace=agent_args["workspace"],
        )

        agent_args["ability_registry"] = cls._get_system_instance(
            "ability_registry",
            agent_settings,
            logger,
            workspace=agent_args["workspace"],
            memory=agent_args["memory"],
            model_providers={"openai": agent_args["openai_provider"]},
        )

        return cls(**agent_args)

    async def build_initial_plan(self) -> dict:
        plan = await self._planning.make_initial_plan(
            agent_name=self._configuration.name,
            agent_role=self._configuration.role,
            agent_goals=self._configuration.goals,
            abilities=self._ability_registry.list_abilities(),
        )
        tasks = [Task.parse_obj(task) for task in plan.parsed_result["task_list"]]

        # TODO: Should probably do a step to evaluate the quality of the generated tasks
        #  and ensure that they have actionable ready and acceptance criteria

        self._task_queue.extend(tasks)
        self._task_queue.sort(key=lambda t: t.priority, reverse=True)
        self._task_queue[-1].context.status = TaskStatus.READY
        return plan.parsed_result

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
        self._next_ability = next_ability.parsed_result
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
            # TODO: Eval whether there is enough information to start the task (w/ LLM).
            task.context.enough_info = True
            task.context.status = TaskStatus.IN_PROGRESS
            return task

    async def _choose_next_ability(
        self,
        task: Task,
        ability_specs: list[CompletionModelFunction],
    ):
        """Choose the next ability to use for the task."""
        self._logger.debug(f"Choosing next ability for task {task}.")
        if task.context.cycle_count > self._configuration.max_task_cycle_count:
            # Don't hit the LLM, just set the next action as "breakdown_task"
            #  with an appropriate reason
            raise NotImplementedError
        elif not task.context.enough_info:
            # Don't ask the LLM, just set the next action as "breakdown_task"
            #  with an appropriate reason
            raise NotImplementedError
        else:
            next_ability = await self._planning.determine_next_ability(
                task, ability_specs
            )
            return next_ability

    async def _update_tasks_and_memory(self, ability_result: AbilityResult):
        self._current_task.context.cycle_count += 1
        self._current_task.context.prior_actions.append(ability_result)
        # TODO: Summarize new knowledge
        # TODO: store knowledge and summaries in memory and in relevant tasks
        # TODO: evaluate whether the task is complete

    def __repr__(self):
        return "SimpleAgent()"

    ################################################################
    # Factory interface for agent bootstrapping and initialization #
    ################################################################

    @classmethod
    def build_user_configuration(cls) -> dict[str, Any]:
        """Build the user's configuration."""
        configuration_dict = {
            "agent": cls.get_user_config(),
        }

        system_locations = configuration_dict["agent"]["configuration"]["systems"]
        for system_name, system_location in system_locations.items():
            system_class = SimplePluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.get_user_config()
        configuration_dict = _prune_empty_dicts(configuration_dict)
        return configuration_dict

    @classmethod
    def compile_settings(
        cls, logger: logging.Logger, user_configuration: dict
    ) -> AgentSettings:
        """Compile the user's configuration with the defaults."""
        logger.debug("Processing agent system configuration.")
        configuration_dict = {
            "agent": cls.build_agent_configuration(
                user_configuration.get("agent", {})
            ).dict(),
        }

        system_locations = configuration_dict["agent"]["configuration"]["systems"]

        # Build up default configuration
        for system_name, system_location in system_locations.items():
            logger.debug(f"Compiling configuration for system {system_name}")
            system_class = SimplePluginService.get_plugin(system_location)
            configuration_dict[system_name] = system_class.build_agent_configuration(
                user_configuration.get(system_name, {})
            ).dict()

        return AgentSettings.parse_obj(configuration_dict)

    @classmethod
    async def determine_agent_name_and_goals(
        cls,
        user_objective: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    ) -> dict:
        logger.debug("Loading OpenAI provider.")
        provider: OpenAIProvider = cls._get_system_instance(
            "openai_provider",
            agent_settings,
            logger=logger,
        )
        logger.debug("Loading agent planner.")
        agent_planner: SimplePlanner = cls._get_system_instance(
            "planning",
            agent_settings,
            logger=logger,
            model_providers={"openai": provider},
        )
        logger.debug("determining agent name and goals.")
        model_response = await agent_planner.decide_name_and_goals(
            user_objective,
        )

        return model_response.parsed_result

    @classmethod
    def provision_agent(
        cls,
        agent_settings: AgentSettings,
        logger: logging.Logger,
    ):
        agent_settings.agent.configuration.creation_time = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
        workspace: SimpleWorkspace = cls._get_system_instance(
            "workspace",
            agent_settings,
            logger=logger,
        )
        return workspace.setup_workspace(agent_settings, logger)

    @classmethod
    def _get_system_instance(
        cls,
        system_name: str,
        agent_settings: AgentSettings,
        logger: logging.Logger,
        *args,
        **kwargs,
    ):
        system_locations = agent_settings.agent.configuration.systems.dict()

        system_settings = getattr(agent_settings, system_name)
        system_class = SimplePluginService.get_plugin(system_locations[system_name])
        system_instance = system_class(
            system_settings,
            *args,
            logger=logger.getChild(system_name),
            **kwargs,
        )
        return system_instance


def _prune_empty_dicts(d: dict) -> dict:
    """
    Prune branches from a nested dictionary if the branch only contains empty
    dictionaries at the leaves.

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
