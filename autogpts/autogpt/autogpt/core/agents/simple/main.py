from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Awaitable, Callable, List, Tuple

from autogpt.core.tools import ToolResult, SimpleToolRegistry, TOOL_CATEGORIES
from autogpt.core.agents.base.main import BaseAgent
from autogpt.core.agents.simple.loop import SimpleLoop
from autogpt.core.agents.simple.models import (
    SimpleAgentConfiguration,
    SimpleAgentSettings,
    SimpleAgentSystems,
    SimpleAgentSystemSettings,
)
from autogpt.core.configuration import Configurable
from autogpt.core.memory.base import Memory
from autogpt.core.planning import SimplePlanner
from autogpt.core.planning.models.tasks import TaskStatusList
from autogpt.core.planning.models.plan import Plan, Task
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.resource.model_providers import OpenAIProvider
from autogpt.core.workspace.simple import SimpleWorkspace

if TYPE_CHECKING:
    from autogpt.core.agents.base.loop import BaseLoopHook

from autogpt.core.agents.base.loop import BaseLoopHook


class SimpleAgent(BaseAgent):
    ################################################################################
    ##################### REFERENCE SETTINGS FOR FACTORY ###########################
    ################################################################################

    CLASS_SYSTEM_SETINGS = SimpleAgentSystemSettings
    CLASS_CONFIGURATION = SimpleAgentConfiguration
    CLASS_SETTINGS = SimpleAgentSettings
    CLASS_SYSTEMS = SimpleAgentSystems

    default_settings = SimpleAgentSystemSettings(
        name="simple_agent",
        description="A simple agent.",
        configuration=SimpleAgentConfiguration(
            agent_name="Entrepreneur-GPT",
            agent_role=(
                "An AI designed to autonomously develop and run businesses with "
                "the sole goal of increasing your net worth."
            ),
            agent_goals=[
                "Increase net worth",
                "Grow Twitter Account",
                "Develop and manage multiple businesses autonomously",
            ],
            agent_goal_sentence="""Increase net worth
                and Grow Twitter Account
                and Develop and manage multiple businesses autonomously""",
            cycle_count=0,
            max_task_cycle_count=3,
            creation_time="",
            systems=SimpleAgentSystems(
                tool_registry=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.tools.SimpleToolRegistry",
                ),
                memory=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.memory.base.Memory",
                ),
                openai_provider=PluginLocation(
                    storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
                    storage_route="autogpt.core.resource.model_providers.OpenAIProvider",
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
        settings: SimpleAgentSystemSettings,
        logger: logging.Logger,
        tool_registry: SimpleToolRegistry,
        memory: Memory,
        openai_provider: OpenAIProvider,
        workspace: SimpleWorkspace,
        planning: SimplePlanner,
        user_id: uuid.UUID,
        agent_id: uuid.UUID = None,
    ):
        super().__init__(
            settings=settings,
            logger=logger,
            memory=memory,
            workspace=workspace,
            user_id=user_id,
            agent_id=agent_id,
        )

        # These are specific
        self._openai_provider = openai_provider
        self._planning = planning
        self._tool_registry = SimpleToolRegistry.with_tool_modules(
            modules = TOOL_CATEGORIES,
            agent=self,
            logger=self._logger,
            memory=memory,
            workspace=workspace,
            model_providers=openai_provider
            )

        self.plan: Plan = None
        self._loop = SimpleLoop(agent=self)
        self.prompt_settings = self.load_prompt_settings()

        # NOTE : MOVE ADD HOOK TO BaseLoop

        self.add_hook(
            hook=BaseLoopHook(
                name="begin_run",
                function=test_hook,
                kwargs=["foo_bar"],
                expected_return=True,
                callback_function=None,
            ),
            uuid=uuid.uuid4(),
        )
        # self.add_hook(name: str, hook: BaseLoopHook, uuid: uuid.UUID)
        # self.add_hook(name: str, hook: BaseLoopHook, uuid: uuid.UUID)
        # self.add_hook(name: str, hook: BaseLoopHook, uuid: uuid.UUID)

    def loophooks(self) -> SimpleLoop.LoophooksDict:
        if not self._loop._loophooks:
            self._loop._loophooks = {}
        return self._loop._loophooks

    def loop(self) -> SimpleLoop:
        return self._loop

    def add_hook(self, hook: BaseLoopHook, uuid: uuid.UUID):
        super().add_hook(hook, uuid)

    ################################################################################
    ################################ LOOP MANAGEMENT################################
    ################################################################################

    async def start(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ):
        return_var = await super().start(
            user_input_handler=user_input_handler,
            user_message_handler=user_message_handler,
        )
        return return_var

    async def stop(
        self,
        user_input_handler: Callable[[str], Awaitable[str]],
        user_message_handler: Callable[[str], Awaitable[str]],
    ):
        return_var = await super().stop(
            agent=self,
            user_input_handler=user_input_handler,
            user_message_handler=user_message_handler,
        )
        return return_var

    ################################################################################
    ################################FACTORY SPECIFIC################################
    ################################################################################

    @classmethod
    def _create_agent_custom_treatment(
        cls, agent_settings: SimpleAgentSettings, logger: logging.Logger
    ) -> None:
        return cls._create_workspace(agent_settings=agent_settings, logger=logger)

    @classmethod
    def _create_workspace(
        cls,
        agent_settings: SimpleAgentSettings,
        logger: logging.Logger,
    ):
        from autogpt.core.workspace import SimpleWorkspace

        return SimpleWorkspace.create_workspace(
            user_id=agent_settings.user_id,
            agent_id=agent_settings.agent_id,
            settings=agent_settings,
            logger=logger,
        )

    @classmethod
    def _get_agent_from_settings(
        cls,
        agent_settings: SimpleAgentSettings,
        agent_args: list,
        logger: logging.Logger,
    ) -> Tuple[SimpleAgentSettings, list]:
        agent_args["openai_provider"] = cls._get_system_instance(
            "openai_provider",
            agent_settings,
            logger,
        )

        # TODO : Continue refactorization => move to loop ?
        from autogpt.core.agents.simple import strategies
        from autogpt.core.agents.simple.strategies import (
            Strategies,
            StrategiesConfiguration,
        )

        # strategies_config = SimplePromptStrategiesConfiguration(
        #         name_and_goals=strategies.NameAndGoals.default_configuration,
        #         initial_plan=strategies.InitialPlan.default_configuration,
        #         next_ability=strategies.NextTool.default_configuration,)
        # #agent_settings.planning.configuration.prompt_strategies = strategies_config

        # #
        # # Dynamicaly load all strategies
        # # To do so Intanciate all class that inherit from PromptStrategy in package Strategy
        # #
        # simple_strategies = {}
        # import inspect
        # from  autogpt.core.planning.base import PromptStrategy
        # for strategy_name, strategy_config in strategies_config.__dict__.items():
        #     strategy_module = getattr(strategies, strategy_name)
        #     # Filter classes that are subclasses of PromptStrategy and are defined within that module
        #     strategy_classes = [member for name, member in inspect.getmembers(strategy_module)
        #                         if inspect.isclass(member) and
        #                         issubclass(member, PromptStrategy) and
        #                         member.__module__ == strategy_module.__name__]
        #     if not strategy_classes:
        #         raise ValueError(f"No valid class found in module {strategy_name}")
        #     strategy_instance = strategy_classes[0](**strategy_config.dict())

        #     simple_strategies[strategy_name] = strategy_instance

        simple_strategies = Strategies.get_strategies(logger=logger)
        # NOTE : Can't be moved to super() because require agent_args["openai_provider"]
        agent_args["planning"] = cls._get_system_instance(
            "planning",
            agent_settings,
            logger,
            model_providers={"openai": agent_args["openai_provider"]},
            strategies=simple_strategies,
        )

        # NOTE : Can't be moved to super() because require agent_args["openai_provider"]
        agent_args["tool_registry"] = cls._get_system_instance(
            "tool_registry",
            agent_settings,
            logger,
            workspace=agent_args["workspace"],
            memory=agent_args["memory"],
            model_providers={"openai": agent_args["openai_provider"]},
        )

        # agent = cls(**agent_args)

        # items = agent_settings.dict().items()
        # for key, value in items:
        #     if key not in agent_settings.__class__.Config.default_exclude:
        #         setattr(agent, key, value)

        return agent_settings, agent_args

    @classmethod
    async def determine_agent_name_and_goals(
        cls,
        user_objective: str,
        agent_settings: SimpleAgentSettings,
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

        return model_response.content

    def __repr__(self):
        return "SimpleAgent()"

    @classmethod
    def load_prompt_settings(cls):
        return super().load_prompt_settings(erase = False, file_path= __file__)


def test_hook(**kwargs):
    test = "foo_bar"
    for key, value in kwargs.items():
        print(f"{key}: {value}")
