from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from pydantic import Field

from autogpts.AFAAS.app.lib.task.plan import Plan
from autogpts.autogpt.autogpt.core.memory.base import AbstractMemory
from autogpts.autogpt.autogpt.core.resource.model_providers import (
    OpenAIProvider, OpenAISettings)
from autogpts.autogpt.autogpt.core.tools import (TOOL_CATEGORIES,
                                                 SimpleToolRegistry)

from ..base import (BaseAgent, BaseLoopHook, PromptManager,
                    ToolExecutor)
from .loop import PlannerLoop
from .models import (PlannerAgentConfiguration,  # PlannerAgentSystemSettings,
                     PlannerAgentSystems)

if TYPE_CHECKING:
    from autogpts.autogpt.autogpt.core.workspace.simple import SimpleWorkspace


class PlannerAgent(BaseAgent):
    ################################################################################
    ##################### REFERENCE SETTINGS FOR FACTORY ###########################
    ################################################################################

    CLASS_CONFIGURATION = PlannerAgentConfiguration
    CLASS_SYSTEMS = PlannerAgentSystems  # PlannerAgentSystems() = cls.SystemSettings().configuration.systems

    class SystemSettings(BaseAgent.SystemSettings):
        name: str = "simple_agent"
        description: str = "A simple agent."
        configuration: PlannerAgentConfiguration = PlannerAgentConfiguration()

        # chat_model_provider: OpenAISettings = Field(default=OpenAISettings(), exclude=True)
        chat_model_provider: OpenAISettings = OpenAISettings()
        tool_registry: SimpleToolRegistry.SystemSettings = (
            SimpleToolRegistry.SystemSettings()
        )
        prompt_manager: PromptManager.SystemSettings = PromptManager.SystemSettings()

        agent_name: str = Field(default="New Agent")
        agent_role: Optional[str] = Field(default=None)
        agent_goals: Optional[list]
        agent_goal_sentence: Optional[str]

        class Config(BaseAgent.SystemSettings.Config):
            pass

        def json(self, *args, **kwargs):
            self.prepare_values_before_serialization()  # Call the custom treatment before .json()
            kwargs["exclude"] = self.Config.default_exclude
            return super().json(*args, **kwargs)

    def __init__(
        self,
        settings: PlannerAgent.SystemSettings,
        logger: logging.Logger,
        memory: AbstractMemory,
        chat_model_provider: OpenAIProvider,
        workspace: SimpleWorkspace,
        prompt_manager: PromptManager,
        user_id: uuid.UUID,
        agent_id: uuid.UUID = None,
        **kwargs,
    ):
        super().__init__(
            settings=settings,
            logger=logger,
            memory=memory,
            workspace=workspace,
            user_id=user_id,
            agent_id=agent_id,
        )
        self.agent_id = settings.agent_id
        self.user_id = settings.user_id
        self.agent_name = settings.agent_name
        self.agent_goals = settings.agent_goals
        self.agent_goal_sentence = settings.agent_goal_sentence

        #
        # Step 1 : Set the chat model provider
        #
        self._chat_model_provider = chat_model_provider
        # self._chat_model_provider.set_agent(agent=self)

        #
        # Step 2 : Load prompt_settings.yaml (configuration)
        #
        self.prompt_settings = self.load_prompt_settings()

        #
        # Step 3 : Set the chat model provider
        #
        self._prompt_manager = prompt_manager
        self._prompt_manager.set_agent(agent=self)

        #
        # Step 4 : Set the ToolRegistry
        #
        self._tool_registry = SimpleToolRegistry.with_tool_modules(
            modules=TOOL_CATEGORIES,
            agent=self,
            logger=self._logger,
            memory=memory,
            workspace=workspace,
            model_providers=chat_model_provider,
        )
        # self._tool_registry.set_agent(agent=self)

        ###
        ### Step 5 : Create the Loop
        ###
        self._loop: PlannerLoop = PlannerLoop()
        self._loop.set_agent(agent=self)

        # Set tool Executor
        self._tool_executor = ToolExecutor()
        self._tool_executor.set_agent(agent=self)

        ###
        ### Step 5a : Create the plan
        ###
        # FIXME: Long term : PlannerLoop / Pipeline get all ready tasks & launch them => Parralelle processing of tasks
        if hasattr( settings, "plan_id" ) and settings.plan_id is not None :
            self.plan: Plan = Plan.get_plan_from_db(settings.plan_id) # Plan(user_id=user_id)
            self._loop.set_current_task(self.plan.get_first_ready_task())
        else :
            self.plan: Plan = Plan.create_plan(agent= self)
            #self._loop.add_initial_tasks()
            self._loop.set_current_task(task =self.plan[0])
            self.plan_id = self.plan.plan_id

        ###
        ### Step 6 : add hooks/pluggins to the loop
        ###
        # TODO : Get hook added from configuration files
        # Exemple :
        # self.add_hook( hook: BaseLoopHook, uuid: uuid.UUID)
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

    def loophooks(self) -> PlannerLoop.LoophooksDict:
        if not self._loop._loophooks:
            self._loop._loophooks = {}
        return self._loop._loophooks

    def loop(self) -> PlannerLoop:
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
        cls, agent_settings: PlannerAgent.SystemSettings, logger: logging.Logger
    ) -> None:
        return cls._create_workspace(agent_settings=agent_settings, logger=logger)

    @classmethod
    def _create_workspace(
        cls,
        agent_settings: PlannerAgent.SystemSettings,
        logger: logging.Logger,
    ):
        from autogpts.autogpt.autogpt.core.workspace import SimpleWorkspace

        return SimpleWorkspace.create_workspace(
            user_id=agent_settings.user_id,
            agent_id=agent_settings.agent_id,
            settings=agent_settings,
            logger=logger,
        )

    @classmethod
    def get_strategies(cls) -> list:
        # TODO : Continue refactorization => move to loop ?
        from autogpts.autogpt.autogpt.core.agents.planner.strategies import (
            StrategiesSet)

        return StrategiesSet.get_strategies()

        # strategies_config = SimplePromptStrategiesConfiguration(
        #         name_and_goals=strategies.NameAndGoals.default_configuration,
        #         initial_plan=strategies.InitialPlan.default_configuration,
        #         next_ability=strategies.NextTool.default_configuration,)
        # #agent_settings.prompt_manager.configuration.prompt_strategies = strategies_config

        # #
        # # Dynamicaly load all strategies
        # # To do so Intanciate all class that inherit from PromptStrategy in package Strategy
        # #
        # simple_strategies = {}
        # import inspect
        # from  autogpt.core.agents.simple.lib.base import PromptStrategy
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

    @classmethod
    async def determine_agent_name_and_goals(
        cls,
        user_objective: str,
        agent_settings: PlannerAgent.SystemSettings,
        logger: logging.Logger,
    ) -> dict:
        logger.debug("Loading OpenAI provider.")
        provider: OpenAIProvider = cls._get_system_instance(
            "chat_model_provider",
            agent_settings,
            logger=logger,
        )
        logger.debug("Loading agent planner.")
        agent_planner: PromptManager = cls._get_system_instance(
            "prompt_manager",
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
        return "PlannerAgent()"

    @classmethod
    def load_prompt_settings(cls):
        return super().load_prompt_settings(erase=False, file_path=__file__)


def test_hook(**kwargs):
    test = "foo_bar"
    for key, value in kwargs.items():
        print(f"{key}: {value}")
