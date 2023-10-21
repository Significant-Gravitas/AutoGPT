from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Awaitable, Callable, List, Tuple, Optional
from pydantic import Field

from autogpts.autogpt.autogpt.core.tools import ToolResult, SimpleToolRegistry

from ..base import BaseAgent, PromptManager , BaseLoopHook
from .loop import UserContextLoop
from .models import (
    UserContextAgentConfiguration,
    UserContextAgentSettings,
    UserContextAgentSystems,
)

from autogpts.autogpt.autogpt.core.configuration import Configurable
from autogpts.autogpt.autogpt.core.memory.base import AbstractMemory
from autogpts.autogpt.autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpts.autogpt.autogpt.core.resource.model_providers import OpenAIProvider
from autogpts.autogpt.autogpt.core.workspace.simple import SimpleWorkspace


from autogpts.AFAAS.app.lib.tasks import Task, TaskStatusList


class UserContextAgent(BaseAgent):
    ################################################################################
    ##################### REFERENCE SETTINGS FOR FACTORY ###########################
    ################################################################################

    CLASS_CONFIGURATION = UserContextAgentConfiguration
    CLASS_SETTINGS = UserContextAgentSettings
    CLASS_SYSTEMS = UserContextAgentSystems


    class SystemSettings(BaseAgent.SystemSettings):
        configuration : UserContextAgentConfiguration = UserContextAgentConfiguration()
        name="usercontext_agent"
        description="An agent that improve the quality of input provided by users."
        # user_id: Optional[uuid.UUID] = Field(default=None)
        # agent_id: Optional[uuid.UUID] = Field(default=None)

        class Config(BaseAgent.SystemSettings.Config):
            pass

    def __init__(
        self,
        settings: UserContextAgent.SystemSettings,
        logger: logging.Logger,
        memory: AbstractMemory,
        chat_model_provider: OpenAIProvider,
        workspace: SimpleWorkspace,
        prompt_manager: PromptManager,
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

        self._loop : UserContextLoop = UserContextLoop()
        self._loop.set_agent(agent=self)

    def loophooks(self) -> UserContextLoop.LoophooksDict:
        if not self._loop._loophooks:
            self._loop._loophooks = {}
        return self._loop._loophooks

    def loop(self) -> UserContextLoop:
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
        cls, agent_settings: UserContextAgentSettings, logger: logging.Logger
    ) -> None:
        pass

    @classmethod
    def _get_agent_from_settings(
        cls,
        agent_settings: UserContextAgentSettings,
        agent_args: list,
        logger: logging.Logger,
    ) -> Tuple[UserContextAgentSettings, list]:
        agent_args["chat_model_provider"] = cls._get_system_instance(
            "chat_model_provider",
            agent_settings,
            logger,
        )
        from autogpts.autogpt.autogpt.core.agents.usercontext.strategies import (
            StrategiesSet,
            StrategiesSetConfiguration,
        )

        user_context_strategies = StrategiesSet.get_strategies(logger=logger)
        agent_args["prompt_manager"] = cls._get_system_instance(
            "prompt_manager",
            agent_settings,
            logger,
            model_providers={"openai": agent_args["chat_model_provider"]},
            strategies=user_context_strategies,
        )

        return agent_settings, agent_args

    """@classmethod
    def get_agent_from_settings(
        cls,
        agent_settings: UserContextAgentSettings,
        logger: logging.Logger,
    ) -> Agent:
        agent_settings, agent_args = super().get_agent_from_settings(
            agent_settings=agent_settings, 
            logger=logger
        )
        agent_args["chat_model_provider"] = cls._get_system_instance(
            "chat_model_provider",
            agent_settings,
            logger,
        )
        agent_args["prompt_manager"] = cls._get_system_instance(
            "prompt_manager",
            agent_settings,
            logger,
            model_providers={"openai": agent_args["chat_model_provider"]},
        )

        # NOTE : Can't be moved to super() because require agent_args["chat_model_provider"]
        agent_args["ability_registry"] = cls._get_system_instance(
            "ability_registry",
            agent_settings,
            logger,
            workspace=agent_args["workspace"],
            memory=agent_args["memory"],
            model_providers={"openai": agent_args["chat_model_provider"]},
        )
        return cls(**agent_args)"""

    @classmethod
    async def determine_agent_name_and_goals(
        cls,
        user_objective: str,
        agent_settings: UserContextAgentSettings,
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
    
    def load_prompt_settings(self) :
        self._logger.warning("TODO : load prompts via a jinja file")


    def __repr__(self):
        return "UserContextAgent()"
