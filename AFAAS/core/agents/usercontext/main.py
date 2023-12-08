from __future__ import annotations

import logging
import uuid
from typing import Awaitable, Callable

from AFAAS.app.core.memory.base import AbstractMemory
from AFAAS.app.core.resource.model_providers import \
    OpenAIProvider
from AFAAS.app.core.workspace.simple import SimpleWorkspace

from ..base import BaseAgent, BaseLoopHook, PromptManager
from .loop import UserContextLoop
from .models import UserContextAgentConfiguration, UserContextAgentSystems


class UserContextAgent(BaseAgent):
    ################################################################################
    ##################### REFERENCE SETTINGS FOR FACTORY ###########################
    ################################################################################

    CLASS_CONFIGURATION = UserContextAgentConfiguration
    CLASS_SYSTEMS = UserContextAgentSystems


    class SystemSettings(BaseAgent.SystemSettings):
        configuration : UserContextAgentConfiguration = UserContextAgentConfiguration()
        name="usercontext_agent"
        description="An agent that improve the quality of input provided by users."

        prompt_manager: PromptManager.SystemSettings = PromptManager.SystemSettings()

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
        **kwargs
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
        cls, agent_settings: UserContextAgent.SystemSettings, logger: logging.Logger
    ) -> None:
        pass

    @classmethod
    def get_strategies(cls)-> list :
        from AFAAS.app.core.agents.usercontext.strategies import \
            StrategiesSet
        return StrategiesSet.get_strategies()

    @classmethod
    async def determine_agent_name_and_goals(
        cls,
        user_objective: str,
        agent_settings: UserContextAgent.SystemSettings,
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
