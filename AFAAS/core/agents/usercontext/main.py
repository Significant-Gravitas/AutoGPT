from __future__ import annotations

import uuid
from typing import Awaitable, Callable

from AFAAS.interfaces.db import AbstractMemory
from AFAAS.core.adapters.openai import AFAASChatOpenAI
from AFAAS.interfaces.workspace import AbstractFileWorkspace

from AFAAS.interfaces.agent import BaseAgent, BaseLoopHook, BasePromptManager
from .loop import UserContextLoop
from .models import UserContextAgentConfiguration, UserContextAgentSystems
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


class UserContextAgent(BaseAgent):
    ################################################################################
    ##################### REFERENCE SETTINGS FOR FACTORY ###########################
    ################################################################################

    CLASS_CONFIGURATION = UserContextAgentConfiguration
    CLASS_SYSTEMS = UserContextAgentSystems

    class SystemSettings(BaseAgent.SystemSettings):
        configuration: UserContextAgentConfiguration = UserContextAgentConfiguration()
        name = "usercontext_agent"
        description = "An agent that improve the quality of input provided by users."

        class Config(BaseAgent.SystemSettings.Config):
            pass

    def __init__(
        self,
        settings: UserContextAgent.SystemSettings,
        memory: AbstractMemory,
        chat_model_provider: AFAASChatOpenAI,
        prompt_manager: BasePromptManager,
        user_id: uuid.UUID,
        workspace: AbstractFileWorkspace,  # = AGPTLocalFileWorkspace.SystemSettings(),
        agent_id: uuid.UUID = None,
        **kwargs,
    ):
        super().__init__(
            settings=settings,
            memory=memory,
            workspace=workspace,
            prompt_manager=prompt_manager,
            user_id=user_id,
            agent_id=agent_id,
        )

        #
        # Step 1 : Set the chat model provider
        #
        self.default_llm_provider = chat_model_provider



        #
        # Step 3 : Set the chat model provider
        #
        # self._prompt_manager = prompt_manager
        # self._prompt_manager.set_agent(agent=self)

        self._loop: UserContextLoop = UserContextLoop()
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
        cls,
        agent_settings: UserContextAgent.SystemSettings,
    ) -> None:
        pass

    def __repr__(self):
        return "UserContextAgent()"
