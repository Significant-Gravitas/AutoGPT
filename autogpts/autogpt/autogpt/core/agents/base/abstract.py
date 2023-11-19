from __future__ import annotations

import datetime
import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional

import yaml
from pydantic import Field, root_validator

from autogpts.AFAAS.app.lib.message_agent_user import MessageAgentUser
from autogpts.AFAAS.app.lib.message_agent_agent import MessageAgentAgent
from autogpts.AFAAS.app.lib.message_agent_llm import MessageAgentLLM

from autogpts.autogpt.autogpt.core.agents.base.loop import (  # Import only where it's needed
    BaseLoop)
from autogpts.autogpt.autogpt.core.agents.base.models import (
    BaseAgentConfiguration)
from autogpts.autogpt.autogpt.core.configuration import (
                                                         SystemSettings)


from  autogpts.AFAAS.app.sdk import forge_log
LOG = forge_log.ForgeLogger(__name__)

if(TYPE_CHECKING) :
    from .main import BaseAgent

class AbstractAgent(ABC):
    class SystemSettings(SystemSettings):
        configuration: BaseAgentConfiguration = BaseAgentConfiguration()

        user_id: str
        agent_id: str = Field(default_factory=lambda: "A" + str(uuid.uuid4()))

        # TODO: #22 https://github.com/ph-ausseil/afaas/issues/22
        modified_at: datetime.datetime = datetime.datetime.now()
        # TODO: #21 https://github.com/ph-ausseil/afaas/issues/21
        created_at: datetime.datetime = datetime.datetime.now()

        @staticmethod
        def _get_message_agent_user(agent_id):
            LOG.debug(f'Retriving :  _get_message_agent_user{agent_id}')
            return []
            # return MessageAgentUser.get_from_db(agent_id)

        @staticmethod
        def _get_message_agent_agent(agent_id):
            LOG.debug(f'Retriving :  _get_message_agent_agent{agent_id}')
            return []
            # return MessageAgentAgent.get_from_db(agent_id)

        @staticmethod
        def _get_message_agent_llm(agent_id):
            LOG.debug(f'Retriving :  _get_message_agent_llm{agent_id}')
            return []
            # return MessageAgentLLM.get_from_db(agent_id)

        # Use lambda functions to pass the agent_id
        # message_agent_user: list[MessageAgentUser] = Field(
        #     default_factory=lambda self: AbstractAgent.SystemSettings._get_message_agent_user(self.agent_id)
        #     )
        # message_agent_agent: list[MessageAgentAgent] = Field(
        #     default_factory=lambda self: AbstractAgent.SystemSettings._get_message_agent_agent(self.agent_id)
        #     )
        # message_agent_llm: list[MessageAgentLLM] = Field(
        #     default_factory=lambda self: AbstractAgent.SystemSettings._get_message_agent_llm(self.agent_id)
        #     )

        # NOTE: Should we switch to :  
        message_agent_user: list[MessageAgentUser] = []
        message_agent_agent: list[MessageAgentAgent] = []
        message_agent_llm: list[MessageAgentLLM] = []
        @root_validator(pre=True)
        def set_default_messages(cls, values):
            agent_id = values.get('agent_id', "A" + str(uuid.uuid4()))
            values['message_agent_user'] = cls._get_message_agent_user(agent_id)
            values['message_agent_agent'] = cls._get_message_agent_agent(agent_id)
            values['message_agent_llm'] = cls._get_message_agent_llm(agent_id)
            return values


        @property
        def _type_(self):
            # Nested Class
            return (
                self.__module__
                + "."
                + ".".join(self.__class__.__qualname__.split(".")[:-1])
            )

        @classmethod
        @property
        def settings_agent_class_(cls):
            return (
                cls.__qualname__.partition(".")[0]
            )

        @classmethod
        @property
        def settings_agent_module_(cls):
            return (
                cls.__module__
                + "."
                + ".".join(cls.__qualname__.split(".")[:-1])
            )
        
    _type_ : str = __name__
    _module_ = (
            __module__ + "." + __name__
        )
    
    
    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Abstract method for the initialization of the agent.

        Note: Implementation required in subclass.
        """
        ...

    @classmethod
    @abstractmethod
    def get_instance_from_settings(
        cls,
        agent_settings: BaseAgent.SystemSettings,
        logger: logging.Logger,
    ) -> "AbstractAgent":
        """
        Abstract method to retrieve an agent instance using provided settings.

        Args:
            agent_settings (BaseAgent.SystemSettings): The settings for the agent.
            logger (logging.Logger): Logger instance for logging purposes.

        Returns:
            BaseAgent: An instance of BaseAgent.

        Note: Implementation required in subclass.
        """
        ...

    @abstractmethod
    def __repr__(self):
        """
        Abstract method for the string representation of the agent.

        Returns:
            str: A string representation of the agent.

        Note: Implementation required in subclass.
        """
        ...

    _loop: BaseLoop = None
    # _loophooks: Dict[str, BaseLoop.LoophooksDict] = {}



AbstractAgent.SystemSettings.update_forward_refs()