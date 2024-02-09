from __future__ import annotations

import platform
import time
from typing import TYPE_CHECKING, Any

import importlib
from AFAAS.interfaces.agent.assistants.prompt_manager import AbstractPromptManager

import AFAAS.prompts.common as common_module
from AFAAS.interfaces.prompts.strategy import AbstractPromptStrategy
from AFAAS.prompts import BaseTaskRagStrategy, load_all_strategies

from AFAAS.interfaces.agent.features.agentmixin import AgentMixin
from AFAAS.interfaces.prompts.strategy import AbstractPromptStrategy

if TYPE_CHECKING:
    from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy)
    from AFAAS.interfaces.agent.main import BaseAgent

from AFAAS.interfaces.adapters import (
    AbstractChatModelProvider,
    AbstractChatModelResponse,
)
from AFAAS.interfaces.adapters.chatmodel import ChatModelWrapper
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


# FIXME: Find somewhere more appropriate
class SystemInfo(dict):
    os_info: str
    # provider : OpenAIProvider
    api_budget: float
    current_time: str

class BasePromptManager(AgentMixin, AbstractPromptManager):
    """Manages the agent's planning and goal-setting by constructing language model prompts."""

    def __init__(
        self,
    ) -> None:

        self._prompt_strategies = {}
        super().__init__()

    def add_strategies(self, strategies : list[AbstractPromptStrategy])->None : 
        for strategy in strategies:
            self._prompt_strategies[strategy.STRATEGY_NAME] = strategy

    def add_strategy(self, strategy: AbstractPromptStrategy) -> None:
        self._prompt_strategies[strategy.STRATEGY_NAME] = strategy

    def get_strategy(self, strategy_name: str) -> AbstractPromptStrategy:
        return self._prompt_strategies[strategy_name]

    def set_agent(self, agent: "BaseAgent"):
        if not hasattr(self, "_agent") or self._agent is None:
            super().set_agent(agent)
        self.load_strategies()

    def load_strategies(self) -> list[AbstractPromptStrategy]:

        # Dynamically load strategies from AFAAS.prompts.common
        for attribute_name in dir(common_module):
            attribute = getattr(common_module, attribute_name)
            if isinstance(attribute, type) and issubclass(attribute, AbstractPromptStrategy) and attribute not in (AbstractPromptStrategy, BaseTaskRagStrategy):
                self.load_strategy(strategy_module_attr=attribute)

        # TODO : This part should be migrated 
        # 1. Each Tool is in a Folder
        # 2. Each Folder has prompts
        # 3. Start migration with refine_user_context
        strategies: list[AbstractPromptStrategy] = []
        strategies = load_all_strategies()
        for strategy in strategies:
            self.add_strategy(strategy=strategy)

        return self._prompt_strategies

    def load_strategy(self, strategy_module_attr : type) : 
        if isinstance(strategy_module_attr, type) and issubclass(strategy_module_attr, AbstractPromptStrategy) and strategy_module_attr not in (AbstractPromptStrategy, BaseTaskRagStrategy):
            self.add_strategy( strategy_module_attr(**strategy_module_attr.default_configuration.dict()))

    async def execute_strategy(self, strategy_name: str, **kwargs) -> AbstractChatModelResponse:
        if strategy_name not in self._prompt_strategies:
            raise ValueError(f"Invalid strategy name {strategy_name}")

        prompt_strategy: AbstractPromptStrategy = self.get_strategy(strategy_name=strategy_name ) 
        if not hasattr(prompt_strategy, "_agent") or prompt_strategy._agent is None:
            prompt_strategy.set_agent(agent=self._agent)

        kwargs.update(self.get_system_info(prompt_strategy))

        LOG.trace(
            f"Executing strategy : {prompt_strategy.STRATEGY_NAME}"
        )

        prompt_strategy.set_tools(**kwargs)

        return await self.send_to_chatmodel(prompt_strategy, **kwargs)

    async def send(self, prompt_strategy : AbstractPromptStrategy, **kwargs):
        llm_provider = prompt_strategy.get_llm_provider()
        if (isinstance(llm_provider, AbstractChatModelProvider)):
            return await self.send_to_chatmodel(prompt_strategy, **kwargs)
        else :
            return await self.send_to_languagemodel(prompt_strategy, **kwargs)

    async def send_to_languagemodel(
        self,
        prompt_strategy: AbstractPromptStrategy,
        **kwargs,
    ) :
        raise NotImplementedError("Language Model not implemented")

    async def send_to_chatmodel(
        self,
        prompt_strategy: AbstractPromptStrategy,
        **kwargs,
    ) -> AbstractChatModelResponse:

        provider : AbstractChatModelProvider = prompt_strategy.get_llm_provider()
        model_configuration = prompt_strategy.get_prompt_config().dict()

        LOG.trace(f"Using model configuration: {model_configuration}")

        # FIXME : Check if Removable
        template_kwargs = self.get_system_info(prompt_strategy)
        template_kwargs.update(kwargs)
        template_kwargs.update(model_configuration)

        prompt = await prompt_strategy.build_message(**template_kwargs)
        llm_wrapper = ChatModelWrapper(llm_model=provider)

        response: AbstractChatModelResponse = await llm_wrapper.create_chat_completion(
            chat_messages=prompt.messages,
            tools=prompt.tools,
            **model_configuration,
            completion_parser=prompt_strategy.parse_response_content,
            tool_choice=prompt.tool_choice,
            default_tool_choice=prompt.default_tool_choice,
        )

        response.chat_messages = prompt.messages
        response.system_prompt = prompt.messages[0].content
        return response

    def get_system_info(self, strategy: AbstractPromptStrategy) -> SystemInfo:
        provider = strategy.get_llm_provider()
        template_kwargs = {
            "os_info": self.get_os_info(),
            "api_budget": provider.get_remaining_budget(),
            "current_time": time.strftime("%c"),
        }
        return template_kwargs

    @staticmethod
    def get_os_info() -> str:

        os_name = platform.system()
        if os_name != "Linux" :
            return platform.platform(terse=True)
        else :
            import distro
            return distro.name(pretty=True)

    def __repr__(self) -> str | tuple[Any, ...]:
        return f"{__class__.__name__}():\nAgent:{self._agent.agent_id}\nStrategies:{self._prompt_strategies}"
