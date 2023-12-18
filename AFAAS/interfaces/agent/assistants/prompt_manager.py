from __future__ import annotations

import platform
import time
from typing import TYPE_CHECKING

from pydantic import validator
from AFAAS.interfaces.agent.features.agentmixin import \
    AgentMixin

if TYPE_CHECKING:
    from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy)

from AFAAS.configs import (Configurable,
                                                         SystemConfiguration,
                                                         SystemSettings)


from AFAAS.interfaces.adapters import (
    BaseChatModelProvider, ChatModelResponse, ModelProviderName,
)
from AFAAS.core.adapters.openai import OpenAIModelName
from AFAAS.lib.sdk.logger import AFAASLogger


LOG = AFAASLogger(name=__name__)


# FIXME: Find somewhere more appropriate
class SystemInfo(dict):
    os_info: str
    # provider : OpenAIProvider
    api_budget: float
    current_time: str


class PromptManagerConfiguration(SystemConfiguration):
    """Configuration for the PromptManager subsystem."""
    pass


    # @validator("models")
    # def validate_models(cls, models):
    #     expected_keys = set( PromptStrategyLanguageModelClassification)
    #     actual_keys = set(models.keys())

    #     if expected_keys != actual_keys:
    #         missing_keys = expected_keys - actual_keys
    #         raise ValueError(f"Missing keys in 'models': {missing_keys}")

    #     return models

class PromptManager(Configurable, AgentMixin):
    """Manages the agent's planning and goal-setting by constructing language model prompts."""

    # default_settings = PromptManager.SystemSettings()
    class SystemSettings(SystemSettings):
        configuration: PromptManagerConfiguration = PromptManagerConfiguration()
        name = "prompt_manager"
        description = "Manages the agent's planning and goal-setting by constructing language model prompts."

    def __init__(
        self,
        settings: PromptManager.SystemSettings,
        agent_systems: list[Configurable],
        strategies: list[AbstractPromptStrategy],
    ) -> None:
        super().__init__(settings=settings)

        self._prompt_strategies = {}
        for strategy in strategies:
            self._prompt_strategies[strategy.STRATEGY_NAME] = strategy

        LOG.trace(
            f"PromptManager created with strategies : {self._prompt_strategies}"
        )

    async def execute_strategy(self, strategy_name: str, **kwargs) -> ChatModelResponse:
        """
        await simple_planner.execute_strategy('name_and_goals', user_objective='Learn Python')
        await simple_planner.execute_strategy('initial_plan', agent_name='Alice', agent_role='Student', agent_goals=['Learn Python'], tools=['coding'])
        await simple_planner.execute_strategy('initial_plan', agent_name='Alice', agent_role='Student', agent_goal_sentence=['Learn Python'], tools=['coding'])
        """
        if strategy_name not in self._prompt_strategies:
            raise ValueError(f"Invalid strategy name {strategy_name}")

        prompt_strategy: AbstractPromptStrategy = self._prompt_strategies[strategy_name]
        if not hasattr(prompt_strategy, "_agent") or prompt_strategy._agent is None:
            prompt_strategy.set_agent(agent=self._agent)

        kwargs.update(self.get_system_info(prompt_strategy))

        LOG.trace(
            f"Executing strategy : {prompt_strategy.STRATEGY_NAME}"
        )

        # MAKE FUNCTION DYNAMICS
        prompt_strategy.set_tools(**kwargs)

        return await self.chat_with_model(prompt_strategy, **kwargs)

    async def chat_with_model(
        self,
        prompt_strategy: AbstractPromptStrategy,
        **kwargs,
    ) -> ChatModelResponse:
        
        provider : BaseChatModelProvider = prompt_strategy.get_llm_provider()
        model_configuration = prompt_strategy.get_prompt_config().dict()

        LOG.trace(f"Using model configuration: {model_configuration}")
        
        # FIXME : Check if Removable
        template_kwargs = self.get_system_info(prompt_strategy)
        template_kwargs.update(kwargs)
        template_kwargs.update(model_configuration)

        prompt = prompt_strategy.build_message(**template_kwargs)

        response: ChatModelResponse = await provider.create_chat_completion(
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
            "os_info": get_os_info(),
            "api_budget": provider.get_remaining_budget(),
            "current_time": time.strftime("%c"),
        }
        return template_kwargs
    

def get_os_info() -> str:

    os_name = platform.system()
    if os_name != "Linux" :
        return platform.platform(terse=True)
    else :
        import distro
        return distro.name(pretty=True)
