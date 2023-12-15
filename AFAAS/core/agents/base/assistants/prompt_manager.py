from __future__ import annotations

import platform
import time
from typing import TYPE_CHECKING

from pydantic import validator

from AFAAS.core.agents.base.features.agentmixin import \
    AgentMixin

if TYPE_CHECKING:
    pass

from AFAAS.core.configuration import (Configurable,
                                                         SystemConfiguration,
                                                         SystemSettings,
                                                         UserConfigurable)
from AFAAS.core.prompting.base import (
    AbstractPromptStrategy, BasePromptStrategy)
from AFAAS.core.prompting.schema import \
    LanguageModelClassification
from AFAAS.core.resource.model_providers import (
    BaseChatModelProvider, ChatModelResponse, ModelProviderName,
    OpenAIModelName)
from AFAAS.core.workspace import AbstractFileWorkspace
from AFAAS.core.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)


# FIXME: Find somewhere more appropriate
class SystemInfo(dict):
    os_info: str
    # provider : OpenAIProvider
    api_budget: float
    current_time: str


class LanguageModelConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    model_name: str = UserConfigurable()
    provider_name: ModelProviderName = UserConfigurable()
    temperature: float = UserConfigurable()


class PromptManagerConfiguration(SystemConfiguration):
    """Configuration for the PromptManager subsystem."""

    models: dict[LanguageModelClassification, LanguageModelConfiguration] = {
        LanguageModelClassification.FAST_MODEL_4K: LanguageModelConfiguration(
            model_name=OpenAIModelName.GPT3,
            provider_name=ModelProviderName.OPENAI,
            temperature=0.9,
        ),
        LanguageModelClassification.FAST_MODEL_16K: LanguageModelConfiguration(
            model_name=OpenAIModelName.GPT3_16k,
            provider_name=ModelProviderName.OPENAI,
            temperature=0.9,
        ),
        LanguageModelClassification.FAST_MODEL_FINE_TUNED_4K: LanguageModelConfiguration(
            model_name=OpenAIModelName.GPT3_FINE_TUNED,
            provider_name=ModelProviderName.OPENAI,
            temperature=0.9,
        ),
        LanguageModelClassification.SMART_MODEL_8K: LanguageModelConfiguration(
            model_name=OpenAIModelName.GPT4,
            provider_name=ModelProviderName.OPENAI,
            temperature=0.9,
        ),
        LanguageModelClassification.SMART_MODEL_32K: LanguageModelConfiguration(
            model_name=OpenAIModelName.GPT4_32k,
            provider_name=ModelProviderName.OPENAI,
            temperature=0.9,
        ),
    }

    @validator("models")
    def validate_models(cls, models):
        expected_keys = set(LanguageModelClassification)
        actual_keys = set(models.keys())

        if expected_keys != actual_keys:
            missing_keys = expected_keys - actual_keys
            raise ValueError(f"Missing keys in 'models': {missing_keys}")

        return models

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
    ) -> None:
        super().__init__(settings=settings)
        model_providers: dict[ModelProviderName, BaseChatModelProvider] = {
            "openai": agent_systems["chat_model_provider"]
        }
        strategies: dict[str, AbstractPromptStrategy] = agent_systems["strategies"]
        workspace: AbstractFileWorkspace = agent_systems["workspace"]

        self._workspace = workspace

        self._providers: dict[LanguageModelClassification, BaseChatModelProvider] = {}
        for model, model_config in self._configuration.models.items():
            self._providers[model] = model_providers[model_config.provider_name]

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

        prompt_strategy: BasePromptStrategy = self._prompt_strategies[strategy_name]
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
        model_classification = prompt_strategy.model_classification
        model_configuration = self._configuration.models[model_classification].dict()
        LOG.trace(f"Using model configuration: {model_configuration}")
        del model_configuration["provider_name"]
        provider = self._providers[model_classification]

        # FIXME : Check if Removable
        template_kwargs = self.get_system_info(prompt_strategy)
        template_kwargs.update(kwargs)
        template_kwargs.update(model_configuration)

        prompt = prompt_strategy.build_prompt(**template_kwargs)

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
        provider = self._providers[strategy.model_classification]
        template_kwargs = {
            "os_info": get_os_info(),
            # "provider" :  provider,
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
