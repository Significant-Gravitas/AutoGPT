import logging
import platform
import time

from pydantic import validator

from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)

from autogpt.core.planning.schema import Task
from autogpt.core.prompting.base import (
    BasePromptStrategy,
    PromptStrategy,
    PromptStrategiesConfiguration,
)
from autogpt.core.prompting.schema import LanguageModelClassification
from autogpt.core.resource.model_providers import (
    ChatModelProvider,
    ChatModelResponse,
    CompletionModelFunction,
    ModelProviderName,
    OpenAIModelName,
    OpenAIProvider
)
from autogpt.core.workspace import Workspace

# FIXME : Find somewhere more appropriate
class SystemInfo(dict) : 
    os_info: str
    # provider : OpenAIProvider
    api_budget: float
    current_time: str


class LanguageModelConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    model_name: str = UserConfigurable()
    provider_name: ModelProviderName = UserConfigurable()
    temperature: float = UserConfigurable()


class PlannerConfiguration(SystemConfiguration):
    """Configuration for the Planner subsystem."""

    models: dict[LanguageModelClassification, LanguageModelConfiguration]

    @validator("models")
    def validate_models(cls, models):
        expected_keys = set(LanguageModelClassification)
        actual_keys = set(models.keys())

        if expected_keys != actual_keys:
            missing_keys = expected_keys - actual_keys
            raise ValueError(f"Missing keys in 'models': {missing_keys}")

        return models


class PlannerSettings(SystemSettings):
    """Settings for the Planner subsystem."""

    configuration: PlannerConfiguration


class SimplePlanner(Configurable):
    """Manages the agent's planning and goal-setting by constructing language model prompts."""

    default_settings = PlannerSettings(
        name="planner",
        description="Manages the agent's planning and goal-setting by constructing language model prompts.",
        configuration=PlannerConfiguration(
            models={
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
            },
        ),
    )

    def __init__(
        self,
        settings: PlannerSettings,
        logger: logging.Logger,
        model_providers: dict[ModelProviderName, ChatModelProvider],
        strategies: dict[str, PromptStrategy],
        workspace: Workspace = None,  # Workspace is not available during bootstrapping.
    ) -> None:
        super().__init__(settings, logger)
        self._workspace = workspace

        self._providers: dict[LanguageModelClassification, ChatModelProvider] = {}
        for model, model_config in self._configuration.models.items():
            self._providers[model] = model_providers[model_config.provider_name]

        self._prompt_strategies = {}
        for strategy in strategies:
            self._prompt_strategies[strategy.STRATEGY_NAME] = strategy

        logger.debug(f"Planner created with strategies : {self._prompt_strategies}")
        # self._prompt_strategies = strategies

    async def execute_strategy(self, strategy_name: str, **kwargs) -> ChatModelResponse:
        """
        await simple_planner.execute_strategy('name_and_goals', user_objective='Learn Python')
        await simple_planner.execute_strategy('initial_plan', agent_name='Alice', agent_role='Student', agent_goals=['Learn Python'], tools=['coding'])
        await simple_planner.execute_strategy('initial_plan', agent_name='Alice', agent_role='Student', agent_goal_sentence=['Learn Python'], tools=['coding'])
        """
        if strategy_name not in self._prompt_strategies:
            raise ValueError(f"Invalid strategy name {strategy_name}")

        prompt_strategy = self._prompt_strategies[strategy_name]

        kwargs.update(self.get_system_info(prompt_strategy))

        return await self.chat_with_model(prompt_strategy, **kwargs)

    async def decide_name_and_goals(self, user_objective: str) -> ChatModelResponse:
        return await self.chat_with_model(
            self._prompt_strategies["name_and_goals"],
            user_objective=user_objective,
        )

    async def make_initial_plan(
        self,
        agent_name: str,
        agent_role: str,
        agent_goals: list[str],
        tools: list[str],
    ) -> ChatModelResponse:
        return await self.chat_with_model(
            self._prompt_strategies["initial_plan"],
            agent_name=agent_name,
            agent_role=agent_role,
            agent_goals=agent_goals,
            tools=tools,
        )

    async def determine_next_ability(
        self,
        task: Task,
        ability_specs: list[CompletionModelFunction],
    ):
        return await self.chat_with_model(
            self._prompt_strategies["next_ability"],
            task=task,
            ability_specs=ability_specs,
        )

    async def chat_with_model(
        self,
        prompt_strategy: PromptStrategy,
        **kwargs,
    ) -> ChatModelResponse:
        model_classification = prompt_strategy.model_classification
        model_configuration = self._configuration.models[model_classification].dict()
        self._logger.debug(f"Using model configuration: {model_configuration}")
        del model_configuration["provider_name"]
        provider = self._providers[model_classification]

        # FIXME : Check if Removable
        template_kwargs = self.get_system_info(prompt_strategy)
        template_kwargs.update(kwargs)
        template_kwargs.update(model_configuration)

        prompt = prompt_strategy.build_prompt(**template_kwargs)

        self._logger.debug(f"Using prompt:\n{prompt}\n\n")
        response = await provider.create_chat_completion(
            model_prompt=prompt.messages,
            functions=prompt.functions,
            **model_configuration,
            completion_parser=prompt_strategy.parse_response_content,
            function_call=prompt.function_call,
            default_function_call=prompt.default_function_call,
        )
        return response

    def get_system_info(self, strategy: PromptStrategy) -> SystemInfo:
        provider = self._providers[strategy.model_classification]
        template_kwargs = {
            "os_info": get_os_info(),
            # "provider" :  provider,
            "api_budget": provider.get_remaining_budget(),
            "current_time": time.strftime("%c"),
        }
        return template_kwargs


# FIXME : Only import distro when required
import distro
def get_os_info() -> str:
    os_name = platform.system()
    os_info = (
        platform.platform(terse=True)
        if os_name != "Linux"
        else distro.name(pretty=True)
    )
    return os_info
