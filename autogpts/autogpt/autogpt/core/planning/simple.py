import logging
import platform
import time

import distro
from forge.llm.prompting import PromptStrategy
from forge.llm.prompting.schema import LanguageModelClassification
from forge.llm.providers import (
    ChatModelProvider,
    ChatModelResponse,
    CompletionModelFunction,
    ModelProviderName,
    OpenAIModelName,
)
from forge.models.config import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)

from autogpt.core.planning import prompt_strategies
from autogpt.core.planning.schema import Task
from autogpt.core.runner.client_lib.logging.helpers import dump_prompt
from autogpt.core.workspace import Workspace


class LanguageModelConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    model_name: str = UserConfigurable()
    provider_name: ModelProviderName = UserConfigurable()
    temperature: float = UserConfigurable()


class PromptStrategiesConfiguration(SystemConfiguration):
    name_and_goals: prompt_strategies.NameAndGoalsConfiguration
    initial_plan: prompt_strategies.InitialPlanConfiguration
    next_ability: prompt_strategies.NextAbilityConfiguration


class PlannerConfiguration(SystemConfiguration):
    """Configuration for the Planner subsystem."""

    models: dict[LanguageModelClassification, LanguageModelConfiguration]
    prompt_strategies: PromptStrategiesConfiguration


class PlannerSettings(SystemSettings):
    """Settings for the Planner subsystem."""

    configuration: PlannerConfiguration


class SimplePlanner(Configurable):
    """
    Manages the agent's planning and goal-setting
    by constructing language model prompts.
    """

    default_settings = PlannerSettings(
        name="planner",
        description=(
            "Manages the agent's planning and goal-setting "
            "by constructing language model prompts."
        ),
        configuration=PlannerConfiguration(
            models={
                LanguageModelClassification.FAST_MODEL: LanguageModelConfiguration(
                    model_name=OpenAIModelName.GPT3,
                    provider_name=ModelProviderName.OPENAI,
                    temperature=0.9,
                ),
                LanguageModelClassification.SMART_MODEL: LanguageModelConfiguration(
                    model_name=OpenAIModelName.GPT4,
                    provider_name=ModelProviderName.OPENAI,
                    temperature=0.9,
                ),
            },
            prompt_strategies=PromptStrategiesConfiguration(
                name_and_goals=prompt_strategies.NameAndGoals.default_configuration,
                initial_plan=prompt_strategies.InitialPlan.default_configuration,
                next_ability=prompt_strategies.NextAbility.default_configuration,
            ),
        ),
    )

    def __init__(
        self,
        settings: PlannerSettings,
        logger: logging.Logger,
        model_providers: dict[ModelProviderName, ChatModelProvider],
        workspace: Workspace = None,  # Workspace is not available during bootstrapping.
    ) -> None:
        self._configuration = settings.configuration
        self._logger = logger
        self._workspace = workspace

        self._providers: dict[LanguageModelClassification, ChatModelProvider] = {}
        for model, model_config in self._configuration.models.items():
            self._providers[model] = model_providers[model_config.provider_name]

        self._prompt_strategies = {
            "name_and_goals": prompt_strategies.NameAndGoals(
                **self._configuration.prompt_strategies.name_and_goals.dict()
            ),
            "initial_plan": prompt_strategies.InitialPlan(
                **self._configuration.prompt_strategies.initial_plan.dict()
            ),
            "next_ability": prompt_strategies.NextAbility(
                **self._configuration.prompt_strategies.next_ability.dict()
            ),
        }

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
        abilities: list[str],
    ) -> ChatModelResponse:
        return await self.chat_with_model(
            self._prompt_strategies["initial_plan"],
            agent_name=agent_name,
            agent_role=agent_role,
            agent_goals=agent_goals,
            abilities=abilities,
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

        template_kwargs = self._make_template_kwargs_for_strategy(prompt_strategy)
        template_kwargs.update(kwargs)
        prompt = prompt_strategy.build_prompt(**template_kwargs)

        self._logger.debug(f"Using prompt:\n{dump_prompt(prompt)}\n")
        response = await provider.create_chat_completion(
            model_prompt=prompt.messages,
            functions=prompt.functions,
            **model_configuration,
            completion_parser=prompt_strategy.parse_response_content,
        )
        return response

    def _make_template_kwargs_for_strategy(self, strategy: PromptStrategy):
        provider = self._providers[strategy.model_classification]
        template_kwargs = {
            "os_info": get_os_info(),
            "api_budget": provider.get_remaining_budget(),
            "current_time": time.strftime("%c"),
        }
        return template_kwargs


def get_os_info() -> str:
    os_name = platform.system()
    os_info = (
        platform.platform(terse=True)
        if os_name != "Linux"
        else distro.name(pretty=True)
    )
    return os_info
