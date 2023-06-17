import json
import logging
import platform
import time

import distro

from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
from autogpt.core.memory import Memory
from autogpt.core.planning import templates

from autogpt.core.planning.base import (
    PromptStrategy,
)
from autogpt.core.planning.schema import (
    LanguageModelClassification,
    LanguageModelPrompt,
    LanguageModelResponse,
)
from autogpt.core.planning import strategies
from autogpt.core.resource.model_providers import (
    LanguageModelProvider,
    MessageRole,
    LanguageModelMessage,
    ModelProviderName,
    OpenAIModelName,
)
from autogpt.core.workspace import Workspace

from autogpt.json_utils.json_fix_llm import fix_json_using_multiple_techniques


class LanguageModelConfiguration(SystemConfiguration):
    """Struct for model configuration."""

    model_name: str = UserConfigurable()
    provider_name: ModelProviderName = UserConfigurable()
    temperature: float = UserConfigurable()


class PromptStrategiesConfiguration(SystemConfiguration):
    name_and_goals: strategies.NameAndGoalsConfiguration
    initial_plan: strategies.InitialPlanConfiguration


class PlannerConfiguration(SystemConfiguration):
    """Configuration for the Planner subsystem."""

    agent_name: str
    agent_role: str
    agent_goals: list[str]
    models: dict[LanguageModelClassification, LanguageModelConfiguration]
    prompt_strategies: PromptStrategiesConfiguration


class PlannerSettings(SystemSettings):
    """Settings for the Planner subsystem."""

    configuration: PlannerConfiguration


class SimplePlanner(Configurable):
    """Manages the agent's planning and goal-setting by constructing language model prompts."""

    defaults = PlannerSettings(
        name="planner",
        description="Manages the agent's planning and goal-setting by constructing language model prompts.",
        configuration=PlannerConfiguration(
            agent_name=templates.AGENT_NAME,
            agent_role=templates.AGENT_ROLE,
            agent_goals=templates.AGENT_GOALS,
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
                name_and_goals=strategies.NameAndGoalsConfiguration(
                    model_classification=LanguageModelClassification.SMART_MODEL,
                    system_prompt=strategies.NameAndGoals.DEFAULT_SYSTEM_PROMPT,
                    user_prompt_template=strategies.NameAndGoals.DEFAULT_USER_PROMPT_TEMPLATE,
                ),
                initial_plan=strategies.InitialPlanConfiguration(
                    model_classification=LanguageModelClassification.SMART_MODEL,
                    agent_preamble=strategies.InitialPlan.DEFAULT_AGENT_PREAMBLE,
                    agent_info=strategies.InitialPlan.DEFAULT_AGENT_INFO,
                    task_format=strategies.InitialPlan.DEFAULT_TASK_FORMAT,
                    triggering_prompt_template=strategies.InitialPlan.DEFAULT_TRIGGERING_PROMPT_TEMPLATE,
                    system_prompt_template=strategies.InitialPlan.DEFAULT_SYSTEM_PROMPT_TEMPLATE,
                ),
            ),
        ),
    )

    def __init__(
        self,
        settings: PlannerSettings,
        logger: logging.Logger,
        model_providers: dict[ModelProviderName, LanguageModelProvider],
        workspace: Workspace = None,  # Workspace is not available during bootstrapping.
    ) -> None:
        self._configuration = settings.configuration
        self._logger = logger
        self._workspace = workspace

        self._providers: dict[LanguageModelClassification, LanguageModelProvider] = {}
        for model, model_config in self._configuration.models.items():
            self._providers[model] = model_providers[model_config.provider_name]

        self._prompt_strategies = {
            'name_and_goals': strategies.NameAndGoals(
                **self._configuration.prompt_strategies.name_and_goals.dict()
            ),
            'initial_plan': strategies.InitialPlan(
                **self._configuration.prompt_strategies.initial_plan.dict()
            ),
        }

    async def decide_name_and_goals(self, user_objective: str) -> LanguageModelResponse:
        return await self.chat_with_model(
            self._prompt_strategies['name_and_goals'],
            user_objective=user_objective,
        )

    async def make_initial_plan(self) -> LanguageModelResponse:
        return await self.chat_with_model(
            self._prompt_strategies['initial_plan'],
            abilities=list(templates.ABILITIES),
        )

    async def chat_with_model(
        self,
        prompt_strategy: PromptStrategy,
        **kwargs,
    ) -> LanguageModelResponse:
        model_classification = prompt_strategy.model_classification
        model_configuration = self._configuration.models[model_classification].dict()
        self._logger.debug(f"Using model configuration: {model_configuration}")
        del model_configuration["provider_name"]
        provider = self._providers[model_classification]

        template_kwargs = self._make_template_kwargs_for_strategy(prompt_strategy)
        template_kwargs.update(kwargs)
        prompt = prompt_strategy.build_prompt(**template_kwargs)

        self._logger.debug(f"Using prompt:\n{prompt}\n\n")
        response = await provider.create_language_completion(
            model_prompt=prompt.messages,
            **model_configuration,
            completion_parser=prompt_strategy.parse_response_content,
        )
        return LanguageModelResponse.parse_obj(response.dict())

    def _make_template_kwargs_for_strategy(self, strategy: PromptStrategy):
        provider = self._providers[strategy.model_classification]
        template_kwargs = {
            "agent_name": self._configuration.agent_name,
            "agent_role": self._configuration.agent_role,
            "agent_goals": self._configuration.agent_goals,
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

