import enum
import logging
import re

from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.model.base import (
    LanguageModel,
    LanguageModelResponse,
    ModelConfiguration,
)
from autogpt.core.planning import ModelPrompt
from autogpt.core.resource.model_providers import (
    LanguageModelProvider,
    ModelProviderName,
    OpenAIModelName,
)


class LanguageModelClassification(str, enum.Enum):
    """The LanguageModelClassification is a functional description of the model.

    This is used to determine what kind of model to use for a given prompt.
    Sometimes we prefer a faster or cheaper model to accomplish a task when
    possible.

    """

    FAST_MODEL: str = "fast_model"
    SMART_MODEL: str = "smart_model"


class LanguageModelConfiguration(SystemConfiguration):
    """Configuration for the language model."""

    models: dict[LanguageModelClassification, ModelConfiguration]


class LanguageModelSettings(SystemSettings):
    configuration: LanguageModelConfiguration


class SimpleLanguageModel(LanguageModel, Configurable):
    defaults = LanguageModelSettings(
        name="simple_language_model",
        description="A simple language model.",
        configuration=LanguageModelConfiguration(
            models={
                LanguageModelClassification.FAST_MODEL: ModelConfiguration(
                    model_name=OpenAIModelName.GPT3,
                    provider_name=ModelProviderName.OPENAI,
                    temperature=0.9,
                ),
                LanguageModelClassification.SMART_MODEL: ModelConfiguration(
                    model_name=OpenAIModelName.GPT4,
                    provider_name=ModelProviderName.OPENAI,
                    temperature=0.9,
                ),
            },
        ),
    )

    def __init__(
        self,
        settings: LanguageModelSettings,
        logger: logging.Logger,
        model_providers: dict[ModelProviderName, LanguageModelProvider],
    ):
        self._configuration = settings.configuration
        self._logger = logger

        # Map model classifications to model providers
        self._providers: dict[LanguageModelClassification, LanguageModelProvider] = {}
        for model, model_config in self._configuration.models.items():
            self._providers[model] = model_providers[model_config.provider_name]

    async def determine_agent_objective(
        self,
        objective_prompt: ModelPrompt,
        **kwargs,
    ) -> LanguageModelResponse:
        """Chat with a language model to determine the agent's objective.

        Args:
            objective_prompt: The prompt to use to determine the agent's objective.

        Returns:
            The response from the language model.

        """
        model_classification = LanguageModelClassification.FAST_MODEL
        model_config = self._configuration.models[model_classification].dict(
            exclude_none=True
        )
        # Provider name is useful for us but not for the provider which
        # already knows its own name.
        del model_config["provider_name"]
        provider = self._providers[model_classification]
        response = await provider.create_language_completion(
            model_prompt=objective_prompt,
            **model_config,
            completion_parser=self._parse_agent_objective_model_response,
        )
        return LanguageModelResponse.parse_obj(response.dict())

    async def plan_next_action(
        self,
        planning_prompt: ModelPrompt,
        **kwargs,
    ) -> LanguageModelResponse:
        model_classification = LanguageModelClassification.FAST_MODEL
        model_config = self._configuration.models[model_classification]
        provider = self._providers[model_classification]
        response = await provider.create_language_completion(
            model_prompt=planning_prompt,
            **model_config.dict(exclude_none=True),
            completion_parser=self._parse_agent_action_model_response,
        )

        return LanguageModelResponse.parse_obj(response.dict())

    async def get_self_feedback(
        self,
        self_feedback_prompt: ModelPrompt,
        **kwargs,
    ) -> LanguageModelResponse:
        model_classification = LanguageModelClassification.FAST_MODEL
        model_name = self._configuration.models[model_classification].name
        provider = self._providers[model_classification]
        response = await provider.create_language_completion(
            model_prompt=self_feedback_prompt,
            model_name=model_name,
            completion_parser=self._parse_agent_feedback_model_response,
        )

        return LanguageModelResponse.parse_obj(response.dict())

    @staticmethod
    def _parse_agent_objective_model_response(
        response_text: str,
    ) -> dict:
        """Parse the actual text response from the objective model.

        Args:
            response_text: The raw response text from the objective model.

        Returns:
            The parsed response.

        """
        agent_name = re.search(
            r"Name(?:\s*):(?:\s*)(.*)", response_text, re.IGNORECASE
        ).group(1)
        agent_role = (
            re.search(
                r"Description(?:\s*):(?:\s*)(.*?)(?:(?:\n)|Goals)",
                response_text,
                re.IGNORECASE | re.DOTALL,
            )
            .group(1)
            .strip()
        )
        agent_goals = re.findall(r"(?<=\n)-\s*(.*)", response_text)
        parsed_response = {
            "agent_name": agent_name,
            "agent_role": agent_role,
            "agent_goals": agent_goals,
        }
        return parsed_response

    @staticmethod
    def _parse_agent_action_model_response(
        response_text: str,
    ) -> dict:
        pass

    @staticmethod
    def _parse_agent_feedback_model_response(
        response_text: str,
    ) -> dict:
        pass

    def __repr__(self):
        return f"SimpleLanguageModel()"
