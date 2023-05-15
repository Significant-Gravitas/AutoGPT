import enum
import logging
import re

from pydantic import Field

from autogpt.core.configuration import Configurable, SystemConfiguration, SystemSettings
from autogpt.core.model.base import (
    LanguageModel,
    LanguageModelProvider,
    LanguageModelResponse,
    ModelConfiguration,
    ProviderName,
)
from autogpt.core.model.providers.openai import OpenAIModelNames
from autogpt.core.planning import ModelPrompt


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


class SimpleLanguageModel(LanguageModel, Configurable):
    defaults = SystemSettings(
        name="simple_language_model",
        description="A simple language model.",
        configuration=LanguageModelConfiguration(
            models={
                LanguageModelClassification.FAST_MODEL: ModelConfiguration(
                    name=OpenAIModelNames.GPT3,
                    provider_name=ProviderName.OPENAI,
                    max_tokens=100,
                    temperature=0.9,
                ),
                LanguageModelClassification.SMART_MODEL: ModelConfiguration(
                    name=OpenAIModelNames.GPT3,
                    provider_name=ProviderName.OPENAI,
                    max_tokens=100,
                    temperature=0.9,
                ),
            },
        ),
    )

    def __init__(
        self,
        configuration: LanguageModelConfiguration,
        logger: logging.Logger,
        model_providers: dict[ProviderName, LanguageModelProvider],
    ):
        self._configuration = configuration
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
        model_config = self._configuration.models[model_classification]
        provider = self._providers[model_classification]
        return await provider.create_language_completion(
            model_prompt=objective_prompt,
            **model_config.dict(exclude_none=True),
            completion_parser=self._parse_agent_objective_model_response,
        )

    async def plan_next_action(
        self,
        planning_prompt: ModelPrompt,
        **kwargs,
    ) -> LanguageModelResponse:
        model_classification = LanguageModelClassification.FAST_MODEL
        model_name = self._configuration.models[model_classification].name
        provider = self._providers[model_classification]
        return await provider.create_language_completion(
            model_prompt=planning_prompt,
            model_name=model_name,
            completion_parser=self._parse_agent_action_model_response,
        )

    async def get_self_feedback(
        self,
        self_feedback_prompt: ModelPrompt,
        **kwargs,
    ) -> LanguageModelResponse:
        model_classification = LanguageModelClassification.FAST_MODEL
        model_name = self._configuration.models[model_classification].name
        provider = self._providers[model_classification]
        return await provider.create_language_completion(
            model_prompt=self_feedback_prompt,
            model_name=model_name,
            completion_parser=self._parse_agent_feedback_model_response,
        )

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
        ai_name = re.search(
            r"Name(?:\s*):(?:\s*)(.*)", response_text, re.IGNORECASE
        ).group(1)
        ai_role = (
            re.search(
                r"Description(?:\s*):(?:\s*)(.*?)(?:(?:\n)|Goals)",
                response_text,
                re.IGNORECASE | re.DOTALL,
            )
            .group(1)
            .strip()
        )
        ai_goals = re.findall(r"(?<=\n)-\s*(.*)", response_text)
        parsed_response = {
            "ai_name": ai_name,
            "ai_role": ai_role,
            "ai_goals": ai_goals,
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
