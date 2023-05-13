import logging
import re

from autogpt.core.configuration import Configuration
from autogpt.core.model.language.base import (
    LanguageModel,
    LanguageModelProvider,
    LanguageModelResponse,
)
from autogpt.core.planning import ModelPrompt


class SimpleLanguageModel(LanguageModel):
    def __init__(
        self,
        configuration: Configuration,
        logger: logging.Logger,
        model_provider: LanguageModelProvider,
    ):
        self._configuration = configuration.language_model
        self._logger = logger
        self._provider = model_provider

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
        model = "fast_model"
        return await self._provider.create_language_completion(
            model_prompt=objective_prompt,
            model_name=model,
            completion_parser=self._parse_agent_objective_model_response,
        )

    async def plan_next_action(
        self,
        planning_prompt: ModelPrompt,
        **kwargs,
    ) -> LanguageModelResponse:
        model = "fast_model"
        return await self._provider.create_language_completion(
            model_prompt=planning_prompt,
            model_name=model,
            completion_parser=self._parse_agent_action_model_response,
        )

    async def get_self_feedback(
        self,
        self_feedback_prompt: ModelPrompt,
        **kwargs,
    ) -> LanguageModelResponse:
        model = "fast_model"
        return await self._provider.create_language_completion(
            model_prompt=self_feedback_prompt,
            model_name=model,
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
