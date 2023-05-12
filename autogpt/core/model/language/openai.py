import logging
import re
from typing import Dict

from autogpt.core.configuration import Configuration
from autogpt.core.credentials import CredentialsManager
from autogpt.core.model import LanguageModelInfo, LanguageModelResponse
from autogpt.core.model.language import LanguageModel
from autogpt.core.model.providers.openai import (
    OPEN_AI_LANGUAGE_MODELS,
    OpenAIRetryHandler,
    create_completion,
    parse_credentials,
    parse_openai_response,
)
from autogpt.core.planning import ModelPrompt


class OpenAILanguageModel(LanguageModel):
    """OpenAI Language Model."""

    configuration_defaults = {
        "language_model": {
            "retries_per_request": 10,
            "use_azure": False,
            "models": {
                "fast_model": {
                    "name": "gpt-3.5-turbo",
                    "max_tokens": 100,
                    "temperature": 0.9,
                },
                "smart_model": {
                    "name": "gpt-4",
                    "max_tokens": 100,
                    "temperature": 0.9,
                },
            },
        }
    }

    def __init__(
        self,
        configuration: Configuration,
        logger: logging.Logger,
        credentials_manager: CredentialsManager,
    ):
        """Initialize the OpenAI Language Model.

        Args:
            model_info (openai.LanguageModelInfo): Model information.
        """
        self._configuration = configuration.language_model
        self._logger = logger

        raw_credentials = credentials_manager.get_credentials("openai")
        use_azure = self._configuration["use_azure"]
        self._credentials = {
            model_name: parse_credentials(model_name, raw_credentials, use_azure)
            for model_name in self._configuration["models"]
        }

        retry_handler = OpenAIRetryHandler(
            logger=self._logger,
            num_retries=self._configuration["retries_per_request"],
        )
        self._create_completion = retry_handler(create_completion)

    def list_models(self) -> Dict[str, LanguageModelInfo]:
        return OPEN_AI_LANGUAGE_MODELS.copy()

    def get_model_info(self, model_name: str) -> LanguageModelInfo:
        return OPEN_AI_LANGUAGE_MODELS[model_name]

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
        model_name = self._configuration["models"][model]["name"]

        response = await self._create_completion(
            messages=objective_prompt,
            **self._get_chat_kwargs(model, **kwargs),
        )
        return parse_openai_response(
            model_name,
            response,
            content_parser=self._parse_agent_objective_model_response,
        )

    async def plan_next_action(
        self,
        planning_prompt: ModelPrompt,
        **kwargs,
    ) -> LanguageModelResponse:
        pass

    async def get_self_feedback(
        self,
        self_feedback_prompt: ModelPrompt,
        **kwargs,
    ) -> LanguageModelResponse:
        pass

    def _get_chat_kwargs(self, model: str, **kwargs) -> dict:
        """Get kwargs for chat API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            The kwargs for the chat API call.

        """
        model_config = self._configuration["models"][model]
        chat_kwargs = {
            "model": model_config["name"],
            "max_tokens": kwargs.pop("max_tokens", model_config["max_tokens"]),
            "temperature": kwargs.pop("temperature", model_config["temperature"]),
            **self._credentials[model],
        }

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs.keys()}")

        return chat_kwargs

    @staticmethod
    def _parse_agent_objective_model_response(
        response_text: str,
    ) -> dict[str, str]:
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
