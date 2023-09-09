import logging

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.planning.simple import LanguageModelConfiguration
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.resource.model_providers import (
    LanguageModelMessage,
    LanguageModelProvider,
    MessageRole,
    ModelProviderName,
    OpenAIModelName,
)


class QueryLanguageModel(Ability):
    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.QueryLanguageModel",
        ),
        language_model_required=LanguageModelConfiguration(
            model_name=OpenAIModelName.GPT3,
            provider_name=ModelProviderName.OPENAI,
            temperature=0.9,
        ),
    )

    def __init__(
        self,
        logger: logging.Logger,
        configuration: AbilityConfiguration,
        language_model_provider: LanguageModelProvider,
    ):
        self._logger = logger
        self._configuration = configuration
        self._language_model_provider = language_model_provider

    @classmethod
    def description(cls) -> str:
        return "Query a language model. A query should be a question and any relevant context."

    @classmethod
    def arguments(cls) -> dict:
        return {
            "query": {
                "type": "string",
                "description": "A query for a language model. A query should contain a question and any relevant context.",
            },
        }

    @classmethod
    def required_arguments(cls) -> list[str]:
        return ["query"]

    async def __call__(self, query: str) -> AbilityResult:
        messages = [
            LanguageModelMessage(
                content=query,
                role=MessageRole.USER,
            ),
        ]
        model_response = await self._language_model_provider.create_language_completion(
            model_prompt=messages,
            functions=[],
            model_name=self._configuration.language_model_required.model_name,
            completion_parser=self._parse_response,
        )
        return AbilityResult(
            ability_name=self.name(),
            ability_args={"query": query},
            success=True,
            message=model_response.content["content"],
        )

    @staticmethod
    def _parse_response(response_content: dict) -> dict:
        return {"content": response_content["content"]}
