import logging
from typing import ClassVar

from forge.json.model import JSONSchema
from forge.llm.providers import (
    ChatMessage,
    ChatModelProvider,
    ModelProviderName,
    OpenAIModelName,
)

from autogpt.core.ability.base import Ability, AbilityConfiguration
from autogpt.core.ability.schema import AbilityResult
from autogpt.core.planning.simple import LanguageModelConfiguration
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat


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
        language_model_provider: ChatModelProvider,
    ):
        self._logger = logger
        self._configuration = configuration
        self._language_model_provider = language_model_provider

    description: ClassVar[str] = (
        "Query a language model."
        " A query should be a question and any relevant context."
    )

    parameters: ClassVar[dict[str, JSONSchema]] = {
        "query": JSONSchema(
            type=JSONSchema.Type.STRING,
            description=(
                "A query for a language model. "
                "A query should contain a question and any relevant context."
            ),
        )
    }

    async def __call__(self, query: str) -> AbilityResult:
        model_response = await self._language_model_provider.create_chat_completion(
            model_prompt=[ChatMessage.user(query)],
            functions=[],
            model_name=self._configuration.language_model_required.model_name,
        )
        return AbilityResult(
            ability_name=self.name(),
            ability_args={"query": query},
            success=True,
            message=model_response.response.content or "",
        )
