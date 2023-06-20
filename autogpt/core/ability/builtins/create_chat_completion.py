import logging

from autogpt.core.ability.base import AbilityConfiguration, Ability
from autogpt.core.ability.schema import (
    AbilityResult,
)
from autogpt.core.planning.simple import (
    LanguageModelConfiguration,
)
from autogpt.core.resource.model_providers import (
    LanguageModelProvider,
    MessageRole,
    LanguageModelMessage,
    ModelProviderName,
    OpenAIModelName,
)
from autogpt.core.plugin.simple import (
    PluginLocation,
    PluginStorageFormat,
)


class CreateChatCompletion(Ability):

    default_configuration = AbilityConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.ability.builtins.CreateChatCompletion",
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
        return "Create a chat completion from a list of messages."

    @classmethod
    def arguments(cls) -> dict:
        return {
            "messages": {
                "type": "array",
                "description": "A list of messages to use as the prompt for the chat completion.",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"enum": [r.value for r in MessageRole]},
                        "content": {"type": "string"},
                    },
                },
            },
        }

    async def __call__(self, messages: list[LanguageModelMessage]) -> AbilityResult:
        model_response = await self._language_model_provider.create_language_completion(
            model_prompt=messages,
            functions=[],
            model_name=self._configuration.language_model_required.model_name,
            completion_parser=self._parse_response,
        )
        return AbilityResult(
            success=True,
            message=model_response.content["text"],
        )

    @staticmethod
    def _parse_response(response_content: dict) -> dict:
        return {"content": response_content["content"]}
