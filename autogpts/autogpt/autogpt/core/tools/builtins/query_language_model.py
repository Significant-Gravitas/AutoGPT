import logging

from autogpt.core.tools.base import Tool, ToolConfiguration
from autogpt.core.tools.schema import ToolResult
from autogpt.core.planning.simple import LanguageModelConfiguration
from autogpt.core.plugin.simple import PluginLocation, PluginStorageFormat
from autogpt.core.resource.model_providers import (
    ChatMessage,
    ChatModelProvider,
    ModelProviderName,
    OpenAIModelName,
)


class QueryLanguageModel(Tool):
    default_configuration = ToolConfiguration(
        location=PluginLocation(
            storage_format=PluginStorageFormat.INSTALLED_PACKAGE,
            storage_route="autogpt.core.tools.builtins.QueryLanguageModel",
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
        configuration: ToolConfiguration,
        language_model_provider: ChatModelProvider,
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

    async def __call__(self, query: str) -> ToolResult:
        messages = [
            ChatMessage.user(
                content=query,
            ),
        ]
        model_response = await self._language_model_provider.create_language_completion(
            model_prompt=messages,
            functions=[],
            model_name=self._configuration.language_model_required.model_name,
            completion_parser=self._parse_response,
        )
        return ToolResult(
            ability_name=self.name(),
            ability_args={"query": query},
            success=True,
            message=model_response.content["content"],
        )

    @staticmethod
    def _parse_response(response_content: dict) -> dict:
        return {"content": response_content["content"]}
