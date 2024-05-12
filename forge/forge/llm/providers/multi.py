from __future__ import annotations

import logging
from typing import Callable, Iterator, Optional, TypeVar

from pydantic import ValidationError

from forge.models.config import Configurable

from .anthropic import ANTHROPIC_CHAT_MODELS, AnthropicModelName, AnthropicProvider
from .openai import OPEN_AI_CHAT_MODELS, OpenAIModelName, OpenAIProvider
from .schema import (
    AssistantChatMessage,
    ChatMessage,
    ChatModelInfo,
    ChatModelProvider,
    ChatModelResponse,
    CompletionModelFunction,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
)

_T = TypeVar("_T")

ModelName = AnthropicModelName | OpenAIModelName

CHAT_MODELS = {**ANTHROPIC_CHAT_MODELS, **OPEN_AI_CHAT_MODELS}


class MultiProvider(Configurable[ModelProviderSettings], ChatModelProvider):
    default_settings = ModelProviderSettings(
        name="multi_provider",
        description=(
            "Provides access to all of the available models, regardless of provider."
        ),
        configuration=ModelProviderConfiguration(
            retries_per_request=7,
        ),
        budget=ModelProviderBudget(),
    )

    _budget: ModelProviderBudget

    _provider_instances: dict[ModelProviderName, ChatModelProvider]

    def __init__(
        self,
        settings: Optional[ModelProviderSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super(MultiProvider, self).__init__(settings=settings, logger=logger)
        self._budget = self._settings.budget or ModelProviderBudget()

        self._provider_instances = {}

    async def get_available_models(self) -> list[ChatModelInfo]:
        models = []
        for provider in self.get_available_providers():
            models.extend(await provider.get_available_models())
        return models

    def get_token_limit(self, model_name: ModelName) -> int:
        """Get the token limit for a given model."""
        return self.get_model_provider(model_name).get_token_limit(model_name)

    @classmethod
    def get_tokenizer(cls, model_name: ModelName) -> ModelTokenizer:
        return cls._get_model_provider_class(model_name).get_tokenizer(model_name)

    @classmethod
    def count_tokens(cls, text: str, model_name: ModelName) -> int:
        return cls._get_model_provider_class(model_name).count_tokens(
            text=text, model_name=model_name
        )

    @classmethod
    def count_message_tokens(
        cls, messages: ChatMessage | list[ChatMessage], model_name: ModelName
    ) -> int:
        return cls._get_model_provider_class(model_name).count_message_tokens(
            messages=messages, model_name=model_name
        )

    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: ModelName,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        prefill_response: str = "",
        **kwargs,
    ) -> ChatModelResponse[_T]:
        """Create a completion using the Anthropic API."""
        return await self.get_model_provider(model_name).create_chat_completion(
            model_prompt=model_prompt,
            model_name=model_name,
            completion_parser=completion_parser,
            functions=functions,
            max_output_tokens=max_output_tokens,
            prefill_response=prefill_response,
            **kwargs,
        )

    def get_model_provider(self, model: ModelName) -> ChatModelProvider:
        model_info = CHAT_MODELS[model]
        return self._get_provider(model_info.provider_name)

    def get_available_providers(self) -> Iterator[ChatModelProvider]:
        for provider_name in ModelProviderName:
            try:
                yield self._get_provider(provider_name)
            except Exception:
                pass

    def _get_provider(self, provider_name: ModelProviderName) -> ChatModelProvider:
        _provider = self._provider_instances.get(provider_name)
        if not _provider:
            Provider = self._get_provider_class(provider_name)
            settings = Provider.default_settings.copy(deep=True)
            settings.budget = self._budget
            settings.configuration.extra_request_headers.update(
                self._settings.configuration.extra_request_headers
            )
            if settings.credentials is None:
                try:
                    Credentials = settings.__fields__["credentials"].type_
                    settings.credentials = Credentials.from_env()
                except ValidationError as e:
                    raise ValueError(
                        f"{provider_name} is unavailable: can't load credentials"
                    ) from e

            self._provider_instances[provider_name] = _provider = Provider(
                settings=settings, logger=self._logger
            )
            _provider._budget = self._budget  # Object binding not preserved by Pydantic
        return _provider

    @classmethod
    def _get_model_provider_class(
        cls, model_name: ModelName
    ) -> type[AnthropicProvider | OpenAIProvider]:
        return cls._get_provider_class(CHAT_MODELS[model_name].provider_name)

    @classmethod
    def _get_provider_class(
        cls, provider_name: ModelProviderName
    ) -> type[AnthropicProvider | OpenAIProvider]:
        try:
            return {
                ModelProviderName.ANTHROPIC: AnthropicProvider,
                ModelProviderName.OPENAI: OpenAIProvider,
            }[provider_name]
        except KeyError:
            raise ValueError(f"{provider_name} is not a known provider") from None

    def __repr__(self):
        return f"{self.__class__.__name__}()"
