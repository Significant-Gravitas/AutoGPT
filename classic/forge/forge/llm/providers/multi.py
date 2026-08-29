from __future__ import annotations

import logging
from typing import Any, AsyncIterator, Callable, Optional, Sequence, TypeVar, get_args

from pydantic import ValidationError

from .anthropic import ANTHROPIC_CHAT_MODELS, AnthropicModelName, AnthropicProvider
from .groq import GROQ_CHAT_MODELS, GroqModelName, GroqProvider
from .llamafile import LLAMAFILE_CHAT_MODELS, LlamafileModelName, LlamafileProvider
from .openai import OPEN_AI_CHAT_MODELS, OpenAIModelName, OpenAIProvider
from .schema import (
    AssistantChatMessage,
    BaseChatModelProvider,
    ChatMessage,
    ChatModelInfo,
    ChatModelResponse,
    CompletionModelFunction,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
)

_T = TypeVar("_T")

ModelName = AnthropicModelName | GroqModelName | LlamafileModelName | OpenAIModelName
EmbeddingModelProvider = OpenAIProvider

CHAT_MODELS = {
    **ANTHROPIC_CHAT_MODELS,
    **GROQ_CHAT_MODELS,
    **LLAMAFILE_CHAT_MODELS,
    **OPEN_AI_CHAT_MODELS,
}


class MultiProvider(BaseChatModelProvider[ModelName, ModelProviderSettings]):
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

    async def get_available_models(self) -> Sequence[ChatModelInfo[ModelName]]:
        # TODO: support embeddings
        return await self.get_available_chat_models()

    async def get_available_chat_models(self) -> Sequence[ChatModelInfo[ModelName]]:
        models = []
        async for provider in self.get_available_providers():
            models.extend(await provider.get_available_chat_models())
        return models

    def get_token_limit(self, model_name: ModelName) -> int:
        """Get the token limit for a given model."""
        return self.get_model_provider(model_name).get_token_limit(
            model_name  # type: ignore
        )

    def get_tokenizer(self, model_name: ModelName) -> ModelTokenizer[Any]:
        return self.get_model_provider(model_name).get_tokenizer(
            model_name  # type: ignore
        )

    def count_tokens(self, text: str, model_name: ModelName) -> int:
        return self.get_model_provider(model_name).count_tokens(
            text=text, model_name=model_name  # type: ignore
        )

    def count_message_tokens(
        self, messages: ChatMessage | list[ChatMessage], model_name: ModelName
    ) -> int:
        return self.get_model_provider(model_name).count_message_tokens(
            messages=messages, model_name=model_name  # type: ignore
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
            model_name=model_name,  # type: ignore
            completion_parser=completion_parser,
            functions=functions,
            max_output_tokens=max_output_tokens,
            prefill_response=prefill_response,
            **kwargs,
        )

    def get_model_provider(self, model: ModelName) -> ChatModelProvider:
        model_info = CHAT_MODELS[model]
        return self._get_provider(model_info.provider_name)

    async def get_available_providers(self) -> AsyncIterator[ChatModelProvider]:
        for provider_name in ModelProviderName:
            self._logger.debug(f"Checking if provider {provider_name} is available...")
            try:
                provider = self._get_provider(provider_name)
                await provider.get_available_models()  # check connection
                yield provider
                self._logger.debug(f"Provider '{provider_name}' is available!")
            except ValueError:
                pass
            except Exception as e:
                self._logger.debug(f"Provider '{provider_name}' is failing: {e}")

    def _get_provider(self, provider_name: ModelProviderName) -> ChatModelProvider:
        _provider = self._provider_instances.get(provider_name)
        if not _provider:
            Provider = self._get_provider_class(provider_name)
            self._logger.debug(
                f"{Provider.__name__} not yet in cache, trying to init..."
            )

            settings = Provider.default_settings.model_copy(deep=True)
            settings.budget = self._budget
            settings.configuration.extra_request_headers.update(
                self._settings.configuration.extra_request_headers
            )
            if settings.credentials is None:
                credentials_field = settings.model_fields["credentials"]
                Credentials = get_args(  # Union[Credentials, None] -> Credentials
                    credentials_field.annotation
                )[0]
                self._logger.debug(f"Loading {Credentials.__name__}...")
                try:
                    settings.credentials = Credentials.from_env()
                except ValidationError as e:
                    if credentials_field.is_required():
                        self._logger.debug(
                            f"Could not load (required) {Credentials.__name__}"
                        )
                        raise ValueError(
                            f"{Provider.__name__} is unavailable: "
                            "can't load credentials"
                        ) from e
                    self._logger.debug(
                        f"Could not load {Credentials.__name__}, continuing without..."
                    )

            self._provider_instances[provider_name] = _provider = Provider(
                settings=settings, logger=self._logger  # type: ignore
            )
            _provider._budget = self._budget  # Object binding not preserved by Pydantic
            self._logger.debug(f"Initialized {Provider.__name__}!")
        return _provider

    @classmethod
    def _get_provider_class(
        cls, provider_name: ModelProviderName
    ) -> type[AnthropicProvider | GroqProvider | OpenAIProvider]:
        try:
            return {
                ModelProviderName.ANTHROPIC: AnthropicProvider,
                ModelProviderName.GROQ: GroqProvider,
                ModelProviderName.LLAMAFILE: LlamafileProvider,
                ModelProviderName.OPENAI: OpenAIProvider,
            }[provider_name]
        except KeyError:
            raise ValueError(f"{provider_name} is not a known provider") from None

    def __repr__(self):
        return f"{self.__class__.__name__}()"


ChatModelProvider = (
    AnthropicProvider
    | GroqProvider
    | LlamafileProvider
    | OpenAIProvider
    | MultiProvider
)
