from __future__ import annotations

import enum
import logging
from typing import Any, Optional

import tiktoken
from pydantic import SecretStr

from forge.models.config import UserConfigurable

from ._openai_base import BaseOpenAIChatProvider
from .schema import (
    ChatModelInfo,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
)


class GroqModelName(str, enum.Enum):
    LLAMA3_8B = "llama3-8b-8192"
    LLAMA3_70B = "llama3-70b-8192"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    GEMMA_7B = "gemma-7b-it"


GROQ_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=GroqModelName.LLAMA3_8B,
            provider_name=ModelProviderName.GROQ,
            prompt_token_cost=0.05 / 1e6,
            completion_token_cost=0.10 / 1e6,
            max_tokens=8192,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=GroqModelName.LLAMA3_70B,
            provider_name=ModelProviderName.GROQ,
            prompt_token_cost=0.59 / 1e6,
            completion_token_cost=0.79 / 1e6,
            max_tokens=8192,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=GroqModelName.MIXTRAL_8X7B,
            provider_name=ModelProviderName.GROQ,
            prompt_token_cost=0.27 / 1e6,
            completion_token_cost=0.27 / 1e6,
            max_tokens=32768,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=GroqModelName.GEMMA_7B,
            provider_name=ModelProviderName.GROQ,
            prompt_token_cost=0.10 / 1e6,
            completion_token_cost=0.10 / 1e6,
            max_tokens=8192,
            has_function_call_api=True,
        ),
    ]
}


class GroqCredentials(ModelProviderCredentials):
    """Credentials for Groq."""

    api_key: SecretStr = UserConfigurable(from_env="GROQ_API_KEY")  # type: ignore
    api_base: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="GROQ_API_BASE_URL"
    )

    def get_api_access_kwargs(self) -> dict[str, str]:
        return {
            k: v.get_secret_value()
            for k, v in {
                "api_key": self.api_key,
                "base_url": self.api_base,
            }.items()
            if v is not None
        }


class GroqSettings(ModelProviderSettings):
    credentials: Optional[GroqCredentials]  # type: ignore
    budget: ModelProviderBudget  # type: ignore


class GroqProvider(BaseOpenAIChatProvider[GroqModelName, GroqSettings]):
    CHAT_MODELS = GROQ_CHAT_MODELS
    MODELS = CHAT_MODELS

    default_settings = GroqSettings(
        name="groq_provider",
        description="Provides access to Groq's API.",
        configuration=ModelProviderConfiguration(),
        credentials=None,
        budget=ModelProviderBudget(),
    )

    _settings: GroqSettings
    _configuration: ModelProviderConfiguration
    _credentials: GroqCredentials
    _budget: ModelProviderBudget

    def __init__(
        self,
        settings: Optional[GroqSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super(GroqProvider, self).__init__(settings=settings, logger=logger)

        from groq import AsyncGroq

        self._client = AsyncGroq(
            **self._credentials.get_api_access_kwargs()  # type: ignore
        )

    def get_tokenizer(self, model_name: GroqModelName) -> ModelTokenizer[Any]:
        # HACK: No official tokenizer is available for Groq
        return tiktoken.encoding_for_model("gpt-3.5-turbo")
