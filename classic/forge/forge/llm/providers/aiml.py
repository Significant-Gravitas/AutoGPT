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


class AimlModelName(str, enum.Enum):
    AIML_QWEN2_5_72B = "Qwen/Qwen2.5-72B-Instruct-Turbo"
    AIML_LLAMA3_1_70B = "nvidia/llama-3.1-nemotron-70b-instruct"
    AIML_LLAMA3_3_70B = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    AIML_LLAMA_3_2_3B = "meta-llama/Llama-3.2-3B-Instruct-Turbo"
    AIML_META_LLAMA_3_1_70B = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"


AIML_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=AimlModelName.AIML_QWEN2_5_72B,
            provider_name=ModelProviderName.AIML,
            prompt_token_cost=1.26 / 1e6,
            completion_token_cost=1.26 / 1e6,
            max_tokens=32000,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=AimlModelName.AIML_LLAMA3_1_70B,
            provider_name=ModelProviderName.AIML,
            prompt_token_cost=0.368 / 1e6,
            completion_token_cost=0.42 / 1e6,
            max_tokens=128000,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=AimlModelName.AIML_LLAMA3_3_70B,
            provider_name=ModelProviderName.AIML,
            prompt_token_cost=0.924 / 1e6,
            completion_token_cost=0.924 / 1e6,
            max_tokens=128000,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=AimlModelName.AIML_META_LLAMA_3_1_70B,
            provider_name=ModelProviderName.AIML,
            prompt_token_cost=0.063 / 1e6,
            completion_token_cost=0.063 / 1e6,
            max_tokens=131000,
            has_function_call_api=False,
        ),
        ChatModelInfo(
            name=AimlModelName.AIML_LLAMA_3_2_3B,
            provider_name=ModelProviderName.AIML,
            prompt_token_cost=0.924 / 1e6,
            completion_token_cost=0.924 / 1e6,
            max_tokens=128000,
            has_function_call_api=False,
        ),
    ]
}


class AimlCredentials(ModelProviderCredentials):
    """Credentials for Aiml."""

    api_key: SecretStr = UserConfigurable(from_env="AIML_API_KEY")  # type: ignore
    api_base: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="AIML_API_BASE_URL"
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


class AimlSettings(ModelProviderSettings):
    credentials: Optional[AimlCredentials]  # type: ignore
    budget: ModelProviderBudget  # type: ignore


class AimlProvider(BaseOpenAIChatProvider[AimlModelName, AimlSettings]):
    CHAT_MODELS = AIML_CHAT_MODELS
    MODELS = CHAT_MODELS

    default_settings = AimlSettings(
        name="aiml_provider",
        description="Provides access to AIML's API.",
        configuration=ModelProviderConfiguration(),
        credentials=None,
        budget=ModelProviderBudget(),
    )

    _settings: AimlSettings
    _configuration: ModelProviderConfiguration
    _credentials: AimlCredentials
    _budget: ModelProviderBudget

    def __init__(
        self,
        settings: Optional[AimlSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super(AimlProvider, self).__init__(settings=settings, logger=logger)

        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            **self._credentials.get_api_access_kwargs()  # type: ignore
        )

    def get_tokenizer(self, model_name: AimlModelName) -> ModelTokenizer[Any]:
        # HACK: No official tokenizer is available for AIML
        return tiktoken.encoding_for_model("gpt-3.5-turbo")
