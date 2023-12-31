from __future__ import annotations

import abc
import enum
from typing import Callable, ClassVar, Protocol

from pydantic import BaseModel, Field, validator

from AFAAS.configs import SystemConfiguration, UserConfigurable
from AFAAS.interfaces.adapters.configuration import (
    BaseProviderBudget,
    BaseProviderCredentials,
    BaseProviderSettings,
    BaseProviderUsage,
    Embedding,
)


class ModelProviderService(str, enum.Enum):
    """A ModelService describes what kind of service the model provides."""

    EMBEDDING: str = "embedding"
    CHAT: str = "language"
    TEXT: str = "text"


class ModelProviderName(str, enum.Enum):
    OPENAI: str = "openai"


class BaseModelInfo(BaseModel):
    """Struct for model information.

    Would be lovely to eventually get this directly from APIs, but needs to be
    scraped from websites for now.
    """

    name: str
    service: ModelProviderService
    provider_name: ModelProviderName
    prompt_token_cost: float = 0.0
    completion_token_cost: float = 0.0


class BaseModelResponse(BaseModel):
    """Standard response struct for a response from a model."""

    prompt_tokens_used: int
    completion_tokens_used: int
    model_info: BaseModelInfo


class BaseModelProviderConfiguration(SystemConfiguration):
    retries_per_request: int = UserConfigurable()
    extra_request_headers: dict[str, str] = Field(default_factory=dict)


class BaseModelProviderCredentials(BaseProviderCredentials):
    """Credentials for a model provider."""

    api_key: str | None = UserConfigurable(default=None)
    api_type: str | None = UserConfigurable(default=None)
    api_base: str | None = UserConfigurable(default=None)
    api_version: str | None = UserConfigurable(default=None)
    deployment_id: str | None = UserConfigurable(default=None)

    class Config:
        extra = "ignore"


class BaseModelProviderUsage(BaseProviderUsage):
    """Usage for a particular model from a model provider."""

    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0

    def update_usage(
        self,
        model_response: BaseModelResponse,
    ) -> None:
        self.completion_tokens += model_response.completion_tokens_used
        self.prompt_tokens += model_response.prompt_tokens_used
        self.total_tokens += (
            model_response.completion_tokens_used + model_response.prompt_tokens_used
        )


class BaseModelProviderBudget(BaseProviderBudget):
    total_budget: float = UserConfigurable()
    total_cost: float
    remaining_budget: float
    usage: BaseModelProviderUsage

    def update_usage_and_cost(
        self,
        model_response: BaseModelResponse,
    ) -> None:
        """Update the usage and cost of the provider."""
        model_info = model_response.model_info
        self.usage.update_usage(model_response)
        incurred_cost = (
            model_response.completion_tokens_used * model_info.completion_token_cost
            + model_response.prompt_tokens_used * model_info.prompt_token_cost
        )
        self.total_cost += incurred_cost
        if abs(self.remaining_budget) != float("inf"):
            self.remaining_budget -= incurred_cost


class BaseModelProviderSettings(BaseProviderSettings):
    configuration: BaseModelProviderConfiguration
    credentials: BaseModelProviderCredentials
    budget: BaseModelProviderBudget


class AbstractModelProvider(abc.ABC):
    """A ModelProvider abstracts the details of a particular provider of models."""

    default_settings: ClassVar[BaseModelProviderSettings]

    _configuration: BaseModelProviderConfiguration

    @abc.abstractmethod
    def count_tokens(self, text: str, model_name: str) -> int:
        ...

    @abc.abstractmethod
    def get_tokenizer(self, model_name: str) -> "ModelTokenizer":
        ...

    @abc.abstractmethod
    def get_token_limit(self, model_name: str) -> int:
        ...

    @abc.abstractmethod
    def get_remaining_budget(self) -> float:
        ...


class AbstractLanguageModelProvider(AbstractModelProvider):
    @abc.abstractmethod
    def has_oa_tool_calls_api(self, model_name: str) -> bool:
        ...

    @abc.abstractmethod
    def get_default_config(self) -> AbstractPromptConfiguration:
        ...


class ModelTokenizer(Protocol):
    """A ModelTokenizer provides tokenization specific to a model."""

    @abc.abstractmethod
    def encode(self, text: str) -> list:
        ...

    @abc.abstractmethod
    def decode(self, tokens: list) -> str:
        ...


####################
# Embedding Models #
####################


class EmbeddingModelInfo(BaseModelInfo):
    """Struct for embedding model information."""

    llm_service = ModelProviderService.EMBEDDING
    embedding_dimensions: int


class EmbeddingModelResponse(BaseModelResponse):
    """Standard response struct for a response from an embedding model."""

    embedding: Embedding = Field(default_factory=list)

    @classmethod
    @validator("completion_tokens_used")
    def _verify_no_completion_tokens_used(cls, v):
        if v > 0:
            raise ValueError("Embeddings should not have completion tokens used.")
        return v


class EmbeddingModelProvider(AbstractModelProvider):
    @abc.abstractmethod
    async def create_embedding(
        self,
        text: str,
        model_name: str,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelResponse:
        ...


class AbstractPromptConfiguration(abc.ABC, SystemConfiguration):
    """Struct for model configuration."""
