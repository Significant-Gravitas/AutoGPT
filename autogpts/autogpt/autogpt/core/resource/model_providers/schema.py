from __future__ import annotations

import abc
import enum
from typing import (TYPE_CHECKING, Any, Callable, ClassVar, Dict, Generic,
                    List, Literal, Optional, Protocol, TypedDict, TypeVar,
                    Union)

from pydantic import BaseModel, Field, SecretStr, validator

from autogpts.autogpt.autogpt.core.configuration import UserConfigurable
from autogpts.autogpt.autogpt.core.resource.schema import (
    BaseProviderBudget, BaseProviderCredentials, BaseProviderSettings,
    BaseProviderUsage, Embedding, ResourceType)
from autogpts.autogpt.autogpt.core.utils.json_schema import JSONSchema


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


class BaseModelProviderCredentials(BaseProviderCredentials):
    """Credentials for a model provider."""

    api_key: str | None = UserConfigurable(default=None)
    api_type: str | None = UserConfigurable(default=None)
    api_base: str | None = UserConfigurable(default=None)
    api_version: str | None = UserConfigurable(default=None)
    deployment_id: str | None = UserConfigurable(default=None)

    def unmasked(self) -> dict:
        return unmask(self)

    class Config:
        extra = "ignore"


def unmask(model: BaseModel):
    unmasked_fields = {}
    for field_name, field in model.__fields__.items():
        value = getattr(model, field_name)
        if isinstance(value, SecretStr):
            unmasked_fields[field_name] = value.get_secret_value()
        else:
            unmasked_fields[field_name] = value
    return unmasked_fields


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
        self.remaining_budget -= incurred_cost


class BaseModelProviderSettings(BaseProviderSettings):
    resource_type = ResourceType.MODEL
    credentials: BaseModelProviderCredentials
    budget: BaseModelProviderBudget


class AbstractModelProvider(abc.ABC):
    """A ModelProvider abstracts the details of a particular provider of models."""

    default_settings: ClassVar[BaseModelProviderSettings]

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
