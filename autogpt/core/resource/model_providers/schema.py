import abc
import enum
from typing import Callable, ClassVar

from pydantic import BaseModel, Field, SecretStr, validator

from autogpt.core.configuration import UserConfigurable
from autogpt.core.resource.schema import (
    Embedding,
    ProviderBudget,
    ProviderCredentials,
    ProviderSettings,
    ProviderUsage,
    ResourceType,
)


class ModelProviderService(str, enum.Enum):
    """A ModelService describes what kind of service the model provides."""

    EMBEDDING: str = "embedding"
    LANGUAGE: str = "language"
    TEXT: str = "text"


class ModelProviderName(str, enum.Enum):
    OPENAI: str = "openai"


class MessageRole(str, enum.Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class LanguageModelMessage(BaseModel):
    role: MessageRole
    content: str


class LanguageModelFunction(BaseModel):
    json_schema: dict


class ModelProviderModelInfo(BaseModel):
    """Struct for model information.

    Would be lovely to eventually get this directly from APIs, but needs to be
    scraped from websites for now.

    """

    name: str
    service: ModelProviderService
    provider_name: ModelProviderName
    prompt_token_cost: float = 0.0
    completion_token_cost: float = 0.0


class ModelProviderModelResponse(BaseModel):
    """Standard response struct for a response from a model."""

    prompt_tokens_used: int
    completion_tokens_used: int
    model_info: ModelProviderModelInfo


class ModelProviderCredentials(ProviderCredentials):
    """Credentials for a model provider."""

    api_key: SecretStr | None = UserConfigurable(default=None)
    api_type: SecretStr | None = UserConfigurable(default=None)
    api_base: SecretStr | None = UserConfigurable(default=None)
    api_version: SecretStr | None = UserConfigurable(default=None)
    deployment_id: SecretStr | None = UserConfigurable(default=None)

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


class ModelProviderUsage(ProviderUsage):
    """Usage for a particular model from a model provider."""

    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0

    def update_usage(
        self,
        model_response: ModelProviderModelResponse,
    ) -> None:
        self.completion_tokens += model_response.completion_tokens_used
        self.prompt_tokens += model_response.prompt_tokens_used
        self.total_tokens += (
            model_response.completion_tokens_used + model_response.prompt_tokens_used
        )


class ModelProviderBudget(ProviderBudget):
    total_budget: float = UserConfigurable()
    total_cost: float
    remaining_budget: float
    usage: ModelProviderUsage

    def update_usage_and_cost(
        self,
        model_response: ModelProviderModelResponse,
    ) -> None:
        """Update the usage and cost of the provider."""
        model_info = model_response.model_info
        self.usage.update_usage(model_response)
        incremental_cost = (
            model_response.completion_tokens_used * model_info.completion_token_cost
            + model_response.prompt_tokens_used * model_info.prompt_token_cost
        ) / 1000.0
        self.total_cost += incremental_cost
        self.remaining_budget -= incremental_cost


class ModelProviderSettings(ProviderSettings):
    resource_type = ResourceType.MODEL
    credentials: ModelProviderCredentials
    budget: ModelProviderBudget


class ModelProvider(abc.ABC):
    """A ModelProvider abstracts the details of a particular provider of models."""

    defaults: ClassVar[ModelProviderSettings]

    @abc.abstractmethod
    def get_token_limit(self, model_name: str) -> int:
        ...

    @abc.abstractmethod
    def get_remaining_budget(self) -> float:
        ...


####################
# Embedding Models #
####################


class EmbeddingModelProviderModelInfo(ModelProviderModelInfo):
    """Struct for embedding model information."""

    model_service = ModelProviderService.EMBEDDING
    embedding_dimensions: int


class EmbeddingModelProviderModelResponse(ModelProviderModelResponse):
    """Standard response struct for a response from an embedding model."""

    embedding: Embedding = Field(default_factory=list)

    @classmethod
    @validator("completion_tokens_used")
    def _verify_no_completion_tokens_used(cls, v):
        if v > 0:
            raise ValueError("Embeddings should not have completion tokens used.")
        return v


class EmbeddingModelProvider(ModelProvider):
    @abc.abstractmethod
    async def create_embedding(
        self,
        text: str,
        model_name: str,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelProviderModelResponse:
        ...


###################
# Language Models #
###################


class LanguageModelProviderModelInfo(ModelProviderModelInfo):
    """Struct for language model information."""

    model_service = ModelProviderService.LANGUAGE
    max_tokens: int


class LanguageModelProviderModelResponse(ModelProviderModelResponse):
    """Standard response struct for a response from a language model."""

    content: dict = None


class LanguageModelProvider(ModelProvider):
    @abc.abstractmethod
    async def create_language_completion(
        self,
        model_prompt: list[LanguageModelMessage],
        functions: list[LanguageModelFunction],
        model_name: str,
        completion_parser: Callable[[dict], dict],
        **kwargs,
    ) -> LanguageModelProviderModelResponse:
        ...
