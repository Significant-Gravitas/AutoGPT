import abc

from pydantic import BaseModel


class ModelInfo(BaseModel):
    """Struct for model information.

    Would be lovely to eventually get this directly from APIs, but needs to be
    scraped from websites for now.

    """

    name: str
    prompt_token_cost: float
    completion_token_cost: float
    max_tokens: int


class ModelResponse(BaseModel):
    """Standard response struct for a response from an LLM model."""

    model_info: ModelInfo
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0


class ModelType(abc.ABC):
    """A ModelType abstracts implementation logic for a particular kind of model.

    Implementation details of a ModelType should be agnostic to the provider of the
    model (e.g. OpenAI, Anthropic, Google, etc.) and should focus on the
    details of domain logic.

    """

    ...


class ModelProvider(abc.ABC):
    """A ModelProvider abstracts the details of a particular provider of models.

    Implementation details of a ModelProvider should handle all translation from
    provider-specific details to the generic ModelType interface.

    """

    ...
