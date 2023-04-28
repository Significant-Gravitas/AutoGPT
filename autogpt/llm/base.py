from dataclasses import dataclass, field
from typing import List, TypedDict


class Message(TypedDict):
    """OpenAI Message object containing a role and the message content"""

    role: str
    content: str


@dataclass
class ModelInfo:
    """Struct for model information.

    Would be lovely to eventually get this directly from APIs, but needs to be scraped from
    websites for now.

    """

    name: str
    prompt_token_cost: float
    completion_token_cost: float
    max_tokens: int


@dataclass
class ChatModelInfo(ModelInfo):
    """Struct for chat model information."""

    pass


@dataclass
class EmbeddingModelInfo(ModelInfo):
    """Struct for embedding model information."""

    embedding_dimensions: int


@dataclass
class LLMResponse:
    """Standard response struct for a response from an LLM model."""

    model_info: ModelInfo
    prompt_tokens_used: int = 0
    completion_tokens_used: int = 0


@dataclass
class EmbeddingModelResponse(LLMResponse):
    """Standard response struct for a response from an embedding model."""

    embedding: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.completion_tokens_used:
            raise ValueError("Embeddings should not have completion tokens used.")


@dataclass
class ChatModelResponse(LLMResponse):
    """Standard response struct for a response from an LLM model."""

    content: str = None
