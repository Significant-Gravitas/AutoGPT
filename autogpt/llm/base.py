from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor
from typing import List, Literal, TypedDict

MessageRole = Literal["system", "user", "assistant"]
MessageType = Literal["ai_response", "action_result"]


class MessageDict(TypedDict):
    role: MessageRole
    content: str


@dataclass
class Message:
    """OpenAI Message object containing a role and the message content"""

    role: MessageRole
    content: str
    type: MessageType | None = None

    def raw(self) -> MessageDict:
        return {"role": self.role, "content": self.content}


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


@dataclass
class TextModelInfo(ModelInfo):
    """Struct for text completion model information."""


@dataclass
class EmbeddingModelInfo(ModelInfo):
    """Struct for embedding model information."""

    embedding_dimensions: int


@dataclass
class ChatSequence:
    """Utility container for a chat sequence"""

    model: ChatModelInfo
    messages: list[Message] = field(default_factory=list)

    def __getitem__(self, i: int):
        return self.messages[i]

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def append(self, message: Message):
        return self.messages.append(message)

    def extend(self, messages: list[Message] | ChatSequence):
        return self.messages.extend(messages)

    def insert(self, index: int, *messages: Message):
        for message in reversed(messages):
            self.messages.insert(index, message)

    @classmethod
    def for_model(cls, model_name: str, messages: list[Message] | ChatSequence = []):
        from autogpt.llm.providers.openai import OPEN_AI_CHAT_MODELS

        if not model_name in OPEN_AI_CHAT_MODELS:
            raise ValueError(f"Unknown chat model '{model_name}'")

        return ChatSequence(
            model=OPEN_AI_CHAT_MODELS[model_name], messages=list(messages)
        )

    def add(self, message_role: MessageRole, content: str):
        self.messages.append(Message(message_role, content))

    @property
    def token_length(self):
        from autogpt.llm.utils import count_message_tokens

        return count_message_tokens(self.messages, self.model.name)

    def raw(self) -> list[MessageDict]:
        return [m.raw() for m in self.messages]

    def dump(self) -> str:
        SEPARATOR_LENGTH = 42

        def separator(text: str):
            half_sep_len = (SEPARATOR_LENGTH - 2 - len(text)) / 2
            return f"{floor(half_sep_len)*'-'} {text.upper()} {ceil(half_sep_len)*'-'}"

        formatted_messages = "\n".join(
            [f"{separator(m.role)}\n{m.content}" for m in self.messages]
        )
        return f"""
============== ChatSequence ==============
Length: {self.token_length} tokens; {len(self.messages)} messages
{formatted_messages}
==========================================
"""


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
