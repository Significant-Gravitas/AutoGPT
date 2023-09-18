from __future__ import annotations
import json

from math import ceil, floor
from pydantic import BaseModel, Field
from typing import Any, Literal, Optional, Type, TypedDict, TypeVar, overload

MessageRole = Literal["system", "user", "assistant", "function"]
MessageType = Literal["ai_response", "action_result"]

TText = list[int]
"""Token array representing tokenized text"""


class MessageDict(TypedDict):
    role: MessageRole
    content: str


class ResponseMessageDict(TypedDict):
    role: Literal["assistant"]
    content: Optional[str]
    function_call: Optional[FunctionCallDict]


class FunctionCallDict(TypedDict):
    name: str
    arguments: str


class Message(BaseModel):
    """OpenAI Message object containing a role and the message content"""

    role: MessageRole
    content: str
    type: Optional[MessageType]

    def __init__(
        self,
        role: MessageRole,
        content: str,
        type: Optional[MessageType] = None
    ):
        super().__init__(
            role=role,
            content=content,
            type=type,
        )

    def raw(self) -> MessageDict:
        return {"role": self.role, "content": self.content}


class ModelInfo(BaseModel):
    """Struct for model information.

    Would be lovely to eventually get this directly from APIs, but needs to be scraped from
    websites for now.
    """

    name: str
    max_tokens: int
    prompt_token_cost: float


class CompletionModelInfo(ModelInfo):
    """Struct for generic completion model information."""

    completion_token_cost: float


class ChatModelInfo(CompletionModelInfo):
    """Struct for chat model information."""

    supports_functions: bool = False


class TextModelInfo(CompletionModelInfo):
    """Struct for text completion model information."""


class EmbeddingModelInfo(ModelInfo):
    """Struct for embedding model information."""

    embedding_dimensions: int


# Can be replaced by Self in Python 3.11
TChatSequence = TypeVar("TChatSequence", bound="ChatSequence")


class ChatSequence(BaseModel):
    """Utility container for a chat sequence"""

    model: ChatModelInfo
    messages: list[Message] = Field(default_factory=list[Message])

    @overload
    def __getitem__(self, key: int) -> Message:
        ...

    @overload
    def __getitem__(self: TChatSequence, key: slice) -> TChatSequence:
        ...

    def __getitem__(self: TChatSequence, key: int | slice) -> Message | TChatSequence:
        if isinstance(key, slice):
            copy = self.copy(deep=True)
            copy.messages = self.messages[key]
            return copy
        return self.messages[key]

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def add(
        self,
        message_role: MessageRole,
        content: str,
        type: MessageType | None = None,
    ) -> None:
        self.append(Message(message_role, content, type))

    def append(self, message: Message):
        return self.messages.append(message)

    def extend(self, messages: list[Message] | ChatSequence):
        return self.messages.extend(messages)

    def insert(self, index: int, *messages: Message):
        for message in reversed(messages):
            self.messages.insert(index, message)

    @classmethod
    def for_model(
        cls: Type[TChatSequence],
        model_name: str,
        messages: list[Message] | ChatSequence = [],
        **kwargs,
    ) -> TChatSequence:
        from autogpt.llm.providers.openai import OPEN_AI_CHAT_MODELS

        if model_name not in OPEN_AI_CHAT_MODELS:
            raise ValueError(f"Unknown chat model '{model_name}'")

        return cls(
            model=OPEN_AI_CHAT_MODELS[model_name], messages=list(messages), **kwargs
        )

    @property
    def token_length(self) -> int:
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
============== {__class__.__name__} ==============
Length: {self.token_length} tokens; {len(self.messages)} messages
{formatted_messages}
==========================================
"""


class LLMResponse(BaseModel):
    """Standard response struct for a response from an LLM model."""

    model_info: ModelInfo


class EmbeddingModelResponse(LLMResponse):
    """Standard response struct for a response from an embedding model."""

    embedding: list[float] = Field(default_factory=list)


class ChatModelResponse(LLMResponse):
    """Standard response struct for a response from a chat LLM."""

    content: Optional[str]
    function_call: Optional[LLMFunctionCall]


class LLMFunctionCall(BaseModel):
    """Represents a function call as generated by an OpenAI model

    Attributes:
        name: the name of the function that the LLM wants to call
        arguments: a stringified JSON object (unverified) containing `arg: value` pairs
    """

    name: str
    arguments: dict[str, Any] = {}

    @staticmethod
    def parse(raw: FunctionCallDict):
        return LLMFunctionCall(
            name=raw["name"],
            arguments=json.loads(raw["arguments"]),
        )


# Complete model initialization; necessary because of order of definition
ChatModelResponse.update_forward_refs()
