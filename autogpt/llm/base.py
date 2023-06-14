from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import ceil, floor
from typing import Any, List, Literal, Optional, TypedDict, Union


class MessageType(Enum):
    AI_RESPONSE = "ai_response"
    AI_FUNCTION_CALL = "ai_function_call"
    ACTION_RESULT = "action_result"


class MessageRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


TText = list[int]
"""Token array representing tokenized text"""


class MessageDict(TypedDict):
    role: str
    content: Optional[str]
    name: Optional[str]

    function_call: Optional[dict]


class MessageCycle:
    user_input: Message
    ai_response: Message
    ai_function_response: Message
    result: Message

    def __init__(
        self,
        user_input: Message,
        ai_response: Message,
        triggering_prompt: Message,
        result: Message,
        ai_function_response: Optional[Message] = None,
    ):
        self.user_input = user_input
        self.triggering_prompt = user_input
        self.ai_response = ai_response
        self.ai_function_response = ai_function_response
        self.result = result

    @property
    def messages(self) -> list[Message]:
        messages = [
            self.user_input,
            self.ai_response,
            self.result,
        ]
        if self.ai_function_response:
            messages.append(self.ai_function_response)

        return messages

    @classmethod
    def construct(
        cls,
        triggering_prompt: str,
        ai_response: str,
        user_input: Optional[str] = None,
        command_result: Optional[str] = None,
        command_name: Optional[str] = None,
    ) -> MessageCycle:
        return cls(
            triggering_prompt=Message(role=MessageRole.USER, content=triggering_prompt),
            ai_response=Message(role=MessageRole.USER, content=ai_response),
            user_input=Message(role=MessageRole.USER, content=user_input)
            if user_input
            else None,
            result=Message(
                role=MessageRole.FUNCTION,
                content=command_result,
                function_name=command_name,
            )
            if command_result
            else None,
        )


class Message:
    """OpenAI Message object containing a role and the message content"""

    def __init__(
        self,
        role: Union[MessageRole, str],
        content: Optional[str] = None,
        function_name: Optional[str] = None,
        function_arguments: Optional[str] = None,
        message_type: Optional[MessageType] = None,
    ):
        if type(role) == MessageRole:
            self.role = role.value
        else:
            self.role: str = role

        self.function_name = function_name
        # This is a JSON-encoded string
        self.function_arguments = function_arguments
        self.message_type = message_type

        # Docs say:
        #   content string Optional
        #   The contents of the message. content is required for all messages except assistant messages
        #   with function calls.
        # This is an assistant message with a function call, but it will error out if we don't include content.
        if not content:
            if function_name and self.role == MessageRole.ASSISTANT.value:
                self.content = ""
            else:
                raise ValueError(
                    "Content is required for all messages except assistant messages with function calls."
                )
        self.content = content

    def raw(self) -> MessageDict:
        raw_message = {"role": self.role}

        if self.content:
            raw_message["content"] = self.content

        if self.role == MessageRole.FUNCTION.value:
            raw_message["name"] = self.function_name

        if self.function_name:
            raw_message["function_call"] = {
                "name": self.function_name,
                "arguments": self.function_arguments or "{}",
            }

        return raw_message


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
