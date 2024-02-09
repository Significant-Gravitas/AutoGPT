from __future__ import annotations
import abc
import enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Literal,
    Optional,
    TypeVar,
    Union,
)
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langchain_core.language_models.chat_models import BaseChatModel
from AFAAS.interfaces.adapters.language_model import (
    AbstractLanguageModelProvider,
    BaseModelInfo,
    BaseModelResponse,
    ModelProviderService,
)
from AFAAS.lib.utils.json_schema import JSONSchema


class AbstractRoleLabels(abc.ABC, BaseModel):
    USER: str
    SYSTEM: str
    ASSISTANT: str
    FUNCTION: Optional[str] = None


class AbstractChatMessage(abc.ABC, BaseModel):
    _role_labels: ClassVar[AbstractRoleLabels]
    role: str
    content: str

    @classmethod
    def assistant(cls, content: str) -> "AbstractChatMessage":
        return cls(role=cls._role_labels.ASSISTANT, content=content)

    @classmethod
    def user(cls, content: str) -> "AbstractChatMessage":
        return cls(role=cls._role_labels.USER, content=content)

    @classmethod
    def system(cls, content: str) -> "AbstractChatMessage":
        return cls(role=cls._role_labels.SYSTEM, content=content)

    def dict(self, **kwargs):
        d = super().dict(**kwargs)
        d["role"] = self.role
        return d


class Role(str, enum.Enum):
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"

    FUNCTION = "function"
    """May be used for the return value of function calls"""


class ChatMessage(BaseModel):
    role: Role
    content: str

    @staticmethod
    def assistant(content: str) -> "ChatMessage":
        return ChatMessage(role=Role.ASSISTANT, content=content)

    @staticmethod
    def user(content: str) -> "ChatMessage":
        return ChatMessage(role=Role.USER, content=content)

    @staticmethod
    def system(content: str) -> "ChatMessage":
        return ChatMessage(role=Role.SYSTEM, content=content)

    def dict(self, **kwargs):
        d = super().dict(**kwargs)
        d["role"] = self.role.value
        return d


class ChatMessageDict(TypedDict):
    role: str
    content: str


class AssistantFunctionCall(BaseModel):
    name: str
    arguments: str


class AssistantFunctionCallDict(TypedDict):
    name: str
    arguments: str


class AssistantToolCall(BaseModel):
    # id: str
    type: Literal["function"]
    function: AssistantFunctionCall


class AssistantToolCallDict(TypedDict):
    # id: str
    type: Literal["function"]
    function: AssistantFunctionCallDict


class AssistantChatMessage(ChatMessage):

    role: Role.ASSISTANT
    content: Optional[str] = None
    tool_calls: Optional[list[AssistantToolCall]] = None


class AssistantChatMessageDict(TypedDict, total=False):

    role: str
    content: Optional[str]
    tool_calls: Optional[list[AssistantToolCallDict]]


class CompletionModelFunction(BaseModel):
    name: str
    description: str
    #parameters: dict[str, "JSONSchema"]
    parameters: Dict[str, JSONSchema]

    @property
    def schema(self) -> dict[str, str | dict | list]:
        """Returns an OpenAI-consumable function specification"""

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    name: param.to_dict() for name, param in self.parameters.items()
                },
                "required": [
                    name for name, param in self.parameters.items() if param.required
                ],
            },
        }

    @staticmethod
    def parse(schema: dict) -> "CompletionModelFunction":
        return CompletionModelFunction(
            name=schema["name"],
            description=schema["description"],
            parameters=JSONSchema.parse_properties(schema["parameters"]),
        )

    def _remove_none_entries(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned_data = {}
        for key, value in data.items():
            if value is not None:
                if isinstance(value, dict):
                    cleaned_data[key] = self._remove_none_entries(value)
                else:
                    cleaned_data[key] = value
        return cleaned_data

    def dict(self, *args, **kwargs):
        # Call the parent class's dict() method to get the original dictionary
        data = super().dict(*args, **kwargs)

        # Remove entries with None values recursively
        cleaned_data = self._remove_none_entries(data)

        return cleaned_data

    def fmt_line(self) -> str:
        params = ", ".join(
            f"{name}: {p.type.value}" for name, p in self.parameters.items()
        )
        return f"{self.name}: {self.description}. Params: ({params})"


class ChatPrompt(BaseModel):
    messages: list[ChatMessage]
    tools: list[CompletionModelFunction] = Field(default_factory=list)
    tool_choice: str
    default_tool_choice: str

    def raw(self) -> list[ChatMessageDict]:
        return [m.dict() for m in self.messages]

    def __str__(self):
        return "\n\n".join(
            f"{m.role.value.upper()}: {m.content}" for m in self.messages
        )


_T = TypeVar("_T")


class AbstractChatModelResponse(BaseModelResponse, Generic[_T]):

    response: Optional[AssistantChatMessageDict] = None
    parsed_result: _T = None
    """Standard response struct for a response from a language model."""

    content: dict = None
    chat_messages: list[ChatMessage] = []
    system_prompt: str = None


class ChatModelInfo(BaseModelInfo):
    llm_service : ModelProviderService = ModelProviderService.CHAT
    max_tokens: int
    has_function_call_api: bool = False


class AbstractChatModelProvider(AbstractLanguageModelProvider): 
    llm_adapter : BaseChatModel
    @abc.abstractmethod
    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: str,
    ) -> int: ...

    async def create_language_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: str,
        completion_parser: Callable[[dict], dict],
        tools: list[CompletionModelFunction],
        tool_choice: str,
        **kwargs,
    ) -> AbstractChatModelResponse: 
        ...

    @abc.abstractmethod
    async def create_chat_completion(
        self,
        llm_model_name: str,
        chat_messages: list[ChatMessage],
        tools: list[CompletionModelFunction] = [],
        completion_parser: Callable[[AssistantChatMessageDict], _T] = lambda _: None,
        **kwargs,
    ) -> AbstractChatModelResponse[_T]: 
        ...
