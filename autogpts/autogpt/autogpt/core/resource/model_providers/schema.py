import abc
import enum
import math
from typing import (
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
)

from pydantic import BaseModel, Field, SecretStr, validator

from autogpt.core.configuration import SystemConfiguration, UserConfigurable
from autogpt.core.resource.schema import (
    Embedding,
    ProviderBudget,
    ProviderCredentials,
    ProviderSettings,
    ProviderUsage,
    ResourceType,
)
from autogpt.core.utils.json_schema import JSONSchema


class ModelProviderService(str, enum.Enum):
    """A ModelService describes what kind of service the model provides."""

    EMBEDDING = "embedding"
    CHAT = "chat_completion"
    TEXT = "text_completion"


class ModelProviderName(str, enum.Enum):
    OPENAI = "openai"
    LLAMAFILE = "llamafile"


class ChatMessage(BaseModel):
    class Role(str, enum.Enum):
        USER = "user"
        SYSTEM = "system"
        ASSISTANT = "assistant"

        TOOL = "tool"
        """May be used for the result of tool calls"""
        FUNCTION = "function"
        """May be used for the return value of function calls"""

    role: Role
    content: str

    @staticmethod
    def user(content: str) -> "ChatMessage":
        return ChatMessage(role=ChatMessage.Role.USER, content=content)

    @staticmethod
    def system(content: str) -> "ChatMessage":
        return ChatMessage(role=ChatMessage.Role.SYSTEM, content=content)


class ChatMessageDict(TypedDict):
    role: str
    content: str


class AssistantFunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class AssistantFunctionCallDict(TypedDict):
    name: str
    arguments: dict[str, Any]


class AssistantToolCall(BaseModel):
    id: str
    type: Literal["function"]
    function: AssistantFunctionCall


class AssistantToolCallDict(TypedDict):
    id: str
    type: Literal["function"]
    function: AssistantFunctionCallDict


class AssistantChatMessage(ChatMessage):
    role: Literal["assistant"] = "assistant"
    content: Optional[str]
    tool_calls: Optional[list[AssistantToolCall]] = None


class AssistantChatMessageDict(TypedDict, total=False):
    role: str
    content: str
    tool_calls: list[AssistantToolCallDict]


class CompletionModelFunction(BaseModel):
    """General representation object for LLM-callable functions."""

    name: str
    description: str
    parameters: dict[str, "JSONSchema"]

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

    def fmt_line(self) -> str:
        params = ", ".join(
            f"{name}{'?' if not p.required else ''}: " f"{p.typescript_type}"
            for name, p in self.parameters.items()
        )
        return f"{self.name}: {self.description}. Params: ({params})"


class ModelInfo(BaseModel):
    """Struct for model information.

    Would be lovely to eventually get this directly from APIs, but needs to be
    scraped from websites for now.
    """

    name: str
    service: ModelProviderService
    provider_name: ModelProviderName
    prompt_token_cost: float = 0.0
    completion_token_cost: float = 0.0


class ModelResponse(BaseModel):
    """Standard response struct for a response from a model."""

    prompt_tokens_used: int
    completion_tokens_used: int
    model_info: ModelInfo


class ModelProviderConfiguration(SystemConfiguration):
    retries_per_request: int = UserConfigurable()
    extra_request_headers: dict[str, str] = Field(default_factory=dict)


class ModelProviderCredentials(ProviderCredentials):
    """Credentials for a model provider."""

    api_key: SecretStr | None = UserConfigurable(default=None)
    api_type: SecretStr | None = UserConfigurable(default=None)
    api_base: SecretStr | None = UserConfigurable(default=None)
    api_version: SecretStr | None = UserConfigurable(default=None)
    deployment_id: SecretStr | None = UserConfigurable(default=None)

    class Config:
        extra = "ignore"


class ModelProviderUsage(ProviderUsage):
    """Usage for a particular model from a model provider."""

    completion_tokens: int = 0
    prompt_tokens: int = 0
    total_tokens: int = 0

    def update_usage(
        self,
        model_response: ModelResponse,
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
        model_response: ModelResponse,
    ) -> float:
        """Update the usage and cost of the provider.

        Returns:
            float: The (calculated) cost of the given model response.
        """
        model_info = model_response.model_info
        self.usage.update_usage(model_response)
        incurred_cost = (
            model_response.completion_tokens_used * model_info.completion_token_cost
            + model_response.prompt_tokens_used * model_info.prompt_token_cost
        )
        self.total_cost += incurred_cost
        self.remaining_budget -= incurred_cost
        return incurred_cost


class ModelProviderSettings(ProviderSettings):
    resource_type: ResourceType = ResourceType.MODEL
    configuration: ModelProviderConfiguration
    credentials: ModelProviderCredentials
    budget: ModelProviderBudget


class ModelProvider(abc.ABC):
    """A ModelProvider abstracts the details of a particular provider of models."""

    default_settings: ClassVar[ModelProviderSettings]

    _budget: Optional[ModelProviderBudget]
    _configuration: ModelProviderConfiguration

    @abc.abstractmethod
    def count_tokens(self, text: str, model_name: str) -> int:
        ...

    @abc.abstractmethod
    def get_tokenizer(self, model_name: str) -> "ModelTokenizer":
        ...

    @abc.abstractmethod
    def get_token_limit(self, model_name: str) -> int:
        ...

    def get_incurred_cost(self) -> float:
        if self._budget:
            return self._budget.total_cost
        return 0

    def get_remaining_budget(self) -> float:
        if self._budget:
            return self._budget.remaining_budget
        return math.inf


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


class EmbeddingModelInfo(ModelInfo):
    """Struct for embedding model information."""

    llm_service = ModelProviderService.EMBEDDING
    max_tokens: int
    embedding_dimensions: int


class EmbeddingModelResponse(ModelResponse):
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
    ) -> EmbeddingModelResponse:
        ...


###############
# Chat Models #
###############


class ChatModelInfo(ModelInfo):
    """Struct for language model information."""

    llm_service = ModelProviderService.CHAT
    max_tokens: int
    has_function_call_api: bool = False


_T = TypeVar("_T")


class ChatModelResponse(ModelResponse, Generic[_T]):
    """Standard response struct for a response from a language model."""

    response: AssistantChatMessage
    parsed_result: _T = None


class ChatModelProvider(ModelProvider):
    @abc.abstractmethod
    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: str,
    ) -> int:
        ...

    @abc.abstractmethod
    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: str,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        **kwargs,
    ) -> ChatModelResponse[_T]:
        ...
