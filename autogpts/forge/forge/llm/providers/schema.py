import abc
import enum
import logging
import math
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
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

from forge.config.schema import SystemConfiguration, UserConfigurable
from forge.json.schema import JSONSchema
from forge.llm.schema import (
    Embedding,
    ProviderBudget,
    ProviderCredentials,
    ProviderSettings,
    ProviderUsage,
    ResourceType,
)
from forge.logging.utils import fmt_kwargs

if TYPE_CHECKING:
    from jsonschema import ValidationError


class ModelProviderService(str, enum.Enum):
    """A ModelService describes what kind of service the model provides."""

    EMBEDDING = "embedding"
    CHAT = "chat_completion"
    TEXT = "text_completion"


class ModelProviderName(str, enum.Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


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

    def __str__(self) -> str:
        return f"{self.name}({fmt_kwargs(self.arguments)})"


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
    role: Literal[ChatMessage.Role.ASSISTANT] = ChatMessage.Role.ASSISTANT
    content: Optional[str]
    tool_calls: Optional[list[AssistantToolCall]] = None


class ToolResultMessage(ChatMessage):
    role: Literal[ChatMessage.Role.TOOL] = ChatMessage.Role.TOOL
    is_error: bool = False
    tool_call_id: str


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

    def validate_call(
        self, function_call: AssistantFunctionCall
    ) -> tuple[bool, list["ValidationError"]]:
        """
        Validates the given function call against the function's parameter specs

        Returns:
            bool: Whether the given set of arguments is valid for this command
            list[ValidationError]: Issues with the set of arguments (if any)

        Raises:
            ValueError: If the function_call doesn't call this function
        """
        if function_call.name != self.name:
            raise ValueError(
                f"Can't validate {function_call.name} call using {self.name} spec"
            )

        params_schema = JSONSchema(
            type=JSONSchema.Type.OBJECT,
            properties={name: spec for name, spec in self.parameters.items()},
        )
        return params_schema.validate_object(function_call.arguments)


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

    def update_usage(
        self,
        input_tokens_used: int,
        output_tokens_used: int = 0,
    ) -> None:
        self.prompt_tokens += input_tokens_used
        self.completion_tokens += output_tokens_used


class ModelProviderBudget(ProviderBudget):
    usage: defaultdict[str, ModelProviderUsage] = defaultdict(ModelProviderUsage)

    def update_usage_and_cost(
        self,
        model_info: ModelInfo,
        input_tokens_used: int,
        output_tokens_used: int = 0,
    ) -> float:
        """Update the usage and cost of the provider.

        Returns:
            float: The (calculated) cost of the given model response.
        """
        self.usage[model_info.name].update_usage(input_tokens_used, output_tokens_used)
        incurred_cost = (
            output_tokens_used * model_info.completion_token_cost
            + input_tokens_used * model_info.prompt_token_cost
        )
        self.total_cost += incurred_cost
        self.remaining_budget -= incurred_cost
        return incurred_cost


class ModelProviderSettings(ProviderSettings):
    resource_type: ResourceType = ResourceType.MODEL
    configuration: ModelProviderConfiguration
    credentials: Optional[ModelProviderCredentials] = None
    budget: Optional[ModelProviderBudget] = None


class ModelProvider(abc.ABC):
    """A ModelProvider abstracts the details of a particular provider of models."""

    default_settings: ClassVar[ModelProviderSettings]

    _settings: ModelProviderSettings
    _configuration: ModelProviderConfiguration
    _credentials: Optional[ModelProviderCredentials] = None
    _budget: Optional[ModelProviderBudget] = None

    _logger: logging.Logger

    def __init__(
        self,
        settings: Optional[ModelProviderSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not settings:
            settings = self.default_settings.copy(deep=True)

        self._settings = settings
        self._configuration = settings.configuration
        self._credentials = settings.credentials
        self._budget = settings.budget

        self._logger = logger or logging.getLogger(self.__module__)

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

    service: Literal[ModelProviderService.EMBEDDING] = ModelProviderService.EMBEDDING
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

    service: Literal[ModelProviderService.CHAT] = ModelProviderService.CHAT
    max_tokens: int
    has_function_call_api: bool = False


_T = TypeVar("_T")


class ChatModelResponse(ModelResponse, Generic[_T]):
    """Standard response struct for a response from a language model."""

    response: AssistantChatMessage
    parsed_result: _T = None


class ChatModelProvider(ModelProvider):
    @abc.abstractmethod
    async def get_available_models(self) -> list[ChatModelInfo]:
        ...

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
        max_output_tokens: Optional[int] = None,
        prefill_response: str = "",
        **kwargs,
    ) -> ChatModelResponse[_T]:
        ...
