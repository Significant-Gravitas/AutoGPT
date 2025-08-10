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
    Sequence,
    TypedDict,
    TypeVar,
)

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from forge.logging.utils import fmt_kwargs
from forge.models.config import (
    Configurable,
    SystemConfiguration,
    SystemSettings,
    UserConfigurable,
)
from forge.models.json_schema import JSONSchema
from forge.models.providers import (
    Embedding,
    ProviderBudget,
    ProviderCredentials,
    ResourceType,
)

if TYPE_CHECKING:
    from jsonschema import ValidationError


_T = TypeVar("_T")

_ModelName = TypeVar("_ModelName", bound=str)


class ModelProviderService(str, enum.Enum):
    """A ModelService describes what kind of service the model provides."""

    EMBEDDING = "embedding"
    CHAT = "chat_completion"
    TEXT = "text_completion"


class ModelProviderName(str, enum.Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
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
    role: Literal[ChatMessage.Role.ASSISTANT] = ChatMessage.Role.ASSISTANT  # type: ignore # noqa
    content: str = ""
    tool_calls: Optional[list[AssistantToolCall]] = None


class ToolResultMessage(ChatMessage):
    role: Literal[ChatMessage.Role.TOOL] = ChatMessage.Role.TOOL  # type: ignore
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


class ModelInfo(BaseModel, Generic[_ModelName]):
    """Struct for model information.

    Would be lovely to eventually get this directly from APIs, but needs to be
    scraped from websites for now.
    """

    name: _ModelName
    service: ClassVar[ModelProviderService]
    provider_name: ModelProviderName
    prompt_token_cost: float = 0.0
    completion_token_cost: float = 0.0


class ModelResponse(BaseModel):
    """Standard response struct for a response from a model."""

    prompt_tokens_used: int
    completion_tokens_used: int
    llm_info: ModelInfo


class ModelProviderConfiguration(SystemConfiguration):
    retries_per_request: int = UserConfigurable(7)
    fix_failed_parse_tries: int = UserConfigurable(3)
    extra_request_headers: dict[str, str] = Field(default_factory=dict)


class ModelProviderCredentials(ProviderCredentials):
    """Credentials for a model provider."""

    api_key: SecretStr | None = UserConfigurable(default=None)
    api_type: SecretStr | None = UserConfigurable(default=None)
    api_base: SecretStr | None = UserConfigurable(default=None)
    api_version: SecretStr | None = UserConfigurable(default=None)
    deployment_id: SecretStr | None = UserConfigurable(default=None)

    model_config = ConfigDict(extra="ignore")


class ModelProviderUsage(BaseModel):
    """Usage for a particular model from a model provider."""

    class ModelUsage(BaseModel):
        completion_tokens: int = 0
        prompt_tokens: int = 0

    usage_per_model: dict[str, ModelUsage] = defaultdict(ModelUsage)

    @property
    def completion_tokens(self) -> int:
        return sum(model.completion_tokens for model in self.usage_per_model.values())

    @property
    def prompt_tokens(self) -> int:
        return sum(model.prompt_tokens for model in self.usage_per_model.values())

    def update_usage(
        self,
        model: str,
        input_tokens_used: int,
        output_tokens_used: int = 0,
    ) -> None:
        self.usage_per_model[model].prompt_tokens += input_tokens_used
        self.usage_per_model[model].completion_tokens += output_tokens_used


class ModelProviderBudget(ProviderBudget[ModelProviderUsage]):
    usage: ModelProviderUsage = Field(default_factory=ModelProviderUsage)

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
        self.usage.update_usage(model_info.name, input_tokens_used, output_tokens_used)
        incurred_cost = (
            output_tokens_used * model_info.completion_token_cost
            + input_tokens_used * model_info.prompt_token_cost
        )
        self.total_cost += incurred_cost
        self.remaining_budget -= incurred_cost
        return incurred_cost


class ModelProviderSettings(SystemSettings):
    resource_type: ClassVar[ResourceType] = ResourceType.MODEL
    configuration: ModelProviderConfiguration
    credentials: Optional[ModelProviderCredentials] = None
    budget: Optional[ModelProviderBudget] = None


_ModelProviderSettings = TypeVar("_ModelProviderSettings", bound=ModelProviderSettings)


# TODO: either use MultiProvider throughout codebase as type for `llm_provider`, or
# replace `_ModelName` by `str` to eliminate type checking difficulties
class BaseModelProvider(
    abc.ABC,
    Generic[_ModelName, _ModelProviderSettings],
    Configurable[_ModelProviderSettings],
):
    """A ModelProvider abstracts the details of a particular provider of models."""

    default_settings: ClassVar[_ModelProviderSettings]  # type: ignore

    _settings: _ModelProviderSettings
    _logger: logging.Logger

    def __init__(
        self,
        settings: Optional[_ModelProviderSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not settings:
            settings = self.default_settings.model_copy(deep=True)

        self._settings = settings
        self._configuration = settings.configuration
        self._credentials = settings.credentials
        self._budget = settings.budget

        self._logger = logger or logging.getLogger(self.__module__)

    @abc.abstractmethod
    async def get_available_models(
        self,
    ) -> Sequence["ChatModelInfo[_ModelName] | EmbeddingModelInfo[_ModelName]"]:
        ...

    @abc.abstractmethod
    def count_tokens(self, text: str, model_name: _ModelName) -> int:
        ...

    @abc.abstractmethod
    def get_tokenizer(self, model_name: _ModelName) -> "ModelTokenizer[Any]":
        ...

    @abc.abstractmethod
    def get_token_limit(self, model_name: _ModelName) -> int:
        ...

    def get_incurred_cost(self) -> float:
        if self._budget:
            return self._budget.total_cost
        return 0

    def get_remaining_budget(self) -> float:
        if self._budget:
            return self._budget.remaining_budget
        return math.inf


class ModelTokenizer(Protocol, Generic[_T]):
    """A ModelTokenizer provides tokenization specific to a model."""

    @abc.abstractmethod
    def encode(self, text: str) -> list[_T]:
        ...

    @abc.abstractmethod
    def decode(self, tokens: list[_T]) -> str:
        ...


####################
# Embedding Models #
####################


class EmbeddingModelInfo(ModelInfo[_ModelName]):
    """Struct for embedding model information."""

    service: Literal[ModelProviderService.EMBEDDING] = ModelProviderService.EMBEDDING  # type: ignore # noqa
    max_tokens: int
    embedding_dimensions: int


class EmbeddingModelResponse(ModelResponse):
    """Standard response struct for a response from an embedding model."""

    embedding: Embedding = Field(default_factory=list)
    completion_tokens_used: int = Field(default=0, frozen=True)


class BaseEmbeddingModelProvider(BaseModelProvider[_ModelName, _ModelProviderSettings]):
    @abc.abstractmethod
    async def get_available_embedding_models(
        self,
    ) -> Sequence[EmbeddingModelInfo[_ModelName]]:
        ...

    @abc.abstractmethod
    async def create_embedding(
        self,
        text: str,
        model_name: _ModelName,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelResponse:
        ...


###############
# Chat Models #
###############


class ChatModelInfo(ModelInfo[_ModelName]):
    """Struct for language model information."""

    service: Literal[ModelProviderService.CHAT] = ModelProviderService.CHAT  # type: ignore # noqa
    max_tokens: int
    has_function_call_api: bool = False


class ChatModelResponse(ModelResponse, Generic[_T]):
    """Standard response struct for a response from a language model."""

    response: AssistantChatMessage
    parsed_result: _T


class BaseChatModelProvider(BaseModelProvider[_ModelName, _ModelProviderSettings]):
    @abc.abstractmethod
    async def get_available_chat_models(self) -> Sequence[ChatModelInfo[_ModelName]]:
        ...

    @abc.abstractmethod
    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: _ModelName,
    ) -> int:
        ...

    @abc.abstractmethod
    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: _ModelName,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        prefill_response: str = "",
        **kwargs,
    ) -> ChatModelResponse[_T]:
        ...
