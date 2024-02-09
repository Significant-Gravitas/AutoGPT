from __future__ import annotations
import abc
import enum
import os
import functools
import time
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
    ParamSpec,
)
from typing_extensions import TypedDict

from pydantic import BaseModel, Field

from langchain_core.language_models.chat_models import BaseChatModel

from AFAAS.interfaces.adapters.language_model import (
    AbstractLanguageModelProvider,
    BaseModelInfo,
    BaseModelResponse,
    ModelProviderService,
    AbstractPromptConfiguration,
)
from AFAAS.lib.utils.json_schema import JSONSchema


from openai import APIError, AsyncOpenAI, RateLimitError
from openai.resources import AsyncCompletions
from AFAAS.lib.sdk.logger import AFAASLogger
LOG = AFAASLogger(name=__name__)
aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

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


class ChatModelWrapper:

    llm_adapter : AbstractChatModelProvider


    def __init__(self, llm_model: BaseChatModel) -> None:

        self.llm_adapter = llm_model

        self.retry_per_request = llm_model._settings.configuration.retries_per_request
        self.maximum_retry = llm_model._settings.configuration.maximum_retry
        self.maximum_retry_before_default_function = llm_model._settings.configuration.maximum_retry_before_default_function

        retry_handler = _RetryHandler(
            num_retries=self.retry_per_request,
        )
        self._create_chat_completion = retry_handler(self.chat)
        self._func_call_fails_count = 0


    async def create_chat_completion(
        self,
        chat_messages: list[ChatMessage],
        tools: list[CompletionModelFunction],
        llm_model_name: str,
        tool_choice: str,
        default_tool_choice: str,  # This one would be called after 3 failed attemps(cf : try/catch block)
        completion_parser: Callable[[AssistantChatMessageDict], _T] = lambda _: None,
        **kwargs,
    ) -> AbstractChatModelResponse[_T]:
        # ##############################################################################
        # ### Step 1: Prepare arguments for API call
        # ##############################################################################
        completion_kwargs = self.llm_adapter._initialize_completion_args(
            model_name=llm_model_name,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        # ##############################################################################
        # ### Step 2: Execute main chat completion and extract details
        # ##############################################################################
        response = await self._get_chat_response(
            model_prompt=chat_messages, **completion_kwargs
        )
        response_message, response_args = self.llm_adapter._extract_response_details(
            response=response, model_name=llm_model_name
        )

        # ##############################################################################
        # ### Step 3: Handle missing function call and retry if necessary
        # ##############################################################################
        if self.llm_adapter._should_retry_function_call(
            tools=tools, response_message=response_message
        ):
            if (
                self._func_call_fails_count
                <= self.maximum_retry
            ):
                return await self._retry_chat_completion(
                    model_prompt=chat_messages,
                    tools=tools,
                    completion_kwargs=completion_kwargs,
                    model_name=llm_model_name,
                    completion_parser=completion_parser,
                    default_tool_choice=default_tool_choice,
                    response=response,
                    response_args=response_args,
                )

            # FIXME, TODO, NOTE: Organize application save feedback loop to improve the prompts, as it is not normal that function are not called
            response_message["tool_calls"] = None
            response.choices[0].message["tool_calls"] = None
            # self._handle_failed_retry(response_message)

        # ##############################################################################
        # ### Step 4: Reset failure count and integrate improvements
        # ##############################################################################
        self._func_call_fails_count = 0

        # ##############################################################################
        # ### Step 5: Self feedback
        # ##############################################################################

        # Create an option to deactivate feedbacks
        # Option : Maximum number of feedbacks allowed

        # Prerequisite : Read OpenAI API (Chat Model) tool_choice section

        # User : 1 shirt take 5 minutes to dry , how long take 10 shirt to dry
        # Assistant : It takes 50 minutes

        # System : "The user question was ....
        # The Assistant Response was ..."
        # Is it ok ?
        # If not provide a feedback

        # => T shirt can be dried at the same time

        # ##############################################################################
        # ### Step 6: Formulate the response
        # ##############################################################################
        return self.llm_adapter._formulate_final_response(
            response_message=response_message,
            completion_parser=completion_parser,
            response_args=response_args,
        )


    async def _retry_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        tools: list[CompletionModelFunction],
        completion_kwargs: Dict[str, Any],
        model_name: str,
        completion_parser: Callable[[AssistantChatMessageDict], _T],
        default_tool_choice: str,
        response: AsyncCompletions,
        response_args: Dict[str, Any],
    ) -> AbstractChatModelResponse[_T]:
        completion_kwargs = self._update_function_call_for_retry(
            completion_kwargs=completion_kwargs,
            default_tool_choice=default_tool_choice,
        )
        completion_kwargs["tools"] = tools
        response.update(response_args)
        self.llm_adapter._budget.update_usage_and_cost(model_response=response)
        return await self.create_chat_completion(
            chat_messages=model_prompt,
            llm_model_name=model_name,
            completion_parser=completion_parser,
            **completion_kwargs,
        )


    def _update_function_call_for_retry(
        self, completion_kwargs: Dict[str, Any], default_tool_choice: str
    ) -> Dict[str, Any]:
        if (
            self._func_call_fails_count
            >= self.maximum_retry_before_default_function
        ):
            completion_kwargs["tool_calls"] = default_tool_choice
        else:
            completion_kwargs["tool_calls"] = completion_kwargs.get(
                "tool_calls", "auto"
            )
        completion_kwargs["default_tool_choice"] = completion_kwargs.get(
            "default_tool_choice", default_tool_choice
        )
        self._func_call_fails_count += 1
        return completion_kwargs


    async def _get_chat_response(
        self, model_prompt: list[ChatMessage], **completion_kwargs: Any
    ) -> AsyncCompletions:
        return await self._create_chat_completion(
            messages=model_prompt, **completion_kwargs
        )


    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: str,
    ) -> int: 
        return self.llm_adapter.count_message_tokens(messages, model_name)



    async def chat(
        self, messages: list[ChatMessage], *_, **kwargs
    ) -> AsyncCompletions:

        raw_messages = [
            message.dict(include={"role", "content", "tool_calls", "name"})
            for message in messages
        ]

        llm_kwargs = self.llm_adapter.make_chat_kwargs(**kwargs)
        LOG.trace(raw_messages[0]["content"])
        LOG.trace(llm_kwargs)
        return_value = await self.llm_adapter.chat(
            messages=raw_messages, **llm_kwargs
        )

        return return_value

    def has_oa_tool_calls_api(self, model_name: str) -> bool:
        self.llm_adapter.has_oa_tool_calls_api(model_name)

    def get_default_config(self) -> AbstractPromptConfiguration:
        return self.llm_adapter.get_default_config()



class AbstractChatModelProvider(AbstractLanguageModelProvider): 

    class CompletionKwargs(BaseModel):
        llm_model_name: str
        tools: Optional[list[CompletionModelFunction]] = None
        tool_choice: Optional[str] = None
        default_tool_choice: Optional[str] = None
        completion_parser: Callable[[AssistantChatMessageDict], _T]

    llm_model : Optional[BaseChatModel] = None

    @abc.abstractmethod
    def make_chat_kwargs(self, **kwargs) -> dict:
        ...

    @abc.abstractmethod
    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: str,
    ) -> int: ...

    @abc.abstractmethod
    async def chat(
        self, messages: list[ChatMessage], *_, **llm_kwargs
    ) -> AsyncCompletions:
        ...

    @abc.abstractmethod
    def make_tool_choice(self , name : str) -> dict:
        ... 

    @abc.abstractmethod
    def has_oa_tool_calls_api(self, model_name: str) -> bool:
        ...

    @abc.abstractmethod
    def get_default_config(self) -> AbstractPromptConfiguration:
        ...


    @abc.abstractmethod
    def _extract_response_details(
        self, response: AsyncCompletions, model_name: str
    ) -> tuple[dict, dict]: 
        ...

    @abc.abstractmethod
    def _should_retry_function_call(
        self, tools: list[CompletionModelFunction], response_message: dict
    ) -> bool: 
        ...

    @abc.abstractmethod
    def _formulate_final_response(
        self,
        response_message: dict,
        completion_parser: Callable[[AssistantChatMessageDict], _T],
        response_args: dict,
    ) -> AbstractChatModelResponse[_T]: 
        ...

    @abc.abstractmethod
    def _initialize_completion_args(
        self,
        model_name: str,
        tools: list[CompletionModelFunction],
        tool_choice: str,
        **kwargs,
    ) -> dict: 
        ...


_P = ParamSpec("_P")

class _RetryHandler:
    """Retry Handler for OpenAI API call.

    Args:
        num_retries int: Number of retries. Defaults to 10.
        backoff_base float: Base for exponential backoff. Defaults to 2.
        warn_user bool: Whether to warn the user. Defaults to True.
    """

    _retry_limit_msg = "Error: Reached rate limit, passing..."
    _api_key_error_msg = (
        "Please double check that you have setup a PAID OpenAI API Account. You can "
        "read more here: https://docs.agpt.co/setup/#getting-an-api-key"
    )
    _backoff_msg = "Error: API Bad gateway. Waiting {backoff} seconds..."

    def __init__(
        self,
        num_retries: int = 10,
        backoff_base: float = 2.0,
        warn_user: bool = True,
    ):
        self._num_retries = num_retries
        self._backoff_base = backoff_base
        self._warn_user = warn_user

    def _log_rate_limit_error(self) -> None:
        LOG.trace(self._retry_limit_msg)
        if self._warn_user:
            LOG.warning(self._api_key_error_msg)
            self._warn_user = False

    def _backoff(self, attempt: int) -> None:
        backoff = self._backoff_base ** (attempt + 2)
        LOG.trace(self._backoff_msg.format(backoff=backoff))
        time.sleep(backoff)

    def __call__(self, func: Callable[_P, _T]) -> Callable[_P, _T]:
        @functools.wraps(func)
        async def _wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _T:
            num_attempts = self._num_retries + 1  # +1 for the first attempt
            for attempt in range(1, num_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except RateLimitError:
                    if attempt == num_attempts:
                        raise
                    self._log_rate_limit_error()

                except APIError as e:
                    if (e.code != 502) or (attempt == num_attempts):
                        raise
                except Exception as e:
                    LOG.warning(e)
                self._backoff(attempt)

        return _wrapped
