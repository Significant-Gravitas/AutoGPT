from __future__ import annotations
import abc
import functools
import time
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Optional,
    TypeVar,
    ParamSpec,
    Union,
)
from pydantic import BaseModel, Field, ConfigDict

from langchain_core.language_models.chat_models import BaseChatModel

from openai import AsyncOpenAI
from autogpt.interfaces.adapters.language_model import (
    AbstractLanguageModelProvider,
    LanguageModelInfo,
    LanguageModelResponse,
    ModelProviderService,
    AbstractPromptConfiguration,
)
from autogpt.core.utils.json_schema import JSONSchema


from openai import APIError, RateLimitError
from openai.resources import AsyncCompletions
import logging
LOG = logging.getLogger(__name__)

from langchain_core.messages import ChatMessage
from autogpt.interfaces.adapters.chatmodel.chatmessage import AssistantChatMessage


class CompletionModelFunction(BaseModel):
    name: str
    description: str
    parameters: Dict[str, JSONSchema]

    def schema(self , schema_builder = Callable) -> dict[str, str | dict | list]:
        return schema_builder(self)

    def _remove_none_entries(self, data: Dict[str, Any]) -> Dict[str, Any]:
        cleaned_data = {}
        for key, value in data.items():
            if value is not None:
                if isinstance(value, dict):
                    cleaned_data[key] = self._remove_none_entries(value)
                else:
                    cleaned_data[key] = value
        return cleaned_data

    def model_dump(self, *args, **kwargs):
        # Call the parent class's model_dump() method to get the original dictionary
        data = super().model_dump(*args, **kwargs)

        # Remove entries with None values recursively
        cleaned_data = self._remove_none_entries(data)

        return cleaned_data

    def fmt_line(self) -> str:
        params = ", ".join(
            f"{name}: {p.type.value}" for name, p in self.parameters.items()
        )
        return f"{self.name}: {self.description}. Params: ({params})"


class ChatPrompt(BaseModel):
    # TODO : Remove or rewrite with 2 arguments
      # messages : list[Union[ChatMessage, AbstractChatMessage]]
      # chat_completion_kwargs : ChatCompletionKwargs
    messages: list
    tools: list[CompletionModelFunction] = Field(default_factory=list)
    tool_choice: str
    default_tool_choice: str

    def raw(self) -> list:
        return [m.dict() for m in self.messages]

    def __str__(self):
        return "\n\n".join(
            [m.dict() for m in self.messages]
        )


_T = TypeVar("_T")


class AbstractChatModelResponse(LanguageModelResponse, Generic[_T]):

    response: Optional[AssistantChatMessage] = None
    parsed_result: _T = None
    """Standard response struct for a response from a language model."""

    content: dict = None
    chat_messages: list[ChatMessage] = []
    system_prompt: str = None


class ChatModelInfo(LanguageModelInfo):
    llm_service : ModelProviderService = ModelProviderService.CHAT
    max_tokens: int


class AbstractChatModelProvider(AbstractLanguageModelProvider): 

    model_config: ConfigDict = ConfigDict(
        extra= "allow",
    )

    llm_api_client : Union [BaseChatModel , AsyncOpenAI , Any]

    llmmodel_default : str
    llmmodel_default : Optional[str] = None
    llmmodel_fine_tuned : Optional[str] = None
    llmmodel_cheap : Optional[str] = None
    llmmodel_code_expert_model : Optional[str] = None
    llmmodel_long_context_model : Optional[str] = None

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
    def make_model_arg(self, model_name : str) -> dict:
        ...

    @abc.abstractmethod
    def make_tool(self, f : CompletionModelFunction) -> dict:
        ...

    @staticmethod
    @abc.abstractmethod
    def tool_builder(func: CompletionModelFunction) -> dict[str, str | dict | list]: 
        ...
    @abc.abstractmethod
    def make_tools_arg(self, tools : list[CompletionModelFunction]) -> dict:
        ...

    @abc.abstractmethod
    def make_tool_choice_arg(self , name : str) -> dict:
        ... 

    @abc.abstractmethod
    def get_default_config(self) -> AbstractPromptConfiguration:
        ...

    @abc.abstractmethod
    def extract_response_details(
        self, response: AsyncCompletions, model_name: str
    ) -> BaseModel:
        ...

    @abc.abstractmethod
    def should_retry_function_call(
        self, tools: list[CompletionModelFunction], response_message: dict
    ) -> bool: 
        ...

    @abc.abstractmethod
    def formulate_final_response(
        self,
        response_message: dict,
        completion_parser: Callable[[AssistantChatMessage], _T],
    ) -> AbstractChatModelResponse[_T]: 
        ...

    def __getattribute__(self, __name: str):

        if not __name.startswith("__llmmodel_"):
            return super().__getattribute__(__name)

        try:
            model_name = super().__getattribute__(__name)
            if model_name is not None:
                return model_name
        except AttributeError:
            LOG.warning(f"Model name {__name} not found in {self.__class__.__name__} , defaulting to {self.llmmodel_default}")

        return self.__llmmodel_default__()

    def __llmmodel_default__(self) -> str:
        return self.llmmodel_default

    def __llmmodel_cheap__(self) -> str:
        return self.llmmodel_cheap

    def __llmmodel_code_expert_model__(self) -> str:
        return self.llmmodel_code_expert_model

    def __llmmodel_long_context_model__(self) -> str:
        return self.llmmodel_long_context_model

    def __llmmodel_fine_tuned__(self) -> str:
        return self.llmmodel_fine_tuned


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
