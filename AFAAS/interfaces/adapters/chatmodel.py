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
from AFAAS.interfaces.adapters.chatmessage import AssistantChatMessage

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

from langchain_core.messages import ChatMessage, HumanMessage, SystemMessage, AIMessage


class CompletionModelFunction(BaseModel):
    name: str
    description: str
    parameters: Dict[str, JSONSchema]

    @property
    def schema(self) -> dict[str, str | dict | list]:
        """Returns an OpenAI-consumable function specification"""
        #FIXME : This is OpenAI specific & should be moved to the OpenAI adapter or implemented via dependency injection

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
        #FIXME : This is OpenAI specific & should be moved to the OpenAI adapter or implemented via dependency injection
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


class AbstractChatModelResponse(BaseModelResponse, Generic[_T]):

    response: Optional[AssistantChatMessage] = None
    parsed_result: _T = None
    """Standard response struct for a response from a language model."""

    content: dict = None
    chat_messages: list[ChatMessage] = []
    system_prompt: str = None


class ChatModelInfo(BaseModelInfo):
    llm_service : ModelProviderService = ModelProviderService.CHAT
    max_tokens: int
    has_function_call_api: bool = False


class ChatCompletionKwargs(BaseModel):
    llm_model_name: str
    """The name of the language model"""
    tools: Optional[list[CompletionModelFunction]] = None
    """List of available tools"""
    tool_choice: Optional[str] = None
    """Force the use of one tool"""
    default_tool_choice: Optional[str] = None
    """This tool would be called after 3 failed attemps(cf : try/catch block)"""

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
        #self._create_chat_completion = retry_handler(self._chat)
        #FIXME: Remove beforce commit
        self._create_chat_completion = self._chat
        self._func_call_fails_count = 0


    async def create_chat_completion(
        self,
        chat_messages: list[ChatMessage],
        completion_kwargs: ChatCompletionKwargs,
        completion_parser: Callable[[AssistantChatMessage], _T], 
        # Function to parse the response, usualy injectect by an AbstractPromptStrategy
        **kwargs,
    ) -> AbstractChatModelResponse[_T]:
        if isinstance(chat_messages, ChatMessage):
            chat_messages = [chat_messages]
        elif not isinstance(chat_messages, list):
            raise TypeError(
                f"Expected ChatMessage or list[ChatMessage], but got {type(chat_messages)}"
            )

        # ##############################################################################
        # ### Prepare arguments for API call using CompletionKwargs
        # ##############################################################################
        llm_kwargs = self._make_chat_kwargs(completion_kwargs=completion_kwargs, **kwargs)

        # ##############################################################################
        # ### Step 2: Execute main chat completion and extract details
        # ##############################################################################

        response = await self._create_chat_completion(
            messages=chat_messages, 
            llm_kwargs = llm_kwargs,
            **kwargs
        )
        response_message = self.llm_adapter.extract_response_details(
            response=response, 
            model_name=completion_kwargs.llm_model_name
        )

        # ##############################################################################
        # ### Step 3: Handle missing function call and retry if necessary
        # ##############################################################################
        # FIXME : Remove before commit
        if self.llm_adapter.should_retry_function_call(
            tools=completion_kwargs.tools, response_message=response_message
        ):
            LOG.error(
                f"Attempt number {self._func_call_fails_count + 1} : Function Call was expected"
            )
            if (
                self._func_call_fails_count
                <= self.maximum_retry
            ):
                return await self._retry_chat_completion(
                    model_prompt=chat_messages,
                    completion_kwargs=completion_kwargs,
                    completion_parser=completion_parser,
                    response=response_message,
                )

            # FIXME, TODO, NOTE: Organize application save feedback loop to improve the prompts, as it is not normal that function are not called
            try : 
                response_message.additional_kwargs['tool_calls'] = None
            except Exception as e:
                response_message['tool_calls'] = None
                LOG.warning(f"Following Exception occurred : {e}")

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
        return self.llm_adapter.formulate_final_response(
            response_message=response_message,
            completion_parser=completion_parser,
        )


    async def _retry_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        completion_kwargs: ChatCompletionKwargs,
        completion_parser: Callable[[AssistantChatMessage], _T],
        response: AsyncCompletions,
        **kwargs
    ) -> AbstractChatModelResponse[_T]:
        self._func_call_fails_count += 1

        self.llm_adapter._budget.update_usage_and_cost(model_response=response.base_response)
        return await self.create_chat_completion(
            chat_messages=model_prompt,
            completion_parser=completion_parser,
            completion_kwargs= completion_kwargs,
            **kwargs
        )

    def _make_chat_kwargs(self, completion_kwargs : ChatCompletionKwargs , **kwargs) -> dict:   

        built_kwargs = {}
        built_kwargs.update(self.llm_adapter.make_model_arg(model_name=completion_kwargs.llm_model_name))

        if completion_kwargs.tools is None or len(completion_kwargs.tools) == 0:
            #if their is no tool we do nothing 
            return built_kwargs

        else:
            built_kwargs.update(self.llm_adapter.make_tools_arg(tools=completion_kwargs.tools))

            if len(completion_kwargs.tools) == 1:
                built_kwargs.update(self.llm_adapter.make_tool_choice_arg(name= completion_kwargs.tools[0].name))
                #built_kwargs.update(self.llm_adapter.make_tool_choice_arg(name= completion_kwargs.tools[0]["function"]["name"]))
            elif completion_kwargs.tool_choice!= "auto":
                if (
                    self._func_call_fails_count
                    >= self.maximum_retry_before_default_function
                ):
                    built_kwargs.update(self.llm_adapter.make_tool_choice_arg(name=completion_kwargs.default_tool_choice))
                else:
                    built_kwargs.update(self.llm_adapter.make_tool_choice_arg(name=completion_kwargs.tool_choice))
        return built_kwargs

    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: str,
    ) -> int: 
        return self.llm_adapter.count_message_tokens(messages, model_name)

    async def _chat(
        self, 
        messages: list[ChatMessage],
        llm_kwargs : dict, 
        *_, 
        **kwargs
    ) -> AsyncCompletions:

        #llm_kwargs = self._make_chat_kwargs(**kwargs)
        LOG.trace(messages[0].content)
        LOG.trace(llm_kwargs)
        return_value = await self.llm_adapter.chat(
            messages=messages, **llm_kwargs
        )

        return return_value

    def has_oa_tool_calls_api(self, model_name: str) -> bool:
        self.llm_adapter.has_oa_tool_calls_api(model_name)

    def get_default_config(self) -> AbstractPromptConfiguration:
        return self.llm_adapter.get_default_config()



class AbstractChatModelProvider(AbstractLanguageModelProvider): 

    llm_model : Optional[BaseChatModel] = None

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

    @abc.abstractmethod
    def make_tools_arg(self, tools : list[CompletionModelFunction]) -> dict:
        ...

    @abc.abstractmethod
    def make_tool_choice_arg(self , name : str) -> dict:
        ... 

    @abc.abstractmethod
    def has_oa_tool_calls_api(self, model_name: str) -> bool:
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
