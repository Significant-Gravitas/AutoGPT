from __future__ import annotations

import enum
import logging
from typing import TYPE_CHECKING, Callable, Optional, ParamSpec, TypeVar

import sentry_sdk
import tenacity
import tiktoken
from groq import APIConnectionError, APIStatusError
from pydantic import SecretStr

from autogpt.core.configuration import Configurable, UserConfigurable
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    AssistantToolCall,
    ChatMessage,
    ChatModelInfo,
    ChatModelProvider,
    ChatModelResponse,
    CompletionModelFunction,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
)
from autogpt.core.utils.json_utils import json_loads

from .utils import validate_tool_calls

if TYPE_CHECKING:
    from groq.types.chat import ChatCompletion, CompletionCreateParams
    from groq.types.chat.chat_completion import ChoiceMessage as ChatCompletionMessage
    from groq.types.chat.completion_create_params import Message as MessageParam

_T = TypeVar("_T")
_P = ParamSpec("_P")


class GroqModelName(str, enum.Enum):
    LLAMA3_8B = "llama3-8b-8192"
    LLAMA3_70B = "llama3-70b-8192"
    MIXTRAL_8X7B = "mixtral-8x7b-32768"
    GEMMA_7B = "gemma-7b-it"


GROQ_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=GroqModelName.LLAMA3_8B,
            provider_name=ModelProviderName.GROQ,
            prompt_token_cost=0.05 / 1e6,
            completion_token_cost=0.10 / 1e6,
            max_tokens=8192,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=GroqModelName.LLAMA3_70B,
            provider_name=ModelProviderName.GROQ,
            prompt_token_cost=0.59 / 1e6,
            completion_token_cost=0.79 / 1e6,
            max_tokens=8192,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=GroqModelName.MIXTRAL_8X7B,
            provider_name=ModelProviderName.GROQ,
            prompt_token_cost=0.27 / 1e6,
            completion_token_cost=0.27 / 1e6,
            max_tokens=32768,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=GroqModelName.GEMMA_7B,
            provider_name=ModelProviderName.GROQ,
            prompt_token_cost=0.10 / 1e6,
            completion_token_cost=0.10 / 1e6,
            max_tokens=8192,
            has_function_call_api=True,
        ),
    ]
}


class GroqConfiguration(ModelProviderConfiguration):
    fix_failed_parse_tries: int = UserConfigurable(3)


class GroqCredentials(ModelProviderCredentials):
    """Credentials for Groq."""

    api_key: SecretStr = UserConfigurable(from_env="GROQ_API_KEY")
    api_base: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="GROQ_API_BASE_URL"
    )

    def get_api_access_kwargs(self) -> dict[str, str]:
        return {
            k: (v.get_secret_value() if type(v) is SecretStr else v)
            for k, v in {
                "api_key": self.api_key,
                "base_url": self.api_base,
            }.items()
            if v is not None
        }


class GroqSettings(ModelProviderSettings):
    configuration: GroqConfiguration
    credentials: Optional[GroqCredentials]
    budget: ModelProviderBudget


class GroqProvider(Configurable[GroqSettings], ChatModelProvider):
    default_settings = GroqSettings(
        name="groq_provider",
        description="Provides access to Groq's API.",
        configuration=GroqConfiguration(
            retries_per_request=7,
        ),
        credentials=None,
        budget=ModelProviderBudget(),
    )

    _settings: GroqSettings
    _configuration: GroqConfiguration
    _credentials: GroqCredentials
    _budget: ModelProviderBudget

    def __init__(
        self,
        settings: Optional[GroqSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not settings:
            settings = self.default_settings.copy(deep=True)
        if not settings.credentials:
            settings.credentials = GroqCredentials.from_env()

        super(GroqProvider, self).__init__(settings=settings, logger=logger)

        from groq import AsyncGroq

        self._client = AsyncGroq(**self._credentials.get_api_access_kwargs())

    async def get_available_models(self) -> list[ChatModelInfo]:
        _models = (await self._client.models.list()).data
        return [GROQ_CHAT_MODELS[m.id] for m in _models if m.id in GROQ_CHAT_MODELS]

    def get_token_limit(self, model_name: str) -> int:
        """Get the token limit for a given model."""
        return GROQ_CHAT_MODELS[model_name].max_tokens

    @classmethod
    def get_tokenizer(cls, model_name: GroqModelName) -> ModelTokenizer:
        # HACK: No official tokenizer is available for Claude 3
        return tiktoken.encoding_for_model(model_name)

    @classmethod
    def count_tokens(cls, text: str, model_name: GroqModelName) -> int:
        return 0  # HACK: No official tokenizer is available for Claude 3

    @classmethod
    def count_message_tokens(
        cls,
        messages: ChatMessage | list[ChatMessage],
        model_name: GroqModelName,
    ) -> int:
        return 0  # HACK: No official tokenizer is available for Claude 3

    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: GroqModelName,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        prefill_response: str = "",
        **kwargs,
    ) -> ChatModelResponse[_T]:
        """Create a completion using the Groq API."""
        groq_messages, completion_kwargs = self._get_chat_completion_args(
            prompt_messages=model_prompt,
            model=model_name,
            functions=functions,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

        total_cost = 0.0
        attempts = 0
        while True:
            completion_kwargs["messages"] = groq_messages.copy()
            _response, _cost, t_input, t_output = await self._create_chat_completion(
                completion_kwargs
            )
            total_cost += _cost

            # If parsing the response fails, append the error to the prompt, and let the
            # LLM fix its mistake(s).
            attempts += 1
            parse_errors: list[Exception] = []

            _assistant_msg = _response.choices[0].message

            tool_calls, _errors = self._parse_assistant_tool_calls(_assistant_msg)
            parse_errors += _errors

            # Validate tool calls
            if not parse_errors and tool_calls and functions:
                parse_errors += validate_tool_calls(tool_calls, functions)

            assistant_msg = AssistantChatMessage(
                content=_assistant_msg.content,
                tool_calls=tool_calls or None,
            )

            parsed_result: _T = None  # type: ignore
            if not parse_errors:
                try:
                    parsed_result = completion_parser(assistant_msg)
                except Exception as e:
                    parse_errors.append(e)

            if not parse_errors:
                if attempts > 1:
                    self._logger.debug(
                        f"Total cost for {attempts} attempts: ${round(total_cost, 5)}"
                    )

                return ChatModelResponse(
                    response=AssistantChatMessage(
                        content=_assistant_msg.content,
                        tool_calls=tool_calls or None,
                    ),
                    parsed_result=parsed_result,
                    model_info=GROQ_CHAT_MODELS[model_name],
                    prompt_tokens_used=t_input,
                    completion_tokens_used=t_output,
                )

            else:
                self._logger.debug(
                    f"Parsing failed on response: '''{_assistant_msg}'''"
                )
                parse_errors_fmt = "\n\n".join(
                    f"{e.__class__.__name__}: {e}" for e in parse_errors
                )
                self._logger.warning(
                    f"Parsing attempt #{attempts} failed: {parse_errors_fmt}"
                )
                for e in parse_errors:
                    sentry_sdk.capture_exception(
                        error=e,
                        extras={"assistant_msg": _assistant_msg, "i_attempt": attempts},
                    )

                if attempts < self._configuration.fix_failed_parse_tries:
                    groq_messages.append(_assistant_msg.dict(exclude_none=True))
                    groq_messages.append(
                        {
                            "role": "system",
                            "content": (
                                f"ERROR PARSING YOUR RESPONSE:\n\n{parse_errors_fmt}"
                            ),
                        }
                    )
                    continue
                else:
                    raise parse_errors[0]

    def _get_chat_completion_args(
        self,
        prompt_messages: list[ChatMessage],
        model: GroqModelName,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs,  # type: ignore
    ) -> tuple[list[MessageParam], CompletionCreateParams]:
        """Prepare chat completion arguments and keyword arguments for API call.

        Args:
            model_prompt: List of ChatMessages.
            model_name: The model to use.
            functions: Optional list of functions available to the LLM.
            kwargs: Additional keyword arguments.

        Returns:
            list[ChatCompletionMessageParam]: Prompt messages for the OpenAI call
            dict[str, Any]: Any other kwargs for the OpenAI call
        """
        kwargs: CompletionCreateParams = kwargs  # type: ignore
        kwargs["model"] = model
        if max_output_tokens:
            kwargs["max_tokens"] = max_output_tokens

        if functions:
            kwargs["tools"] = [
                {"type": "function", "function": f.schema} for f in functions
            ]
            if len(functions) == 1:
                # force the model to call the only specified function
                kwargs["tool_choice"] = {
                    "type": "function",
                    "function": {"name": functions[0].name},
                }

        if extra_headers := self._configuration.extra_request_headers:
            # 'extra_headers' is not on CompletionCreateParams, but is on chat.create()
            kwargs["extra_headers"] = kwargs.get("extra_headers", {})  # type: ignore
            kwargs["extra_headers"].update(extra_headers.copy())  # type: ignore

        groq_messages: list[MessageParam] = [
            message.dict(
                include={"role", "content", "tool_calls", "tool_call_id", "name"},
                exclude_none=True,
            )
            for message in prompt_messages
        ]

        if "messages" in kwargs:
            groq_messages += kwargs["messages"]
            del kwargs["messages"]  # type: ignore - messages are added back later

        return groq_messages, kwargs

    async def _create_chat_completion(
        self, completion_kwargs: CompletionCreateParams
    ) -> tuple[ChatCompletion, float, int, int]:
        """
        Create a chat completion using the Groq API with retry handling.

        Params:
            completion_kwargs: Keyword arguments for an Groq Messages API call

        Returns:
            Message: The message completion object
            float: The cost ($) of this completion
            int: Number of input tokens used
            int: Number of output tokens used
        """

        @self._retry_api_request
        async def _create_chat_completion_with_retry(
            completion_kwargs: CompletionCreateParams,
        ) -> ChatCompletion:
            return await self._client.chat.completions.create(**completion_kwargs)

        response = await _create_chat_completion_with_retry(completion_kwargs)

        cost = self._budget.update_usage_and_cost(
            model_info=GROQ_CHAT_MODELS[completion_kwargs["model"]],
            input_tokens_used=response.usage.prompt_tokens,
            output_tokens_used=response.usage.completion_tokens,
        )
        return (
            response,
            cost,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    def _parse_assistant_tool_calls(
        self, assistant_message: ChatCompletionMessage, compat_mode: bool = False
    ):
        tool_calls: list[AssistantToolCall] = []
        parse_errors: list[Exception] = []

        if assistant_message.tool_calls:
            for _tc in assistant_message.tool_calls:
                try:
                    parsed_arguments = json_loads(_tc.function.arguments)
                except Exception as e:
                    err_message = (
                        f"Decoding arguments for {_tc.function.name} failed: "
                        + str(e.args[0])
                    )
                    parse_errors.append(
                        type(e)(err_message, *e.args[1:]).with_traceback(
                            e.__traceback__
                        )
                    )
                    continue

                tool_calls.append(
                    AssistantToolCall(
                        id=_tc.id,
                        type=_tc.type,
                        function=AssistantFunctionCall(
                            name=_tc.function.name,
                            arguments=parsed_arguments,
                        ),
                    )
                )

            # If parsing of all tool calls succeeds in the end, we ignore any issues
            if len(tool_calls) == len(assistant_message.tool_calls):
                parse_errors = []

        return tool_calls, parse_errors

    def _retry_api_request(self, func: Callable[_P, _T]) -> Callable[_P, _T]:
        return tenacity.retry(
            retry=(
                tenacity.retry_if_exception_type(APIConnectionError)
                | tenacity.retry_if_exception(
                    lambda e: isinstance(e, APIStatusError) and e.status_code >= 500
                )
            ),
            wait=tenacity.wait_exponential(),
            stop=tenacity.stop_after_attempt(self._configuration.retries_per_request),
            after=tenacity.after_log(self._logger, logging.DEBUG),
        )(func)

    def __repr__(self):
        return "GroqProvider()"
