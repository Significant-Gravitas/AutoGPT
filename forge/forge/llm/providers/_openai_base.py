import logging
from typing import (
    Any,
    Awaitable,
    Callable,
    ClassVar,
    Mapping,
    Optional,
    ParamSpec,
    Sequence,
    TypeVar,
    cast,
    get_args,
)

import sentry_sdk
import tenacity
from openai._exceptions import APIConnectionError, APIStatusError
from openai.types import CreateEmbeddingResponse, EmbeddingCreateParams
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    CompletionCreateParams,
)
from openai.types.shared_params import FunctionDefinition

from forge.json.parsing import json_loads

from .schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    AssistantToolCall,
    BaseChatModelProvider,
    BaseEmbeddingModelProvider,
    BaseModelProvider,
    ChatMessage,
    ChatModelInfo,
    ChatModelResponse,
    CompletionModelFunction,
    Embedding,
    EmbeddingModelInfo,
    EmbeddingModelResponse,
    ModelProviderService,
    _ModelName,
    _ModelProviderSettings,
)
from .utils import validate_tool_calls

_T = TypeVar("_T")
_P = ParamSpec("_P")


class _BaseOpenAIProvider(BaseModelProvider[_ModelName, _ModelProviderSettings]):
    """Base class for LLM providers with OpenAI-like APIs"""

    MODELS: ClassVar[
        Mapping[_ModelName, ChatModelInfo[_ModelName] | EmbeddingModelInfo[_ModelName]]  # type: ignore # noqa
    ]

    def __init__(
        self,
        settings: Optional[_ModelProviderSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not getattr(self, "MODELS", None):
            raise ValueError(f"{self.__class__.__name__}.MODELS is not set")

        if not settings:
            settings = self.default_settings.model_copy(deep=True)
        if not settings.credentials:
            settings.credentials = get_args(  # Union[Credentials, None] -> Credentials
                self.default_settings.model_fields["credentials"].annotation
            )[0].from_env()

        super(_BaseOpenAIProvider, self).__init__(settings=settings, logger=logger)

        if not getattr(self, "_client", None):
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                **self._credentials.get_api_access_kwargs()  # type: ignore
            )

    async def get_available_models(
        self,
    ) -> Sequence[ChatModelInfo[_ModelName] | EmbeddingModelInfo[_ModelName]]:
        _models = (await self._client.models.list()).data
        return [
            self.MODELS[cast(_ModelName, m.id)] for m in _models if m.id in self.MODELS
        ]

    def get_token_limit(self, model_name: _ModelName) -> int:
        """Get the maximum number of input tokens for a given model"""
        return self.MODELS[model_name].max_tokens

    def count_tokens(self, text: str, model_name: _ModelName) -> int:
        return len(self.get_tokenizer(model_name).encode(text))

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
        return f"{self.__class__.__name__}()"


class BaseOpenAIChatProvider(
    _BaseOpenAIProvider[_ModelName, _ModelProviderSettings],
    BaseChatModelProvider[_ModelName, _ModelProviderSettings],
):
    CHAT_MODELS: ClassVar[dict[_ModelName, ChatModelInfo[_ModelName]]]  # type: ignore

    def __init__(
        self,
        settings: Optional[_ModelProviderSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not getattr(self, "CHAT_MODELS", None):
            raise ValueError(f"{self.__class__.__name__}.CHAT_MODELS is not set")

        super(BaseOpenAIChatProvider, self).__init__(settings=settings, logger=logger)

    async def get_available_chat_models(self) -> Sequence[ChatModelInfo[_ModelName]]:
        all_available_models = await self.get_available_models()
        return [
            model
            for model in all_available_models
            if model.service == ModelProviderService.CHAT
        ]

    def count_message_tokens(
        self,
        messages: ChatMessage | list[ChatMessage],
        model_name: _ModelName,
    ) -> int:
        if isinstance(messages, ChatMessage):
            messages = [messages]
        return self.count_tokens(
            "\n\n".join(f"{m.role.upper()}: {m.content}" for m in messages), model_name
        )

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
        """Create a chat completion using the API."""

        (
            openai_messages,
            completion_kwargs,
            parse_kwargs,
        ) = self._get_chat_completion_args(
            prompt_messages=model_prompt,
            model=model_name,
            functions=functions,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )

        total_cost = 0.0
        attempts = 0
        while True:
            completion_kwargs["messages"] = openai_messages
            _response, _cost, t_input, t_output = await self._create_chat_completion(
                model=model_name,
                completion_kwargs=completion_kwargs,
            )
            total_cost += _cost

            # If parsing the response fails, append the error to the prompt, and let the
            # LLM fix its mistake(s).
            attempts += 1
            parse_errors: list[Exception] = []

            _assistant_msg = _response.choices[0].message

            tool_calls, _errors = self._parse_assistant_tool_calls(
                _assistant_msg, **parse_kwargs
            )
            parse_errors += _errors

            # Validate tool calls
            if not parse_errors and tool_calls and functions:
                parse_errors += validate_tool_calls(tool_calls, functions)

            assistant_msg = AssistantChatMessage(
                content=_assistant_msg.content or "",
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
                        content=_assistant_msg.content or "",
                        tool_calls=tool_calls or None,
                    ),
                    parsed_result=parsed_result,
                    llm_info=self.CHAT_MODELS[model_name],
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
                    openai_messages.append(
                        cast(
                            ChatCompletionAssistantMessageParam,
                            _assistant_msg.model_dump(exclude_none=True),
                        )
                    )
                    openai_messages.append(
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
        model: _ModelName,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs,
    ) -> tuple[
        list[ChatCompletionMessageParam], CompletionCreateParams, dict[str, Any]
    ]:
        """Prepare keyword arguments for a chat completion API call

        Args:
            prompt_messages: List of ChatMessages
            model: The model to use
            functions (optional): List of functions available to the LLM
            max_output_tokens (optional): Maximum number of tokens to generate

        Returns:
            list[ChatCompletionMessageParam]: Prompt messages for the API call
            CompletionCreateParams: Mapping of other kwargs for the API call
            Mapping[str, Any]: Any keyword arguments to pass on to the completion parser
        """
        kwargs = cast(CompletionCreateParams, kwargs)

        if max_output_tokens:
            kwargs["max_tokens"] = max_output_tokens

        if functions:
            kwargs["tools"] = [  # pyright: ignore - it fails to infer the dict type
                {"type": "function", "function": format_function_def_for_openai(f)}
                for f in functions
            ]
            if len(functions) == 1:
                # force the model to call the only specified function
                kwargs["tool_choice"] = {  # pyright: ignore - type inference failure
                    "type": "function",
                    "function": {"name": functions[0].name},
                }

        if extra_headers := self._configuration.extra_request_headers:
            # 'extra_headers' is not on CompletionCreateParams, but is on chat.create()
            kwargs["extra_headers"] = kwargs.get("extra_headers", {})  # type: ignore
            kwargs["extra_headers"].update(extra_headers.copy())  # type: ignore

        prepped_messages: list[ChatCompletionMessageParam] = [
            message.model_dump(  # type: ignore
                include={"role", "content", "tool_calls", "tool_call_id", "name"},
                exclude_none=True,
            )
            for message in prompt_messages
        ]

        if "messages" in kwargs:
            prepped_messages += kwargs["messages"]
            del kwargs["messages"]  # type: ignore - messages are added back later

        return prepped_messages, kwargs, {}

    async def _create_chat_completion(
        self,
        model: _ModelName,
        completion_kwargs: CompletionCreateParams,
    ) -> tuple[ChatCompletion, float, int, int]:
        """
        Create a chat completion using an OpenAI-like API with retry handling

        Params:
            model: The model to use for the completion
            completion_kwargs: All other arguments for the completion call

        Returns:
            ChatCompletion: The chat completion response object
            float: The cost ($) of this completion
            int: Number of prompt tokens used
            int: Number of completion tokens used
        """
        completion_kwargs["model"] = completion_kwargs.get("model") or model

        @self._retry_api_request
        async def _create_chat_completion_with_retry() -> ChatCompletion:
            return await self._client.chat.completions.create(
                **completion_kwargs,  # type: ignore
            )

        completion = await _create_chat_completion_with_retry()

        if completion.usage:
            prompt_tokens_used = completion.usage.prompt_tokens
            completion_tokens_used = completion.usage.completion_tokens
        else:
            prompt_tokens_used = completion_tokens_used = 0

        if self._budget:
            cost = self._budget.update_usage_and_cost(
                model_info=self.CHAT_MODELS[model],
                input_tokens_used=prompt_tokens_used,
                output_tokens_used=completion_tokens_used,
            )
        else:
            cost = 0

        self._logger.debug(
            f"{model} completion usage: {prompt_tokens_used} input, "
            f"{completion_tokens_used} output - ${round(cost, 5)}"
        )
        return completion, cost, prompt_tokens_used, completion_tokens_used

    def _parse_assistant_tool_calls(
        self, assistant_message: ChatCompletionMessage, **kwargs
    ) -> tuple[list[AssistantToolCall], list[Exception]]:
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


class BaseOpenAIEmbeddingProvider(
    _BaseOpenAIProvider[_ModelName, _ModelProviderSettings],
    BaseEmbeddingModelProvider[_ModelName, _ModelProviderSettings],
):
    EMBEDDING_MODELS: ClassVar[
        dict[_ModelName, EmbeddingModelInfo[_ModelName]]  # type: ignore
    ]

    def __init__(
        self,
        settings: Optional[_ModelProviderSettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not getattr(self, "EMBEDDING_MODELS", None):
            raise ValueError(f"{self.__class__.__name__}.EMBEDDING_MODELS is not set")

        super(BaseOpenAIEmbeddingProvider, self).__init__(
            settings=settings, logger=logger
        )

    async def get_available_embedding_models(
        self,
    ) -> Sequence[EmbeddingModelInfo[_ModelName]]:
        all_available_models = await self.get_available_models()
        return [
            model
            for model in all_available_models
            if model.service == ModelProviderService.EMBEDDING
        ]

    async def create_embedding(
        self,
        text: str,
        model_name: _ModelName,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelResponse:
        """Create an embedding using an OpenAI-like API"""
        embedding_kwargs = self._get_embedding_kwargs(
            input=text, model=model_name, **kwargs
        )
        response = await self._create_embedding(embedding_kwargs)

        return EmbeddingModelResponse(
            embedding=embedding_parser(response.data[0].embedding),
            llm_info=self.EMBEDDING_MODELS[model_name],
            prompt_tokens_used=response.usage.prompt_tokens,
        )

    def _get_embedding_kwargs(
        self, input: str | list[str], model: _ModelName, **kwargs
    ) -> EmbeddingCreateParams:
        """Get kwargs for an embedding API call

        Params:
            input: Text body or list of text bodies to create embedding(s) from
            model: Embedding model to use

        Returns:
            The kwargs for the embedding API call
        """
        kwargs = cast(EmbeddingCreateParams, kwargs)

        kwargs["input"] = input
        kwargs["model"] = model

        if extra_headers := self._configuration.extra_request_headers:
            # 'extra_headers' is not on CompletionCreateParams, but is on embedding.create()  # noqa
            kwargs["extra_headers"] = kwargs.get("extra_headers", {})  # type: ignore
            kwargs["extra_headers"].update(extra_headers.copy())  # type: ignore

        return kwargs

    def _create_embedding(
        self, embedding_kwargs: EmbeddingCreateParams
    ) -> Awaitable[CreateEmbeddingResponse]:
        """Create an embedding using an OpenAI-like API with retry handling."""

        @self._retry_api_request
        async def _create_embedding_with_retry() -> CreateEmbeddingResponse:
            return await self._client.embeddings.create(**embedding_kwargs)

        return _create_embedding_with_retry()


def format_function_def_for_openai(self: CompletionModelFunction) -> FunctionDefinition:
    """Returns an OpenAI-consumable function definition"""

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
