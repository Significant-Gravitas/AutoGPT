import enum
import logging
import os
from pathlib import Path
from typing import Any, Callable, Coroutine, Iterator, Optional, ParamSpec, TypeVar

import sentry_sdk
import tenacity
import tiktoken
import yaml
from openai._exceptions import APIStatusError, RateLimitError
from openai.types import CreateEmbeddingResponse
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from pydantic import SecretStr

from forge.config.schema import Configurable, UserConfigurable
from forge.json.parsing import json_loads
from forge.json.schema import JSONSchema
from forge.llm.providers.schema import (
    AssistantChatMessage,
    AssistantFunctionCall,
    AssistantToolCall,
    AssistantToolCallDict,
    ChatMessage,
    ChatModelInfo,
    ChatModelProvider,
    ChatModelResponse,
    CompletionModelFunction,
    Embedding,
    EmbeddingModelInfo,
    EmbeddingModelProvider,
    EmbeddingModelResponse,
    ModelProviderBudget,
    ModelProviderConfiguration,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderSettings,
    ModelTokenizer,
)

from .utils import validate_tool_calls

_T = TypeVar("_T")
_P = ParamSpec("_P")

OpenAIEmbeddingParser = Callable[[Embedding], Embedding]


class OpenAIModelName(str, enum.Enum):
    EMBEDDING_v2 = "text-embedding-ada-002"
    EMBEDDING_v3_S = "text-embedding-3-small"
    EMBEDDING_v3_L = "text-embedding-3-large"

    GPT3_v1 = "gpt-3.5-turbo-0301"
    GPT3_v2 = "gpt-3.5-turbo-0613"
    GPT3_v2_16k = "gpt-3.5-turbo-16k-0613"
    GPT3_v3 = "gpt-3.5-turbo-1106"
    GPT3_v4 = "gpt-3.5-turbo-0125"
    GPT3_ROLLING = "gpt-3.5-turbo"
    GPT3_ROLLING_16k = "gpt-3.5-turbo-16k"
    GPT3 = GPT3_ROLLING
    GPT3_16k = GPT3_ROLLING_16k

    GPT4_v1 = "gpt-4-0314"
    GPT4_v1_32k = "gpt-4-32k-0314"
    GPT4_v2 = "gpt-4-0613"
    GPT4_v2_32k = "gpt-4-32k-0613"
    GPT4_v3 = "gpt-4-1106-preview"
    GPT4_v3_VISION = "gpt-4-1106-vision-preview"
    GPT4_v4 = "gpt-4-0125-preview"
    GPT4_v5 = "gpt-4-turbo-2024-04-09"
    GPT4_ROLLING = "gpt-4"
    GPT4_ROLLING_32k = "gpt-4-32k"
    GPT4_TURBO = "gpt-4-turbo"
    GPT4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT4_VISION = "gpt-4-vision-preview"
    GPT4 = GPT4_ROLLING
    GPT4_32k = GPT4_ROLLING_32k


OPEN_AI_EMBEDDING_MODELS = {
    info.name: info
    for info in [
        EmbeddingModelInfo(
            name=OpenAIModelName.EMBEDDING_v2,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0001 / 1000,
            max_tokens=8191,
            embedding_dimensions=1536,
        ),
        EmbeddingModelInfo(
            name=OpenAIModelName.EMBEDDING_v3_S,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.00002 / 1000,
            max_tokens=8191,
            embedding_dimensions=1536,
        ),
        EmbeddingModelInfo(
            name=OpenAIModelName.EMBEDDING_v3_L,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.00013 / 1000,
            max_tokens=8191,
            embedding_dimensions=3072,
        ),
    ]
}


OPEN_AI_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=OpenAIModelName.GPT3_v1,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0015 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=4096,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_v2_16k,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.003 / 1000,
            completion_token_cost=0.004 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_v3,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.001 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_v4,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0005 / 1000,
            completion_token_cost=0.0015 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_v1,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.03 / 1000,
            completion_token_cost=0.06 / 1000,
            max_tokens=8191,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_v1_32k,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.06 / 1000,
            completion_token_cost=0.12 / 1000,
            max_tokens=32768,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_TURBO,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.01 / 1000,
            completion_token_cost=0.03 / 1000,
            max_tokens=128000,
            has_function_call_api=True,
        ),
    ]
}
# Copy entries for models with equivalent specs
chat_model_mapping = {
    OpenAIModelName.GPT3_v1: [OpenAIModelName.GPT3_v2],
    OpenAIModelName.GPT3_v2_16k: [OpenAIModelName.GPT3_16k],
    OpenAIModelName.GPT3_v4: [OpenAIModelName.GPT3_ROLLING],
    OpenAIModelName.GPT4_v1: [OpenAIModelName.GPT4_v2, OpenAIModelName.GPT4_ROLLING],
    OpenAIModelName.GPT4_v1_32k: [
        OpenAIModelName.GPT4_v2_32k,
        OpenAIModelName.GPT4_32k,
    ],
    OpenAIModelName.GPT4_TURBO: [
        OpenAIModelName.GPT4_v3,
        OpenAIModelName.GPT4_v3_VISION,
        OpenAIModelName.GPT4_VISION,
        OpenAIModelName.GPT4_v4,
        OpenAIModelName.GPT4_TURBO_PREVIEW,
        OpenAIModelName.GPT4_v5,
    ],
}
for base, copies in chat_model_mapping.items():
    for copy in copies:
        copy_info = OPEN_AI_CHAT_MODELS[base].copy(update={"name": copy})
        OPEN_AI_CHAT_MODELS[copy] = copy_info
        if copy.endswith(("-0301", "-0314")):
            copy_info.has_function_call_api = False


OPEN_AI_MODELS = {
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}


class OpenAIConfiguration(ModelProviderConfiguration):
    fix_failed_parse_tries: int = UserConfigurable(3)


class OpenAICredentials(ModelProviderCredentials):
    """Credentials for OpenAI."""

    api_key: SecretStr = UserConfigurable(from_env="OPENAI_API_KEY")
    api_base: Optional[SecretStr] = UserConfigurable(
        default=None, from_env="OPENAI_API_BASE_URL"
    )
    organization: Optional[SecretStr] = UserConfigurable(from_env="OPENAI_ORGANIZATION")

    api_type: str = UserConfigurable(
        default="",
        from_env=lambda: (
            "azure"
            if os.getenv("USE_AZURE") == "True"
            else os.getenv("OPENAI_API_TYPE")
        ),
    )
    api_version: str = UserConfigurable("", from_env="OPENAI_API_VERSION")
    azure_endpoint: Optional[SecretStr] = None
    azure_model_to_deploy_id_map: Optional[dict[str, str]] = None

    def get_api_access_kwargs(self) -> dict[str, str]:
        kwargs = {
            k: (v.get_secret_value() if type(v) is SecretStr else v)
            for k, v in {
                "api_key": self.api_key,
                "base_url": self.api_base,
                "organization": self.organization,
            }.items()
            if v is not None
        }
        if self.api_type == "azure":
            kwargs["api_version"] = self.api_version
            assert self.azure_endpoint, "Azure endpoint not configured"
            kwargs["azure_endpoint"] = self.azure_endpoint.get_secret_value()
        return kwargs

    def get_model_access_kwargs(self, model: str) -> dict[str, str]:
        kwargs = {"model": model}
        if self.api_type == "azure" and model:
            azure_kwargs = self._get_azure_access_kwargs(model)
            kwargs.update(azure_kwargs)
        return kwargs

    def load_azure_config(self, config_file: Path) -> None:
        with open(config_file) as file:
            config_params = yaml.load(file, Loader=yaml.SafeLoader) or {}

        try:
            assert config_params.get(
                "azure_model_map", {}
            ), "Azure model->deployment_id map is empty"
        except AssertionError as e:
            raise ValueError(*e.args)

        self.api_type = config_params.get("azure_api_type", "azure")
        self.api_version = config_params.get("azure_api_version", "")
        self.azure_endpoint = config_params.get("azure_endpoint")
        self.azure_model_to_deploy_id_map = config_params.get("azure_model_map")

    def _get_azure_access_kwargs(self, model: str) -> dict[str, str]:
        """Get the kwargs for the Azure API."""

        if not self.azure_model_to_deploy_id_map:
            raise ValueError("Azure model deployment map not configured")

        if model not in self.azure_model_to_deploy_id_map:
            raise ValueError(f"No Azure deployment ID configured for model '{model}'")
        deployment_id = self.azure_model_to_deploy_id_map[model]

        return {"model": deployment_id}


class OpenAISettings(ModelProviderSettings):
    configuration: OpenAIConfiguration
    credentials: Optional[OpenAICredentials]
    budget: ModelProviderBudget


class OpenAIProvider(
    Configurable[OpenAISettings], ChatModelProvider, EmbeddingModelProvider
):
    default_settings = OpenAISettings(
        name="openai_provider",
        description="Provides access to OpenAI's API.",
        configuration=OpenAIConfiguration(
            retries_per_request=7,
        ),
        credentials=None,
        budget=ModelProviderBudget(),
    )

    _settings: OpenAISettings
    _configuration: OpenAIConfiguration
    _credentials: OpenAICredentials
    _budget: ModelProviderBudget

    def __init__(
        self,
        settings: Optional[OpenAISettings] = None,
        logger: Optional[logging.Logger] = None,
    ):
        if not settings:
            settings = self.default_settings.copy(deep=True)
        if not settings.credentials:
            settings.credentials = OpenAICredentials.from_env()

        super(OpenAIProvider, self).__init__(settings=settings, logger=logger)

        if self._credentials.api_type == "azure":
            from openai import AsyncAzureOpenAI

            # API key and org (if configured) are passed, the rest of the required
            # credentials is loaded from the environment by the AzureOpenAI client.
            self._client = AsyncAzureOpenAI(**self._credentials.get_api_access_kwargs())
        else:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(**self._credentials.get_api_access_kwargs())

    async def get_available_models(self) -> list[ChatModelInfo]:
        _models = (await self._client.models.list()).data
        return [OPEN_AI_MODELS[m.id] for m in _models if m.id in OPEN_AI_MODELS]

    def get_token_limit(self, model_name: str) -> int:
        """Get the token limit for a given model."""
        return OPEN_AI_MODELS[model_name].max_tokens

    @classmethod
    def get_tokenizer(cls, model_name: OpenAIModelName) -> ModelTokenizer:
        return tiktoken.encoding_for_model(model_name)

    @classmethod
    def count_tokens(cls, text: str, model_name: OpenAIModelName) -> int:
        encoding = cls.get_tokenizer(model_name)
        return len(encoding.encode(text))

    @classmethod
    def count_message_tokens(
        cls,
        messages: ChatMessage | list[ChatMessage],
        model_name: OpenAIModelName,
    ) -> int:
        if isinstance(messages, ChatMessage):
            messages = [messages]

        if model_name.startswith("gpt-3.5-turbo"):
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
            encoding_model = "gpt-3.5-turbo"
        elif model_name.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
            encoding_model = "gpt-4"
        else:
            raise NotImplementedError(
                f"count_message_tokens() is not implemented for model {model_name}.\n"
                " See https://github.com/openai/openai-python/blob/main/chatml.md for"
                " information on how messages are converted to tokens."
            )
        try:
            encoding = tiktoken.encoding_for_model(encoding_model)
        except KeyError:
            logging.getLogger(__class__.__name__).warning(
                f"Model {model_name} not found. Defaulting to cl100k_base encoding."
            )
            encoding = tiktoken.get_encoding("cl100k_base")

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.dict().items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens

    async def create_chat_completion(
        self,
        model_prompt: list[ChatMessage],
        model_name: OpenAIModelName,
        completion_parser: Callable[[AssistantChatMessage], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        max_output_tokens: Optional[int] = None,
        prefill_response: str = "",  # not supported by OpenAI
        **kwargs,
    ) -> ChatModelResponse[_T]:
        """Create a completion using the OpenAI API and parse it."""

        openai_messages, completion_kwargs = self._get_chat_completion_args(
            model_prompt=model_prompt,
            model_name=model_name,
            functions=functions,
            max_tokens=max_output_tokens,
            **kwargs,
        )
        tool_calls_compat_mode = bool(functions and "tools" not in completion_kwargs)

        total_cost = 0.0
        attempts = 0
        while True:
            _response, _cost, t_input, t_output = await self._create_chat_completion(
                messages=openai_messages,
                **completion_kwargs,
            )
            total_cost += _cost

            # If parsing the response fails, append the error to the prompt, and let the
            # LLM fix its mistake(s).
            attempts += 1
            parse_errors: list[Exception] = []

            _assistant_msg = _response.choices[0].message

            tool_calls, _errors = self._parse_assistant_tool_calls(
                _assistant_msg, tool_calls_compat_mode
            )
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
                    model_info=OPEN_AI_CHAT_MODELS[model_name],
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
                    openai_messages.append(_assistant_msg.dict(exclude_none=True))
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

    async def create_embedding(
        self,
        text: str,
        model_name: OpenAIModelName,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelResponse:
        """Create an embedding using the OpenAI API."""
        embedding_kwargs = self._get_embedding_kwargs(model_name, **kwargs)
        response = await self._create_embedding(text=text, **embedding_kwargs)

        response = EmbeddingModelResponse(
            embedding=embedding_parser(response.data[0].embedding),
            model_info=OPEN_AI_EMBEDDING_MODELS[model_name],
            prompt_tokens_used=response.usage.prompt_tokens,
            completion_tokens_used=0,
        )
        self._budget.update_usage_and_cost(response)
        return response

    def _get_chat_completion_args(
        self,
        model_prompt: list[ChatMessage],
        model_name: OpenAIModelName,
        functions: Optional[list[CompletionModelFunction]] = None,
        **kwargs,
    ) -> tuple[list[ChatCompletionMessageParam], dict[str, Any]]:
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
        kwargs.update(self._credentials.get_model_access_kwargs(model_name))

        if functions:
            if OPEN_AI_CHAT_MODELS[model_name].has_function_call_api:
                kwargs["tools"] = [
                    {"type": "function", "function": f.schema} for f in functions
                ]
                if len(functions) == 1:
                    # force the model to call the only specified function
                    kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": functions[0].name},
                    }
            else:
                # Provide compatibility with older models
                _functions_compat_fix_kwargs(functions, kwargs)

        if extra_headers := self._configuration.extra_request_headers:
            kwargs["extra_headers"] = kwargs.get("extra_headers", {})
            kwargs["extra_headers"].update(extra_headers.copy())

        if "messages" in kwargs:
            model_prompt += kwargs["messages"]
            del kwargs["messages"]

        openai_messages: list[ChatCompletionMessageParam] = [
            message.dict(
                include={"role", "content", "tool_calls", "name"},
                exclude_none=True,
            )
            for message in model_prompt
        ]

        return openai_messages, kwargs

    def _get_embedding_kwargs(
        self,
        model_name: OpenAIModelName,
        **kwargs,
    ) -> dict:
        """Get kwargs for embedding API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            The kwargs for the embedding API call.

        """
        kwargs.update(self._credentials.get_model_access_kwargs(model_name))

        if extra_headers := self._configuration.extra_request_headers:
            kwargs["extra_headers"] = kwargs.get("extra_headers", {})
            kwargs["extra_headers"].update(extra_headers.copy())

        return kwargs

    async def _create_chat_completion(
        self,
        messages: list[ChatCompletionMessageParam],
        model: OpenAIModelName,
        *_,
        **kwargs,
    ) -> tuple[ChatCompletion, float, int, int]:
        """
        Create a chat completion using the OpenAI API with retry handling.

        Params:
            openai_messages: List of OpenAI-consumable message dict objects
            model: The model to use for the completion

        Returns:
            ChatCompletion: The chat completion response object
            float: The cost ($) of this completion
            int: Number of prompt tokens used
            int: Number of completion tokens used
        """

        @self._retry_api_request
        async def _create_chat_completion_with_retry(
            messages: list[ChatCompletionMessageParam], **kwargs
        ) -> ChatCompletion:
            return await self._client.chat.completions.create(
                messages=messages,  # type: ignore
                **kwargs,
            )

        completion = await _create_chat_completion_with_retry(
            messages, model=model, **kwargs
        )

        if completion.usage:
            prompt_tokens_used = completion.usage.prompt_tokens
            completion_tokens_used = completion.usage.completion_tokens
        else:
            prompt_tokens_used = completion_tokens_used = 0

        cost = self._budget.update_usage_and_cost(
            model_info=OPEN_AI_CHAT_MODELS[model],
            input_tokens_used=prompt_tokens_used,
            output_tokens_used=completion_tokens_used,
        )
        self._logger.debug(
            f"Completion usage: {prompt_tokens_used} input, "
            f"{completion_tokens_used} output - ${round(cost, 5)}"
        )
        return completion, cost, prompt_tokens_used, completion_tokens_used

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

        elif compat_mode and assistant_message.content:
            try:
                tool_calls = list(
                    _tool_calls_compat_extract_calls(assistant_message.content)
                )
            except Exception as e:
                parse_errors.append(e)

        return tool_calls, parse_errors

    def _create_embedding(
        self, text: str, *_, **kwargs
    ) -> Coroutine[None, None, CreateEmbeddingResponse]:
        """Create an embedding using the OpenAI API with retry handling."""

        @self._retry_api_request
        async def _create_embedding_with_retry(
            text: str, *_, **kwargs
        ) -> CreateEmbeddingResponse:
            return await self._client.embeddings.create(
                input=[text],
                **kwargs,
            )

        return _create_embedding_with_retry(text, *_, **kwargs)

    def _retry_api_request(self, func: Callable[_P, _T]) -> Callable[_P, _T]:
        _log_retry_debug_message = tenacity.after_log(self._logger, logging.DEBUG)

        def _log_on_fail(retry_state: tenacity.RetryCallState) -> None:
            _log_retry_debug_message(retry_state)

            if (
                retry_state.attempt_number == 0
                and retry_state.outcome
                and isinstance(retry_state.outcome.exception(), RateLimitError)
            ):
                self._logger.warning(
                    "Please double check that you have setup a PAID OpenAI API Account."
                    " You can read more here: "
                    "https://docs.agpt.co/setup/#getting-an-openai-api-key"
                )

        return tenacity.retry(
            retry=(
                tenacity.retry_if_exception_type(RateLimitError)
                | tenacity.retry_if_exception(
                    lambda e: isinstance(e, APIStatusError) and e.status_code == 502
                )
            ),
            wait=tenacity.wait_exponential(),
            stop=tenacity.stop_after_attempt(self._configuration.retries_per_request),
            after=_log_on_fail,
        )(func)

    def __repr__(self):
        return "OpenAIProvider()"


def format_function_specs_as_typescript_ns(
    functions: list[CompletionModelFunction],
) -> str:
    """Returns a function signature block in the format used by OpenAI internally:
    https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18

    For use with `count_tokens` to determine token usage of provided functions.

    Example:
    ```ts
    namespace functions {

    // Get the current weather in a given location
    type get_current_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    unit?: "celsius" | "fahrenheit",
    }) => any;

    } // namespace functions
    ```
    """

    return (
        "namespace functions {\n\n"
        + "\n\n".join(format_openai_function_for_prompt(f) for f in functions)
        + "\n\n} // namespace functions"
    )


def format_openai_function_for_prompt(func: CompletionModelFunction) -> str:
    """Returns the function formatted similarly to the way OpenAI does it internally:
    https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18

    Example:
    ```ts
    // Get the current weather in a given location
    type get_current_weather = (_: {
    // The city and state, e.g. San Francisco, CA
    location: string,
    unit?: "celsius" | "fahrenheit",
    }) => any;
    ```
    """

    def param_signature(name: str, spec: JSONSchema) -> str:
        return (
            f"// {spec.description}\n" if spec.description else ""
        ) + f"{name}{'' if spec.required else '?'}: {spec.typescript_type},"

    return "\n".join(
        [
            f"// {func.description}",
            f"type {func.name} = (_ :{{",
            *[param_signature(name, p) for name, p in func.parameters.items()],
            "}) => any;",
        ]
    )


def count_openai_functions_tokens(
    functions: list[CompletionModelFunction], count_tokens: Callable[[str], int]
) -> int:
    """Returns the number of tokens taken up by a set of function definitions

    Reference: https://community.openai.com/t/how-to-calculate-the-tokens-when-using-function-call/266573/18  # noqa: E501
    """
    return count_tokens(
        "# Tools\n\n"
        "## functions\n\n"
        f"{format_function_specs_as_typescript_ns(functions)}"
    )


def _functions_compat_fix_kwargs(
    functions: list[CompletionModelFunction],
    completion_kwargs: dict,
):
    function_definitions = format_function_specs_as_typescript_ns(functions)
    function_call_schema = JSONSchema(
        type=JSONSchema.Type.OBJECT,
        properties={
            "name": JSONSchema(
                description="The name of the function to call",
                enum=[f.name for f in functions],
                required=True,
            ),
            "arguments": JSONSchema(
                description="The arguments for the function call",
                type=JSONSchema.Type.OBJECT,
                required=True,
            ),
        },
    )
    tool_calls_schema = JSONSchema(
        type=JSONSchema.Type.ARRAY,
        items=JSONSchema(
            type=JSONSchema.Type.OBJECT,
            properties={
                "type": JSONSchema(
                    type=JSONSchema.Type.STRING,
                    enum=["function"],
                ),
                "function": function_call_schema,
            },
        ),
    )
    completion_kwargs["messages"] = [
        ChatMessage.system(
            "# tool usage instructions\n\n"
            "Specify a '```tool_calls' block in your response,"
            " with a valid JSON object that adheres to the following schema:\n\n"
            f"{tool_calls_schema.to_dict()}\n\n"
            "Specify any tools that you need to use through this JSON object.\n\n"
            "Put the tool_calls block at the end of your response"
            " and include its fences if it is not the only content.\n\n"
            "## functions\n\n"
            "For the function call itself, use one of the following"
            f" functions:\n\n{function_definitions}"
        ),
    ]


def _tool_calls_compat_extract_calls(response: str) -> Iterator[AssistantToolCall]:
    import re
    import uuid

    logging.debug(f"Trying to extract tool calls from response:\n{response}")

    if response[0] == "[":
        tool_calls: list[AssistantToolCallDict] = json_loads(response)
    else:
        block = re.search(r"```(?:tool_calls)?\n(.*)\n```\s*$", response, re.DOTALL)
        if not block:
            raise ValueError("Could not find tool_calls block in response")
        tool_calls: list[AssistantToolCallDict] = json_loads(block.group(1))

    for t in tool_calls:
        t["id"] = str(uuid.uuid4())
        t["function"]["arguments"] = str(t["function"]["arguments"])  # HACK

        yield AssistantToolCall.parse_obj(t)
