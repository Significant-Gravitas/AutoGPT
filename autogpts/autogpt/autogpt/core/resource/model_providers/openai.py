import enum
import functools
import logging
import math
import os
import time
from pathlib import Path
from typing import Callable, Optional, ParamSpec, TypeVar

import openai
import tiktoken
import yaml
from openai.error import APIError, RateLimitError
from pydantic import SecretStr

from autogpt.core.configuration import Configurable, UserConfigurable
from autogpt.core.resource.model_providers.schema import (
    AssistantChatMessageDict,
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
    ModelProviderService,
    ModelProviderSettings,
    ModelProviderUsage,
    ModelTokenizer,
)
from autogpt.core.utils.json_schema import JSONSchema

_T = TypeVar("_T")
_P = ParamSpec("_P")

OpenAIEmbeddingParser = Callable[[Embedding], Embedding]
OpenAIChatParser = Callable[[str], dict]


class OpenAIModelName(str, enum.Enum):
    ADA = "text-embedding-ada-002"

    GPT3_v1 = "gpt-3.5-turbo-0301"
    GPT3_v2 = "gpt-3.5-turbo-0613"
    GPT3_v2_16k = "gpt-3.5-turbo-16k-0613"
    GPT3_v3 = "gpt-3.5-turbo-1106"
    GPT3_ROLLING = "gpt-3.5-turbo"
    GPT3_ROLLING_16k = "gpt-3.5-turbo-16k"
    GPT3 = GPT3_ROLLING
    GPT3_16k = GPT3_ROLLING_16k

    GPT4_v1 = "gpt-4-0314"
    GPT4_v1_32k = "gpt-4-32k-0314"
    GPT4_v2 = "gpt-4-0613"
    GPT4_v2_32k = "gpt-4-32k-0613"
    GPT4_v3 = "gpt-4-1106-preview"
    GPT4_ROLLING = "gpt-4"
    GPT4_ROLLING_32k = "gpt-4-32k"
    GPT4_VISION = "gpt-4-vision-preview"
    GPT4 = GPT4_ROLLING
    GPT4_32k = GPT4_ROLLING_32k


OPEN_AI_EMBEDDING_MODELS = {
    OpenAIModelName.ADA: EmbeddingModelInfo(
        name=OpenAIModelName.ADA,
        service=ModelProviderService.EMBEDDING,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.0001 / 1000,
        max_tokens=8191,
        embedding_dimensions=1536,
    ),
}


OPEN_AI_CHAT_MODELS = {
    info.name: info
    for info in [
        ChatModelInfo(
            name=OpenAIModelName.GPT3,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0015 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=4096,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_16k,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.003 / 1000,
            completion_token_cost=0.004 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_v3,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.001 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=16384,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.03 / 1000,
            completion_token_cost=0.06 / 1000,
            max_tokens=8191,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_32k,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.06 / 1000,
            completion_token_cost=0.12 / 1000,
            max_tokens=32768,
            has_function_call_api=True,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_v3,
            service=ModelProviderService.CHAT,
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
    OpenAIModelName.GPT3: [OpenAIModelName.GPT3_v1, OpenAIModelName.GPT3_v2],
    OpenAIModelName.GPT3_16k: [OpenAIModelName.GPT3_v2_16k],
    OpenAIModelName.GPT4: [OpenAIModelName.GPT4_v1, OpenAIModelName.GPT4_v2],
    OpenAIModelName.GPT4_32k: [
        OpenAIModelName.GPT4_v1_32k,
        OpenAIModelName.GPT4_v2_32k,
    ],
}
for base, copies in chat_model_mapping.items():
    for copy in copies:
        copy_info = ChatModelInfo(**OPEN_AI_CHAT_MODELS[base].__dict__)
        copy_info.name = copy
        OPEN_AI_CHAT_MODELS[copy] = copy_info
        if copy.endswith(("-0301", "-0314")):
            copy_info.has_function_call_api = False


OPEN_AI_MODELS = {
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}


class OpenAIConfiguration(ModelProviderConfiguration):
    fix_failed_parse_tries: int = UserConfigurable(3)
    pass


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
    azure_model_to_deploy_id_map: Optional[dict[str, str]] = None

    def get_api_access_kwargs(self, model: str = "") -> dict[str, str]:
        credentials = {k: v for k, v in self.unmasked().items() if type(v) is str}
        if self.api_type == "azure" and model:
            azure_credentials = self._get_azure_access_kwargs(model)
            credentials.update(azure_credentials)
        return credentials

    def load_azure_config(self, config_file: Path) -> None:
        with open(config_file) as file:
            config_params = yaml.load(file, Loader=yaml.FullLoader) or {}

        try:
            assert (
                azure_api_base := config_params.get("azure_api_base", "")
            ) != "", "Azure API base URL not set"
            assert config_params.get(
                "azure_model_map", {}
            ), "Azure model->deployment_id map is empty"
        except AssertionError as e:
            raise ValueError(*e.args)

        self.api_base = SecretStr(azure_api_base)
        self.api_type = config_params.get("azure_api_type", "azure")
        self.api_version = config_params.get("azure_api_version", "")
        self.azure_model_to_deploy_id_map = config_params.get("azure_model_map")

    def _get_azure_access_kwargs(self, model: str) -> dict[str, str]:
        """Get the kwargs for the Azure API."""

        if not self.azure_model_to_deploy_id_map:
            raise ValueError("Azure model deployment map not configured")

        if model not in self.azure_model_to_deploy_id_map:
            raise ValueError(f"No Azure deployment ID configured for model '{model}'")
        deployment_id = self.azure_model_to_deploy_id_map[model]

        if model in OPEN_AI_EMBEDDING_MODELS:
            return {"engine": deployment_id}
        else:
            return {"deployment_id": deployment_id}


class OpenAIModelProviderBudget(ModelProviderBudget):
    graceful_shutdown_threshold: float = UserConfigurable()
    warning_threshold: float = UserConfigurable()


class OpenAISettings(ModelProviderSettings):
    configuration: OpenAIConfiguration
    credentials: Optional[OpenAICredentials]
    budget: OpenAIModelProviderBudget


class OpenAIProvider(
    Configurable[OpenAISettings], ChatModelProvider, EmbeddingModelProvider
):
    default_settings = OpenAISettings(
        name="openai_provider",
        description="Provides access to OpenAI's API.",
        configuration=OpenAIConfiguration(
            retries_per_request=10,
        ),
        credentials=None,
        budget=OpenAIModelProviderBudget(
            total_budget=math.inf,
            total_cost=0.0,
            remaining_budget=math.inf,
            usage=ModelProviderUsage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            ),
            graceful_shutdown_threshold=0.005,
            warning_threshold=0.01,
        ),
    )

    _configuration: OpenAIConfiguration

    def __init__(
        self,
        settings: OpenAISettings,
        logger: logging.Logger,
    ):
        assert settings.credentials, "Cannot create OpenAIProvider without credentials"
        self._configuration = settings.configuration
        self._credentials = settings.credentials
        self._budget = settings.budget

        self._logger = logger

        retry_handler = _OpenAIRetryHandler(
            logger=self._logger,
            num_retries=self._configuration.retries_per_request,
        )

        self._create_chat_completion = retry_handler(_create_chat_completion)
        self._create_embedding = retry_handler(_create_embedding)

    def get_token_limit(self, model_name: str) -> int:
        """Get the token limit for a given model."""
        return OPEN_AI_MODELS[model_name].max_tokens

    def get_remaining_budget(self) -> float:
        """Get the remaining budget."""
        return self._budget.remaining_budget

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
            cls._logger.warning(
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
        completion_parser: Callable[[AssistantChatMessageDict], _T] = lambda _: None,
        functions: Optional[list[CompletionModelFunction]] = None,
        **kwargs,
    ) -> ChatModelResponse[_T]:
        """Create a completion using the OpenAI API."""

        completion_kwargs = self._get_completion_kwargs(model_name, functions, **kwargs)
        tool_calls_compat_mode = functions and "tools" not in completion_kwargs
        if "messages" in completion_kwargs:
            model_prompt += completion_kwargs["messages"]
            del completion_kwargs["messages"]

        attempts = 0
        while True:
            response = await self._create_chat_completion(
                messages=model_prompt,
                **completion_kwargs,
            )
            response_args = {
                "model_info": OPEN_AI_CHAT_MODELS[model_name],
                "prompt_tokens_used": response.usage.prompt_tokens,
                "completion_tokens_used": response.usage.completion_tokens,
            }

            response_message = response.choices[0].message.to_dict_recursive()
            if tool_calls_compat_mode:
                response_message["tool_calls"] = _tool_calls_compat_extract_calls(
                    response_message["content"]
                )

            # If parsing the response fails, append the error to the prompt, and let the
            # LLM fix its mistake(s).
            try:
                attempts += 1
                parsed_response = completion_parser(response_message)
                break
            except Exception as e:
                self._logger.warning(f"Parsing attempt #{attempts} failed: {e}")
                self._logger.debug(
                    f"Parsing failed on response: '''{response_message}'''"
                )
                if attempts < self._configuration.fix_failed_parse_tries:
                    model_prompt.append(
                        ChatMessage.system(f"ERROR PARSING YOUR RESPONSE:\n\n{e}")
                    )
                else:
                    raise

        response = ChatModelResponse(
            response=response_message,
            parsed_result=parsed_response,
            **response_args,
        )
        self._budget.update_usage_and_cost(response)
        return response

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

        response_args = {
            "model_info": OPEN_AI_EMBEDDING_MODELS[model_name],
            "prompt_tokens_used": response.usage.prompt_tokens,
            "completion_tokens_used": response.usage.completion_tokens,
        }
        response = EmbeddingModelResponse(
            **response_args,
            embedding=embedding_parser(response.embeddings[0]),
        )
        self._budget.update_usage_and_cost(response)
        return response

    def _get_completion_kwargs(
        self,
        model_name: OpenAIModelName,
        functions: Optional[list[CompletionModelFunction]] = None,
        **kwargs,
    ) -> dict:
        """Get kwargs for completion API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            The kwargs for the chat API call.

        """
        completion_kwargs = {
            "model": model_name,
            **kwargs,
            **self._credentials.get_api_access_kwargs(model_name),
        }

        if functions:
            if OPEN_AI_CHAT_MODELS[model_name].has_function_call_api:
                completion_kwargs["tools"] = [
                    {"type": "function", "function": f.schema} for f in functions
                ]
                if len(functions) == 1:
                    # force the model to call the only specified function
                    completion_kwargs["tool_choice"] = {
                        "type": "function",
                        "function": {"name": functions[0].name},
                    }
            else:
                # Provide compatibility with older models
                _functions_compat_fix_kwargs(functions, completion_kwargs)

        if extra_headers := self._configuration.extra_request_headers:
            if completion_kwargs.get("headers"):
                completion_kwargs["headers"].update(extra_headers)
            else:
                completion_kwargs["headers"] = extra_headers.copy()

        return completion_kwargs

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
        embedding_kwargs = {
            "model": model_name,
            **kwargs,
            **self._credentials.unmasked(),
        }

        if extra_headers := self._configuration.extra_request_headers:
            if embedding_kwargs.get("headers"):
                embedding_kwargs["headers"].update(extra_headers)
            else:
                embedding_kwargs["headers"] = extra_headers.copy()

        return embedding_kwargs

    def __repr__(self):
        return "OpenAIProvider()"


async def _create_embedding(text: str, *_, **kwargs) -> openai.Embedding:
    """Embed text using the OpenAI API.

    Args:
        text str: The text to embed.
        model str: The name of the model to use.

    Returns:
        str: The embedding.
    """
    return await openai.Embedding.acreate(
        input=[text],
        **kwargs,
    )


async def _create_chat_completion(
    messages: list[ChatMessage], *_, **kwargs
) -> openai.Completion:
    """Create a chat completion using the OpenAI API.

    Args:
        messages: The prompt to use.

    Returns:
        The completion.
    """
    raw_messages = [
        message.dict(include={"role", "content", "tool_calls", "name"})
        for message in messages
    ]
    return await openai.ChatCompletion.acreate(
        messages=raw_messages,
        **kwargs,
    )


class _OpenAIRetryHandler:
    """Retry Handler for OpenAI API call.

    Args:
        num_retries int: Number of retries. Defaults to 10.
        backoff_base float: Base for exponential backoff. Defaults to 2.
        warn_user bool: Whether to warn the user. Defaults to True.
    """

    _retry_limit_msg = "Error: Reached rate limit, passing..."
    _api_key_error_msg = (
        "Please double check that you have setup a PAID OpenAI API Account. You can "
        "read more here: https://docs.agpt.co/setup/#getting-an-openai-api-key"
    )
    _backoff_msg = "Error: API Bad gateway. Waiting {backoff} seconds..."

    def __init__(
        self,
        logger: logging.Logger,
        num_retries: int = 10,
        backoff_base: float = 2.0,
        warn_user: bool = True,
    ):
        self._logger = logger
        self._num_retries = num_retries
        self._backoff_base = backoff_base
        self._warn_user = warn_user

    def _log_rate_limit_error(self) -> None:
        self._logger.debug(self._retry_limit_msg)
        if self._warn_user:
            self._logger.warning(self._api_key_error_msg)
            self._warn_user = False

    def _backoff(self, attempt: int) -> None:
        backoff = self._backoff_base ** (attempt + 2)
        self._logger.debug(self._backoff_msg.format(backoff=backoff))
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
                    if (e.http_status != 502) or (attempt == num_attempts):
                        raise

                self._backoff(attempt)

        return _wrapped


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


def _tool_calls_compat_extract_calls(response: str) -> list[AssistantToolCallDict]:
    import json
    import re

    logging.debug(f"Trying to extract tool calls from response:\n{response}")

    if response[0] == "[":
        tool_calls: list[AssistantToolCallDict] = json.loads(response)
    else:
        block = re.search(r"```(?:tool_calls)?\n(.*)\n```\s*$", response, re.DOTALL)
        if not block:
            raise ValueError("Could not find tool calls block in response")
        tool_calls: list[AssistantToolCallDict] = json.loads(block.group(1))

    for t in tool_calls:
        t["function"]["arguments"] = str(t["function"]["arguments"])  # HACK

    return tool_calls
