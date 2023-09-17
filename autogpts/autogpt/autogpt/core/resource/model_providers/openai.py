import enum
import functools
import logging
import math
import time
from typing import Callable, ParamSpec, TypeVar

import openai
import tiktoken
from openai.error import APIError, RateLimitError

from autogpt.core.configuration import (
    Configurable,
    SystemConfiguration,
    UserConfigurable,
)
from autogpt.core.resource.model_providers.schema import (
    Embedding,
    EmbeddingModelProvider,
    EmbeddingModelInfo,
    EmbeddingModelResponse,
    LanguageModelFunction,
    LanguageModelMessage,
    LanguageModelProvider,
    LanguageModelInfo,
    LanguageModelResponse,
    ModelProviderBudget,
    ModelProviderCredentials,
    ModelProviderName,
    ModelProviderService,
    ModelProviderSettings,
    ModelProviderUsage,
)

OpenAIEmbeddingParser = Callable[[Embedding], Embedding]
OpenAIChatParser = Callable[[str], dict]


class OpenAIModelName(str, enum.Enum):
    ADA = "text-embedding-ada-002"

    GPT3_v1 = "gpt-3.5-turbo-0301"
    GPT3_v2 = "gpt-3.5-turbo-0613"
    GPT3_v2_16k = "gpt-3.5-turbo-16k-0613"
    GPT3_ROLLING = "gpt-3.5-turbo"
    GPT3_ROLLING_16k = "gpt-3.5-turbo-16k"
    GPT3 = GPT3_ROLLING
    GPT3_16k = GPT3_ROLLING_16k

    GPT4_v1 = "gpt-4-0314"
    GPT4_v1_32k = "gpt-4-32k-0314"
    GPT4_v2 = "gpt-4-0613"
    GPT4_v2_32k = "gpt-4-32k-0613"
    GPT4_ROLLING = "gpt-4"
    GPT4_ROLLING_32k = "gpt-4-32k"
    GPT4 = GPT4_ROLLING
    GPT4_32k = GPT4_ROLLING_32k


OPEN_AI_EMBEDDING_MODELS = {
    OpenAIModelName.ADA: EmbeddingModelInfo(
        name=OpenAIModelName.ADA,
        service=ModelProviderService.EMBEDDING,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.0001/1000,
        max_tokens=8191,
        embedding_dimensions=1536,
    ),
}


OPEN_AI_LANGUAGE_MODELS = {
    info.name: info
    for info in [
        LanguageModelInfo(
            name=OpenAIModelName.GPT3,
            service=ModelProviderService.LANGUAGE,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0015/1000,
            completion_token_cost=0.002/1000,
            max_tokens=4096,
        ),
        LanguageModelInfo(
            name=OpenAIModelName.GPT3_16k,
            service=ModelProviderService.LANGUAGE,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.003/1000,
            completion_token_cost=0.004/1000,
            max_tokens=16384,
        ),
        LanguageModelInfo(
            name=OpenAIModelName.GPT4,
            service=ModelProviderService.LANGUAGE,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.03/1000,
            completion_token_cost=0.06/1000,
            max_tokens=8192,
        ),
        LanguageModelInfo(
            name=OpenAIModelName.GPT4_32k,
            service=ModelProviderService.LANGUAGE,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.06/1000,
            completion_token_cost=0.12/1000,
            max_tokens=32768,
        ),
    ]
}
# Copy entries for equivalent models
chat_model_mapping = {
    OpenAIModelName.GPT3: [OpenAIModelName.GPT3_v1, OpenAIModelName.GPT3_v2],
    OpenAIModelName.GPT3_16k: [OpenAIModelName.GPT3_v2_16k],
    OpenAIModelName.GPT4: [OpenAIModelName.GPT4_v1, OpenAIModelName.GPT4_v2],
    OpenAIModelName.GPT4_32k: [OpenAIModelName.GPT4_v1_32k, OpenAIModelName.GPT4_v2_32k],
}
for base, copies in chat_model_mapping.items():
    for copy in copies:
        copy_info = LanguageModelInfo(**OPEN_AI_LANGUAGE_MODELS[base].__dict__)
        copy_info.name = copy
        OPEN_AI_LANGUAGE_MODELS[copy] = copy_info


OPEN_AI_MODELS = {
    **OPEN_AI_LANGUAGE_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}


class OpenAIConfiguration(SystemConfiguration):
    retries_per_request: int = UserConfigurable()


class OpenAIModelProviderBudget(ModelProviderBudget):
    graceful_shutdown_threshold: float = UserConfigurable()
    warning_threshold: float = UserConfigurable()


class OpenAISettings(ModelProviderSettings):
    configuration: OpenAIConfiguration
    credentials: ModelProviderCredentials
    budget: OpenAIModelProviderBudget


class OpenAIProvider(Configurable, LanguageModelProvider, EmbeddingModelProvider):
    default_settings = OpenAISettings(
        name="openai_provider",
        description="Provides access to OpenAI's API.",
        configuration=OpenAIConfiguration(
            retries_per_request=10,
        ),
        credentials=ModelProviderCredentials(),
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

    logger = logging.getLogger("model_providers.OpenAIProvider")

    def __init__(
        self,
        settings: OpenAISettings,
        logger: logging.Logger,
    ):
        self._configuration = settings.configuration
        self._credentials = settings.credentials
        self._budget = settings.budget

        self._logger = logger

        retry_handler = _OpenAIRetryHandler(
            logger=self._logger,
            num_retries=self._configuration.retries_per_request,
        )

        self._create_completion = retry_handler(_create_completion)
        self._create_embedding = retry_handler(_create_embedding)

    def get_token_limit(self, model_name: str) -> int:
        """Get the token limit for a given model."""
        return OPEN_AI_MODELS[model_name].max_tokens

    def get_remaining_budget(self) -> float:
        """Get the remaining budget."""
        return self._budget.remaining_budget

    def count_tokens(self, text: str, model_name: OpenAIModelName) -> int:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))

    def count_message_tokens(
        self,
        messages: LanguageModelMessage | list[LanguageModelMessage],
        model_name: OpenAIModelName,
    ) -> int:
        if isinstance(messages, LanguageModelMessage):
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
            self.logger.warn(
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

    async def create_language_completion(
        self,
        model_prompt: list[LanguageModelMessage],
        model_name: OpenAIModelName,
        completion_parser: Callable[[dict], dict],
        functions: list[LanguageModelFunction] = [],
        **kwargs,
    ) -> LanguageModelResponse:
        """Create a completion using the OpenAI API."""
        completion_kwargs = self._get_completion_kwargs(model_name, functions, **kwargs)
        response = await self._create_completion(
            messages=model_prompt,
            **completion_kwargs,
        )
        response_args = {
            "model_info": OPEN_AI_LANGUAGE_MODELS[model_name],
            "prompt_tokens_used": response.usage.prompt_tokens,
            "completion_tokens_used": response.usage.completion_tokens,
        }

        parsed_response = completion_parser(
            response.choices[0].message.to_dict_recursive()
        )
        response = LanguageModelResponse(
            content=parsed_response, **response_args
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
        functions: list[LanguageModelFunction],
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
            **self._credentials.unmasked(),
        }
        if functions:
            completion_kwargs["functions"] = functions

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


async def _create_completion(
    messages: list[LanguageModelMessage], *_, **kwargs
) -> openai.Completion:
    """Create a chat completion using the OpenAI API.

    Args:
        messages: The prompt to use.

    Returns:
        The completion.

    """
    messages = [message.dict() for message in messages]
    if "functions" in kwargs:
        kwargs["functions"] = [function.json_schema for function in kwargs["functions"]]
    return await openai.ChatCompletion.acreate(
        messages=messages,
        **kwargs,
    )


_T = TypeVar("_T")
_P = ParamSpec("_P")


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
        "read more here: https://docs.agpt.co/setup/#getting-an-api-key"
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
