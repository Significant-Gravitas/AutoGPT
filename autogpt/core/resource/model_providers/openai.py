import enum
import functools
import logging
import math
import time
from typing import Callable, ParamSpec, TypeVar

import openai
from openai.error import APIError, RateLimitError

from autogpt.core.configuration import Configurable, SystemConfiguration
from autogpt.core.planning import ModelPrompt
from autogpt.core.resource.model_providers.schema import (
    Embedding,
    EmbeddingModelProvider,
    EmbeddingModelProviderModelInfo,
    EmbeddingModelProviderModelResponse,
    LanguageModelProvider,
    LanguageModelProviderModelInfo,
    LanguageModelProviderModelResponse,
    ModelProviderBudget,
    ModelProviderCredentials,
    ModelProviderModelCredentials,
    ModelProviderName,
    ModelProviderService,
    ModelProviderSettings,
    ModelProviderUsage,
)

OpenAIEmbeddingParser = Callable[[Embedding], Embedding]
OpenAIChatParser = Callable[[str], dict]


class OpenAIModelName(str, enum.Enum):
    ADA = "text-embedding-ada-002"
    GPT3 = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4_32K = "gpt-4-32k"


OPEN_AI_EMBEDDING_MODELS = {
    OpenAIModelName.ADA: EmbeddingModelProviderModelInfo(
        name=OpenAIModelName.ADA,
        service=ModelProviderService.EMBEDDING,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.0004,
        completion_token_cost=0.0,
        max_tokens=8191,
        embedding_dimensions=1536,
    ),
}


OPEN_AI_LANGUAGE_MODELS = {
    OpenAIModelName.GPT3: LanguageModelProviderModelInfo(
        name=OpenAIModelName.GPT3,
        service=ModelProviderService.LANGUAGE,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.002,
        completion_token_cost=0.002,
        max_tokens=4097,
    ),
    OpenAIModelName.GPT4: LanguageModelProviderModelInfo(
        name=OpenAIModelName.GPT4,
        service=ModelProviderService.LANGUAGE,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.03,
        completion_token_cost=0.06,
        max_tokens=8192,
    ),
    OpenAIModelName.GPT4_32K: LanguageModelProviderModelInfo(
        name=OpenAIModelName.GPT4_32K,
        service=ModelProviderService.LANGUAGE,
        provider_name=ModelProviderName.OPENAI,
        prompt_token_cost=0.06,
        completion_token_cost=0.12,
        max_tokens=32768,
    ),
}


OPEN_AI_MODELS = {
    **OPEN_AI_LANGUAGE_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}


class OpenAICredentials(ModelProviderCredentials):
    use_azure: bool = False

    # @root_validator()
    # def validate_api_key(cls, values):
    #     api_key = values.get("api_key")
    #     models = values.get("models")
    #     if api_key is None:
    #         for model, credentials in models.items():
    #             if credentials.api_key is None:
    #                 raise ValueError(
    #                     f"Either api_key for the provider or api_key for model {model} must be provided."
    #                 )


class OpenAIConfiguration(SystemConfiguration):
    retries_per_request: int


class OpenAIModelProviderBudget(ModelProviderBudget):
    graceful_shutdown_threshold: float
    warning_threshold: float


class OpenAISettings(ModelProviderSettings):
    configuration: OpenAIConfiguration
    credentials: OpenAICredentials
    budget: OpenAIModelProviderBudget


class OpenAIProvider(
    Configurable,
    LanguageModelProvider,
    EmbeddingModelProvider,
):
    defaults = OpenAISettings(
        name="openai_provider",
        description="Provides access to OpenAI's API.",
        configuration=OpenAIConfiguration(
            retries_per_request=10,
        ),
        credentials=ModelProviderCredentials(
            models={
                OpenAIModelName.GPT3: ModelProviderModelCredentials(),
                OpenAIModelName.ADA: ModelProviderModelCredentials(),
            },
        ),
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

    def __init__(
        self,
        settings: OpenAISettings,
        logger: logging.Logger,
    ):
        self._configuration = settings.configuration
        # Resolve global credentials with model specific credentials
        self._model_credentials: dict[
            str, ModelProviderModelCredentials
        ] = settings.credentials.get_credentials()
        self._budget = settings.budget

        self._logger = logger

        retry_handler = _OpenAIRetryHandler(
            logger=self._logger,
            num_retries=self._configuration.retries_per_request,
        )

        self._create_completion = retry_handler(_create_completion)
        self._create_embedding = retry_handler(_create_embedding)

    async def create_language_completion(
        self,
        model_prompt: ModelPrompt,
        model_name: OpenAIModelName,
        completion_parser: Callable[[str], dict],
        **kwargs,
    ) -> LanguageModelProviderModelResponse:
        """Create a completion using the OpenAI API."""
        completion_kwargs = self._get_completion_kwargs(model_name, **kwargs)
        response = await self._create_completion(
            messages=model_prompt,
            **completion_kwargs,
        )
        response_args = {
            "model_info": OPEN_AI_LANGUAGE_MODELS[model_name],
            "prompt_tokens_used": response.usage.prompt_tokens,
            "completion_tokens_used": response.usage.completion_tokens,
        }
        content = completion_parser(response.choices[0].message.content)
        response = LanguageModelProviderModelResponse(**response_args, content=content)
        self._budget.update_usage_and_cost(response)
        return response

    async def create_embedding(
        self,
        text: str,
        model_name: OpenAIModelName,
        embedding_parser: Callable[[Embedding], Embedding],
        **kwargs,
    ) -> EmbeddingModelProviderModelResponse:
        """Create an embedding using the OpenAI API."""
        embedding_kwargs = self._get_embedding_kwargs(model_name, **kwargs)
        response = await self._create_embedding(text=text, **embedding_kwargs)

        response_args = {
            "model_info": OPEN_AI_EMBEDDING_MODELS[model_name],
            "prompt_tokens_used": response.usage.prompt_tokens,
            "completion_tokens_used": response.usage.completion_tokens,
        }
        response = EmbeddingModelProviderModelResponse(
            **response_args,
            embedding=embedding_parser(response.embeddings[0]),
        )
        self._budget.update_usage_and_cost(response)
        return response

    def _get_completion_kwargs(self, model_name: OpenAIModelName, **kwargs) -> dict:
        """Get kwargs for completion API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            The kwargs for the chat API call.

        """
        known_kwargs = {}

        completion_kwargs = {
            "model": model_name,
            **kwargs,
            **self._model_credentials[model_name].dict(),
        }

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
            **self._model_credentials[model_name].dict(),
        }

        return embedding_kwargs

    def __repr__(self):
        return "OpenAIProvider()"


async def _create_embedding(text: str, *_, **kwargs) -> openai.Embedding:
    """Embed text using the OpenAI API.

    Args:
        text str: The text to embed.
        model_name str: The name of the model to use.

    Returns:
        str: The embedding.
    """
    return await openai.Embedding.acreate(
        input=[text],
        **kwargs,
    )


async def _create_completion(messages: ModelPrompt, *_, **kwargs) -> openai.Completion:
    """Create a chat completion using the OpenAI API.

    Args:
        messages ModelPrompt: The prompt to use.
        model_name str: The name of the model to use.

    Returns:
        str: The completion.
    """
    messages = [message.dict() for message in messages]
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
