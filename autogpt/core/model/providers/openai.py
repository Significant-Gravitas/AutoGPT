import functools
import logging
import time
from typing import Callable, ParamSpec, TypeVar

import openai
from openai.error import APIError, RateLimitError

from autogpt.core.credentials.simple import (
    CredentialsConsumer,
    SimpleCredentialsService,
)
from autogpt.core.model.embedding.base import (
    Embedding,
    EmbeddingModelInfo,
    EmbeddingModelProvider,
    EmbeddingModelResponse,
)
from autogpt.core.model.language.base import (
    LanguageModelInfo,
    LanguageModelProvider,
    LanguageModelResponse,
)
from autogpt.core.planning import ModelPrompt

OpenAIChatParser = Callable[[str], dict]
OpenAIEmbeddingParser = Callable[[list[float]], list[float]]


OPEN_AI_LANGUAGE_MODELS = {
    "gpt-3.5-turbo": LanguageModelInfo(
        name="gpt-3.5-turbo",
        prompt_token_cost=0.002,
        completion_token_cost=0.002,
        max_tokens=4096,
    ),
    "gpt-4": LanguageModelInfo(
        name="gpt-4",
        prompt_token_cost=0.03,
        completion_token_cost=0.06,
        max_tokens=8192,
    ),
    "gpt-4-32k": LanguageModelInfo(
        name="gpt-4-32k",
        prompt_token_cost=0.06,
        completion_token_cost=0.12,
        max_tokens=32768,
    ),
}

OPEN_AI_EMBEDDING_MODELS = {
    "text-embedding-ada-002": EmbeddingModelInfo(
        name="text-embedding-ada-002",
        prompt_token_cost=0.0004,
        completion_token_cost=0.0,
        max_tokens=8191,
        embedding_dimensions=1536,
    ),
}

OPEN_AI_MODELS = {
    **OPEN_AI_LANGUAGE_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}


class OpenAIProvider(
    CredentialsConsumer,
    LanguageModelProvider,
    EmbeddingModelProvider,
):
    configuration_defaults = {
        "openai": {
            "retries_per_request": 10,
            "models": {
                "fast_model": {
                    "name": "gpt-3.5-turbo",
                    "max_tokens": 100,
                    "temperature": 0.9,
                },
                "smart_model": {
                    "name": "gpt-4",
                    "max_tokens": 100,
                    "temperature": 0.9,
                },
                "embedding_model": {
                    "name": "text-embedding-ada-002",
                },
            },
        },
    }

    credentials_defaults = {
        "openai": {
            "api_key": "YOUR_API_KEY",
            "use_azure": False,
            "azure_configuration": {
                "api_type": "azure",
                "api_base": "YOUR_AZURE_API_BASE",
                "api_version": "YOUR_AZURE_API_VERSION",
                "deployment_ids": {
                    "fast_model": "YOUR_FAST_LLM_MODEL_DEPLOYMENT_ID",
                    "smart_model": "YOUR_SMART_LLM_MODEL_DEPLOYMENT_ID",
                    "embedding_model": "YOUR_EMBEDDING_MODEL_DEPLOYMENT_ID",
                },
            },
        },
    }

    def __init__(
        self,
        configuration: dict,
        logger: logging.Logger,
        credentials_service: SimpleCredentialsService,
    ):
        self._configuration = configuration["openai"]
        self._logger = logger
        self._credentials = self._get_credentials(
            credentials_service,
            models=list(self._configuration["models"]),
        )
        retry_handler = _OpenAIRetryHandler(
            logger=self._logger,
            num_retries=self._configuration["retries_per_request"],
        )

        self._create_completion = retry_handler(_create_completion)
        self._create_embedding = retry_handler(_create_embedding)

    async def create_language_completion(
        self,
        model_prompt: ModelPrompt,
        model_name: str,
        completion_parser: Callable[[str], dict],
        **kwargs,
    ) -> LanguageModelResponse:
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
        content = completion_parser(response.choices[0].text)
        return LanguageModelResponse(**response_args, content=content)

    async def create_embedding(
        self,
        text: str,
        model_name: str,
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
        if model_name in OPEN_AI_EMBEDDING_MODELS:
            return EmbeddingModelResponse(
                **response_args,
                embedding=embedding_parser(response.embeddings[0]),
            )

    @staticmethod
    def _get_credentials(
        credentials_service: SimpleCredentialsService,
        models: list[str],
    ) -> dict:
        """Get the credentials for the OpenAI API."""
        raw_credentials = credentials_service.get_credentials("openai")
        parsed_credentials = {}
        for model in models:
            model_credentials = {
                "api_key": raw_credentials["api_key"],
            }
            if raw_credentials["use_azure"]:
                azure_config = raw_credentials["azure_configuration"]
                model_credentials = {
                    **model_credentials,
                    "api_base": azure_config["api_base"],
                    "api_version": azure_config["api_version"],
                    "deployment_id": azure_config["deployment_ids"][model],
                }
            parsed_credentials[model] = model_credentials

        return parsed_credentials

    def _get_completion_kwargs(self, model: str, **kwargs) -> dict:
        """Get kwargs for completion API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            The kwargs for the chat API call.

        """
        model_config = self._configuration["models"][model]
        completion_kwargs = {
            "model": model_config["name"],
            "max_tokens": kwargs.pop("max_tokens", model_config["max_tokens"]),
            "temperature": kwargs.pop("temperature", model_config["temperature"]),
            **self._credentials[model],
        }

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs.keys()}")

        return completion_kwargs

    def _get_embedding_kwargs(
        self,
        model: str,
        **kwargs,
    ) -> dict:
        """Get kwargs for embedding API call.

        Args:
            model: The model to use.
            kwargs: Keyword arguments to override the default values.

        Returns:
            The kwargs for the embedding API call.

        """
        model_config = self._configuration["models"][model]
        embedding_kwargs = {
            "model": model_config["name"],
            **self._credentials[model],
        }

        if kwargs:
            raise ValueError(f"Unexpected keyword arguments: {kwargs.keys()}")

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
    return await openai.Completion.acreate(
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
