import functools
import logging
import time
from typing import Callable, Dict, List, ParamSpec, TypeVar, overload

import openai
from openai.error import APIError, RateLimitError, Timeout

from autogpt.core.model import ModelResponse
from autogpt.core.model.embedding import EmbeddingModelInfo, EmbeddingModelResponse
from autogpt.core.model.language import LanguageModelInfo, LanguageModelResponse
from autogpt.core.planning import ModelPrompt

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


T = TypeVar("T")
P = ParamSpec("P")


class OpenAIRetryHandler:
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

    def _log_rate_limit_error(self):
        self._logger.debug(self._retry_limit_msg)
        if self._warn_user:
            self._logger.warning(self._api_key_error_msg)
            self._warn_user = False

    def _backoff(self, attempt: int):
        backoff = self._backoff_base ** (attempt + 2)
        self._logger.debug(self._backoff_msg.format(backoff=backoff))
        time.sleep(backoff)

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def _wrapped(*args, **kwargs):
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


async def create_embedding(text: str, *_, **kwargs) -> openai.Embedding:
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


async def create_completion(messages: ModelPrompt, *_, **kwargs) -> openai.Completion:
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


def parse_credentials(model: str, credentials: dict, use_azure: bool) -> Dict[str, str]:
    """Get the credentials for the OpenAI API.

    Args:
        model: The name of the model.
        credentials: The raw credentials from the credentials manager.
        use_azure: Whether to use the Azure API.

    Returns:
        The credentials.
    """
    parsed_credentials = {
        "api_key": credentials["api_key"],
    }
    if use_azure:
        azure_config = credentials["azure_configuration"]
        parsed_credentials = {
            **parsed_credentials,
            "api_base": azure_config["api_base"],
            "api_version": azure_config["api_version"],
            "deployment_id": azure_config["deployment_ids"][model],
        }
    return parsed_credentials


OpenAIChatParser = Callable[[str], dict[str, str]]
OpenAIEmbeddingParser = Callable[[List[float]], List[float]]


@overload
def parse_openai_response(
    model_name: str,
    openai_response: openai.Completion,
    content_parser: OpenAIChatParser,
) -> LanguageModelResponse:
    ...


@overload
def parse_openai_response(
    model_name: str,
    openai_response: openai.Embedding,
    content_parser: OpenAIEmbeddingParser,
) -> EmbeddingModelResponse:
    ...


def parse_openai_response(
    model_name: str,
    openai_response: openai.Completion | openai.Embedding,
    content_parser: OpenAIChatParser | OpenAIEmbeddingParser,
) -> ModelResponse:
    """Parse a response from the"""
    response_args = {
        "model_info": OPEN_AI_MODELS[model_name],
        "prompt_tokens_used": openai_response.usage.prompt_tokens,
        "completion_tokens_used": openai_response.usage.completion_tokens,
    }
    if model_name in OPEN_AI_EMBEDDING_MODELS:
        return EmbeddingModelResponse(
            **response_args,
            embedding=content_parser(openai_response.embeddings[0]),
        )
    elif model_name in OPEN_AI_LANGUAGE_MODELS:
        return LanguageModelResponse(
            **response_args,
            content=content_parser(openai_response.choices[0].text),
        )
    else:
        raise NotImplementedError(f"Unknown model {model_name}")
