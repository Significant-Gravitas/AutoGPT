import functools
import time
from typing import List

import openai
from colorama import Fore, Style
from openai.error import APIError, RateLimitError, Timeout

from autogpt.llm.base import ChatModelInfo, EmbeddingModelInfo, Message
from autogpt.logs import logger

OPEN_AI_CHAT_MODELS = {
    "gpt-3.5-turbo": ChatModelInfo(
        name="gpt-3.5-turbo",
        prompt_token_cost=0.002,
        completion_token_cost=0.002,
        max_tokens=4096,
    ),
    "gpt-4": ChatModelInfo(
        name="gpt-4",
        prompt_token_cost=0.03,
        completion_token_cost=0.06,
        max_tokens=8192,
    ),
    "gpt-4-32k": ChatModelInfo(
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
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}


def retry_api(
    num_retries: int = 10,
    backoff_base: float = 2.0,
    warn_user: bool = True,
):
    """Retry an OpenAI API call.

    Args:
        num_retries int: Number of retries. Defaults to 10.
        backoff_base float: Base for exponential backoff. Defaults to 2.
        warn_user bool: Whether to warn the user. Defaults to True.
    """
    retry_limit_msg = f"{Fore.RED}Error: " f"Reached rate limit, passing...{Fore.RESET}"
    api_key_error_msg = (
        f"Please double check that you have setup a "
        f"{Fore.CYAN + Style.BRIGHT}PAID{Style.RESET_ALL} OpenAI API Account. You can "
        f"read more here: {Fore.CYAN}https://docs.agpt.co/setup/#getting-an-api-key{Fore.RESET}"
    )
    backoff_msg = (
        f"{Fore.RED}Error: API Bad gateway. Waiting {{backoff}} seconds...{Fore.RESET}"
    )

    def _wrapper(func):
        @functools.wraps(func)
        def _wrapped(*args, **kwargs):
            user_warned = not warn_user
            num_attempts = num_retries + 1  # +1 for the first attempt
            for attempt in range(1, num_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except RateLimitError:
                    if attempt == num_attempts:
                        raise

                    logger.debug(retry_limit_msg)
                    if not user_warned:
                        logger.double_check(api_key_error_msg)
                        user_warned = True

                except (APIError, Timeout) as e:
                    if (e.http_status != 502) or (attempt == num_attempts):
                        raise

                backoff = backoff_base ** (attempt + 2)
                logger.debug(backoff_msg.format(backoff=backoff))
                time.sleep(backoff)

        return _wrapped

    return _wrapper


@retry_api()
def create_chat_completion(
    messages: List[Message],
    *_,
    **kwargs,
) -> openai.ChatCompletion:
    """Create a chat completion using the OpenAI API

    Args:
        messages: A list of messages to feed to the chatbot.
        kwargs: Other arguments to pass to the OpenAI API chat completion call.
    Returns:
        openai.ChatCompletion: The chat completion object.

    """
    return openai.ChatCompletion.create(
        messages=messages,
        **kwargs,
    )


@retry_api()
def create_embedding(
    text: str,
    *_,
    **kwargs,
) -> openai.Embedding:
    """Create an embedding using the OpenAI API

    Args:
        text: The text to embed.
        kwargs: Other arguments to pass to the OpenAI API embedding call.
    Returns:
        openai.Embedding: The embedding object.

    """
    return openai.Embedding.create(
        input=text,
        **kwargs,
    )
