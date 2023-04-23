import time

from colorama import Fore, Style
from openai.error import APIError, RateLimitError

from autogpt.api_manager import api_manager
from autogpt.config import Config

CFG = Config()


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return api_manager.embedding_create(
        text_list=[text], model="text-embedding-ada-002"
    )


def create_embedding_with_ada(text) -> list:
    """Create an embedding with text-ada-002 using the OpenAI SDK"""
    num_retries = 10
    for attempt in range(num_retries):
        backoff = 2 ** (attempt + 2)
        try:
            return api_manager.embedding_create(
                text_list=[text], model="text-embedding-ada-002"
            )
        except RateLimitError:
            pass
        except APIError as e:
            if e.http_status != 502:
                raise
            if attempt == num_retries - 1:
                raise
        if CFG.debug_mode:
            print(
                f"{Fore.RED}Error: ",
                f"API Bad gateway. Waiting {backoff} seconds...{Fore.RESET}",
            )
        time.sleep(backoff)
