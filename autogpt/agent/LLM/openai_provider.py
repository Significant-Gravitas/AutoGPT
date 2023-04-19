"""The base class for all LLM providers."""
import enum
import time

import openai
import pydantic
from colorama import Fore, Style
from openai.error import APIError, RateLimitError

from autogpt.agent.LLM.base_llm_provider import BaseLLMProvider
from autogpt.logs import logger


class ChatCompletionModels(enum.Enum):
    """Available Chat Completion Models"""

    GTP_3_5_TURBO = "gpt-3.5-turbo"
    GTP_3_5_TURBO_0301 = "gpt-3.5-turbo-0301"
    GTP_4 = "gpt-4"
    GTP_4_0314 = "gpt-4-0314"
    GTP_4_32K = "gpt-4-32k"
    GTP_4_32K_0314 = "gpt-4-32k-0314"


class EmbeddingModels(enum.Enum):
    """Available Embedding Models"""

    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TEXT_SEARCH_ADA_DOC_001 = "text-search-ada-doc-001"


class OpenAIProvider(BaseLLMProvider):
    """The base class for all LLM providers."""

    def __init__(
        self,
        api_key: str,
        max_tokens: int = 1000,
        chat_completion_model: ChatCompletionModels = ChatCompletionModels.GTP_4,
        embedding_model: EmbeddingModels = EmbeddingModels.TEXT_EMBEDDING_ADA_002,
        temperature: float = 0.9,
        num_retries: int = 10,
        debug_mode: bool = False,
    ) -> None:
        """Initialise the OpenAIProvider"""
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.chat_completion_model = chat_completion_model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.debug_mode = debug_mode
        self.num_retries = num_retries
        self.warned_user = False
        openai.api_key = api_key

    def log_debug_msg(self, message: str) -> None:
        """Log a debug message"""
        if self.debug_mode:
            logger.debug(Fore.GREEN + message + Fore.RESET)

    async def attempt_chat_completion(self, messages: list, attempt: int = 0) -> str:
        """Attempt to create a chat completion"""

        try:
            return await openai.ChatCompletion.acreate(
                model=self.chat_completion_model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except RateLimitError:
            if self.debug_mode:
                logger.error(
                    f"{Fore.RED}Error: Reached rate limit, passing... {Fore.RESET}"
                )
            if not self.warned_user:
                logger.double_check(
                    f"Please double check that you have setup a {Fore.CYAN + Style.BRIGHT}PAID{Style.RESET_ALL} OpenAI API Account. "
                    + f"You can read more here: {Fore.CYAN}https://github.com/Significant-Gravitas/Auto-GPT#openai-api-keys-configuration{Fore.RESET}"
                )
                self.warned_user = True
        except APIError as e:
            if e.http_status != 502:
                raise e
            if attempt == self.num_retries - 1:
                raise e
        back_off = 2 ** (attempt + 2)
        if self.debug_mode:
            logger.error(
                f"{Fore.RED}Error: API Bad gateway. Waiting {back_off} seconds...{Fore.RESET}"
            )
        time.sleep(back_off)

    async def create_chat_completion(self, messages: list) -> str:
        """Create a chat completion using the OpenAI API

        Args:
            messages (list[dict[str, str]]): The messages to send to the chat completion

        Returns:
            str: The response from the chat completion
        """
        response = None
        self.num_retries = 10
        if self.debug_mode:
            # Wrap debug messages in an if statement to avoid the cost of the string formatting
            logger.debug(
                Fore.GREEN
                + f"Creating chat completion with model {self.model}, temperature {self.temperature},"
                f" max_tokens {self.max_tokens}" + Fore.RESET
            )

        for attempt in range(self.num_retries):
            response = await self.attempt_chat_completion(messages, attempt)

        if response is None:
            logger.typewriter_log(
                "FAILED TO GET RESPONSE FROM OPENAI",
                Fore.RED,
                "Auto-GPT has failed to get a response from OpenAI's services. "
                + f"Try running Auto-GPT again, and if the problem the persists try running it with `{Fore.CYAN}--debug{Fore.RESET}`.",
            )
            logger.double_check()
            if self.debug_mode:
                raise RuntimeError(
                    f"Failed to get response after {self.num_retries} retries"
                )
            else:
                quit(1)

        return response.choices[0].message["content"]

    async def create_embedding_with_ada(self, text: str) -> list:
        """Create an embedding with text-ada-002 using the OpenAI SDK"""
        self.num_retries = 10
        for attempt in range(self.num_retries):
            back_off = 2 ** (attempt + 2)
            try:
                response = await openai.Embedding.acreate(
                    input=[text], model=self.embedding_model.value
                )
                return response["data"][0]["embedding"]
            except RateLimitError:
                pass
            except APIError as e:
                if e.http_status != 502:
                    raise e
                if attempt == self.num_retries - 1:
                    raise e
            if self.debug_mode:
                logger.debug(
                    f"{Fore.RED}Error: ",
                    f"API Bad gateway. Waiting {back_off} seconds...{Fore.RESET}",
                )
            time.sleep(back_off)
