import enum
import functools
import math
import os
import time
from typing import Callable, ClassVar, ParamSpec, TypeVar

from openai import APIError, AsyncOpenAI, RateLimitError

from AFAAS.configs.schema import UserConfigurable
from AFAAS.interfaces.adapters.chatmodel import (
    AbstractChatMessage,
    AbstractRoleLabels,
    ChatModelInfo,
)
from AFAAS.interfaces.adapters.language_model import (
    AbstractPromptConfiguration,
    BaseModelProviderBudget,
    BaseModelProviderConfiguration,
    BaseModelProviderCredentials,
    BaseModelProviderSettings,
    BaseModelProviderUsage,
    Embedding,
    EmbeddingModelInfo,
    ModelProviderName,
    ModelProviderService,
)
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

_T = TypeVar("_T")
_P = ParamSpec("_P")


aclient = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

OpenAIEmbeddingParser = Callable[[Embedding], Embedding]
OpenAIChatParser = Callable[[str], dict]


class OpenAIRoleLabel(AbstractRoleLabels):
    USER : str = "user"
    SYSTEM : str = "system"
    ASSISTANT : str = "assistant"

    FUNCTION : str = "function"
    """May be used for the return value of function calls"""


class OpenAIChatMessage(AbstractChatMessage):
    _role_labels: ClassVar[OpenAIRoleLabel] = OpenAIRoleLabel()


class OpenAIModelName(str, enum.Enum):
    """
    Enumeration of OpenAI model names.

    Attributes:
        Each enumeration represents a distinct OpenAI model name.
    """

    # ADA = "text-embedding-ada-002"
    # GPT3 = "gpt-3.5-turbo-instruct"
    # GPT3_16k = "gpt-3.5-turbo-1106"
    # GPT3_FINE_TUNED = "gpt-3.5-turbo-1106" + ""
    # # GPT4 = "gpt-4-0613" # TODO for tests
    # GPT4 = "gpt-3.5-turbo-1106"
    # GPT4_32k = "gpt-4-1106-preview"

    ADA = "text-embedding-ada-002"
    GPT3 = "gpt-3.5-turbo-1106"
    GPT3_16k = "gpt-3.5-turbo-1106"
    GPT3_FINE_TUNED = "gpt-3.5-turbo-1106" + ""
    GPT4 = "gpt-3.5-turbo-1106"
    GPT4_32k = "gpt-3.5-turbo-1106"


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
            name=OpenAIModelName.GPT3_FINE_TUNED,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0120 / 1000,
            completion_token_cost=0.0160 / 1000,
            max_tokens=4096,
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
    ]
}


OPEN_AI_MODELS = {
    **OPEN_AI_CHAT_MODELS,
    **OPEN_AI_EMBEDDING_MODELS,
}


class OpenAIProviderConfiguration(BaseModelProviderConfiguration):
    """Configuration for OpenAI.

    Attributes:
        retries_per_request: The number of retries per request.
        maximum_retry: The maximum number of retries allowed.
        maximum_retry_before_default_function: The maximum number of retries before a default function is used.
    """

    retries_per_request: int = UserConfigurable(default=10)
    maximum_retry: int = 1
    maximum_retry_before_default_function: int = 1


class OpenAIModelProviderBudget(BaseModelProviderBudget):
    """Budget configuration for the OpenAI Model Provider.

    Attributes:
        graceful_shutdown_threshold: The threshold for graceful shutdown.
        warning_threshold: The warning threshold for budget.
    """

    graceful_shutdown_threshold: float = UserConfigurable(default=0.005)
    warning_threshold: float = UserConfigurable(default=0.01)

    total_budget: float = math.inf
    total_cost: float = 0.0
    remaining_budget: float = math.inf
    usage: BaseModelProviderUsage = BaseModelProviderUsage()


class OpenAISettings(BaseModelProviderSettings):
    """Settings for the OpenAI provider.

    Attributes:
        configuration: Configuration settings for OpenAI.
        credentials: The credentials for the model provider.
        budget: Budget settings for the model provider.
    """

    configuration: OpenAIProviderConfiguration = OpenAIProviderConfiguration()
    credentials: BaseModelProviderCredentials = BaseModelProviderCredentials()
    budget: OpenAIModelProviderBudget = OpenAIModelProviderBudget()
    chat: TypeVar = OpenAIChatMessage

    name : str =  "chat_model_provider"
    description : str =  "Provides access to OpenAI's API."


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
        num_retries: int = 10,
        backoff_base: float = 2.0,
        warn_user: bool = True,
    ):
        self._num_retries = num_retries
        self._backoff_base = backoff_base
        self._warn_user = warn_user

    def _log_rate_limit_error(self) -> None:
        LOG.trace(self._retry_limit_msg)
        if self._warn_user:
            LOG.warning(self._api_key_error_msg)
            self._warn_user = False

    def _backoff(self, attempt: int) -> None:
        backoff = self._backoff_base ** (attempt + 2)
        LOG.trace(self._backoff_msg.format(backoff=backoff))
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
                except Exception as e:
                    LOG.warning(e)
                self._backoff(attempt)

        return _wrapped


class OpenAIPromptConfiguration(AbstractPromptConfiguration):
    model_name: str = UserConfigurable()
    temperature: float = UserConfigurable()


class OPEN_AI_DEFAULT_CHAT_CONFIGS:
    FAST_MODEL_4K = OpenAIPromptConfiguration(
        model_name=OpenAIModelName.GPT3,
        temperature=0.9,
    )
    FAST_MODEL_16K = OpenAIPromptConfiguration(
        model_name=OpenAIModelName.GPT3_16k,
        temperature=0.9,
    )
    FAST_MODEL_FINE_TUNED_4K = OpenAIPromptConfiguration(
        model_name=OpenAIModelName.GPT3_FINE_TUNED,
        temperature=0.9,
    )
    MART_MODEL_8K = OpenAIPromptConfiguration(
        model_name=OpenAIModelName.GPT4,
        temperature=0.9,
    )
    SMART_MODEL_32K = OpenAIPromptConfiguration(
        model_name=OpenAIModelName.GPT4_32k,
        temperature=0.9,
    )
