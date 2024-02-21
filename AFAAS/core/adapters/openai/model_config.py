import enum
import math
import os
from typing import Callable, ClassVar, ParamSpec, TypeVar

from openai import AsyncOpenAI

from AFAAS.configs.schema import Field
from AFAAS.interfaces.adapters.chatmodel import (
    ChatModelInfo,
)
from AFAAS.interfaces.adapters.language_model import (
    AbstractPromptConfiguration,
    BaseModelProviderBudget,
    BaseModelProviderConfiguration,
    BaseModelProviderCredentials,
    BaseModelProviderSettings,
    BaseModelProviderUsage,
    ModelProviderName,
    ModelProviderService,
)
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

_T = TypeVar("_T")
_P = ParamSpec("_P")


OpenAIChatParser = Callable[[str], dict]



class OpenAIModelName(str, enum.Enum):
    """
    Enumeration of OpenAI model names.

    Attributes:
        Each enumeration represents a distinct OpenAI model name.
    """

    # ADA = "text-embedding-ada-002"
    # GPT3 = "gpt-3.5-turbo-instruct"
    # GPT3_16k = "gpt-3.5-turbo"
    # GPT3_FINE_TUNED = "gpt-3.5-turbo" + ""
    # # GPT4 = "gpt-4-0613" # TODO for tests
    # GPT4 = "gpt-3.5-turbo"
    # GPT4_32k = "gpt-4-1106-preview"

    ADA = "text-embedding-ada-002"
    GPT3 = "gpt-3.5-turbo"
    GPT3_16k = "gpt-3.5-turbo"
    GPT3_FINE_TUNED = "gpt-3.5-turbo" + ""
    GPT4 = "gpt-3.5-turbo"
    GPT4_32k = "gpt-3.5-turbo"


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
}


class OpenAIProviderConfiguration(BaseModelProviderConfiguration):
    """Configuration for OpenAI.

    Attributes:
        retries_per_request: The number of retries per request.
        maximum_retry: The maximum number of retries allowed.
        maximum_retry_before_default_function: The maximum number of retries before a default function is used.
    """

    retries_per_request: int = Field(default=10)
    maximum_retry: int = 1
    maximum_retry_before_default_function: int = 1


class OpenAIModelProviderBudget(BaseModelProviderBudget):
    """Budget configuration for the OpenAI Model Provider.

    Attributes:
        graceful_shutdown_threshold: The threshold for graceful shutdown.
        warning_threshold: The warning threshold for budget.
    """

    graceful_shutdown_threshold: float = Field(default=0.005)
    warning_threshold: float = Field(default=0.01)

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

    name : str =  "chat_model_provider"
    description : str =  "Provides access to OpenAI's API."


class OpenAIPromptConfiguration(AbstractPromptConfiguration):
    llm_model_name: str = Field()
    temperature: float = Field()


class OPEN_AI_DEFAULT_CHAT_CONFIGS:
    FAST_MODEL_4K = OpenAIPromptConfiguration(
        llm_model_name=OpenAIModelName.GPT3,
        temperature=0.9,
    )
    FAST_MODEL_16K = OpenAIPromptConfiguration(
        llm_model_name=OpenAIModelName.GPT3_16k,
        temperature=0.9,
    )
    FAST_MODEL_FINE_TUNED_4K = OpenAIPromptConfiguration(
        llm_model_name=OpenAIModelName.GPT3_FINE_TUNED,
        temperature=0.9,
    )
    MART_MODEL_8K = OpenAIPromptConfiguration(
        llm_model_name=OpenAIModelName.GPT4,
        temperature=0.9,
    )
    SMART_MODEL_32K = OpenAIPromptConfiguration(
        llm_model_name=OpenAIModelName.GPT4_32k,
        temperature=0.9,
    )
