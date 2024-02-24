import enum
import math
import os
from typing import Callable, ClassVar, ParamSpec, TypeVar

from openai import AsyncOpenAI

from autogpt.core.configuration import Field
from autogpt.interfaces.adapters.chatmodel import (
    ChatModelInfo,
)
from autogpt.interfaces.adapters.chatmodel.chatmessage import AbstractChatMessage, AbstractRoleLabels
from autogpt.interfaces.adapters.language_model import (
    AbstractPromptConfiguration,
    LanguageModelProviderBudget,
    LanguageModelProviderConfiguration,
    BaseModelProviderCredentials,
    LanguageModelProviderSettings,
    BaseModelProviderUsage,
    ModelProviderName,
    ModelProviderService,
)
import logging

LOG = logging.getLogger(__name__)

_T = TypeVar("_T")
_P = ParamSpec("_P")


class OpenAIRoleLabel(AbstractRoleLabels):
    USER : str = "user"
    SYSTEM : str = "system"
    ASSISTANT : str = "assistant"

    FUNCTION : str = "function"
    """May be used for the return value of function calls"""


class OpenAIChatMessage(AbstractChatMessage):
    _role_labels: ClassVar[OpenAIRoleLabel] = OpenAIRoleLabel()


class OpenAIModelName(str, enum.Enum):
    # TODO : Remove
    ADA = "text-embedding-ada-002"
    GPT3 = "gpt-3.5-turbo"
    GPT3_16k = "gpt-3.5-turbo"
    GPT3_FINE_TUNED = "gpt-3.5-turbo" + ""
    GPT4 = "gpt-3.5-turbo"
    GPT4_32k = "gpt-3.5-turbo"


OPEN_AI_CHAT_MODELS = {
    #TODO : USEFULL FOR AGPT BUDGET MANAGEMENT
    info.name: info
    for info in [
        ChatModelInfo(
            name=OpenAIModelName.GPT3,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0015 / 1000,
            completion_token_cost=0.002 / 1000,
            max_tokens=4096,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_16k,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.003 / 1000,
            completion_token_cost=0.004 / 1000,
            max_tokens=16384,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT3_FINE_TUNED,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.0120 / 1000,
            completion_token_cost=0.0160 / 1000,
            max_tokens=4096,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.03 / 1000,
            completion_token_cost=0.06 / 1000,
            max_tokens=8191,
        ),
        ChatModelInfo(
            name=OpenAIModelName.GPT4_32k,
            service=ModelProviderService.CHAT,
            provider_name=ModelProviderName.OPENAI,
            prompt_token_cost=0.06 / 1000,
            completion_token_cost=0.12 / 1000,
            max_tokens=32768,
        ),
    ]
}


OPEN_AI_MODELS = {
    **OPEN_AI_CHAT_MODELS,
}


class OpenAIProviderConfiguration(LanguageModelProviderConfiguration):
    ...

class OpenAIModelProviderBudget(LanguageModelProviderBudget):
    graceful_shutdown_threshold: float = Field(default=0.005)
    warning_threshold: float = Field(default=0.01)

    total_budget: float = math.inf
    total_cost: float = 0.0
    remaining_budget: float = math.inf
    usage: BaseModelProviderUsage = BaseModelProviderUsage()


class OpenAISettings(LanguageModelProviderSettings):
    configuration: OpenAIProviderConfiguration = OpenAIProviderConfiguration()
    credentials: BaseModelProviderCredentials = BaseModelProviderCredentials()
    budget: OpenAIModelProviderBudget = OpenAIModelProviderBudget()
    name : str =  "chat_model_provider"
    description : str =  "Provides access to OpenAI's API."


class OpenAIPromptConfiguration(AbstractPromptConfiguration):
    ...

class OPEN_AI_DEFAULT_CHAT_CONFIGS:
    # TODO : Can be removed
    FAST_MODEL_4K = OpenAIPromptConfiguration(
        llm_model_name=OpenAIModelName.GPT3,
        temperature=0.7,
    )
    FAST_MODEL_16K = OpenAIPromptConfiguration(
        llm_model_name=OpenAIModelName.GPT3_16k,
        temperature=0.7,
    )
    FAST_MODEL_FINE_TUNED_4K = OpenAIPromptConfiguration(
        llm_model_name=OpenAIModelName.GPT3_FINE_TUNED,
        temperature=0.7,
    )
    MART_MODEL_8K = OpenAIPromptConfiguration(
        llm_model_name=OpenAIModelName.GPT4,
        temperature=0.7,
    )
    SMART_MODEL_32K = OpenAIPromptConfiguration(
        llm_model_name=OpenAIModelName.GPT4_32k,
        temperature=0.7,
    )

