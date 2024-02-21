from __future__ import annotations


from pydantic import BaseModel, ConfigDict
from typing import TYPE_CHECKING, Any

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from AFAAS.interfaces.adapters.chatmodel import (
    AbstractChatModelProvider,
    AbstractChatModelResponse,
)
from AFAAS.interfaces.prompts.strategy import AbstractPromptStrategy
from AFAAS.lib.sdk.logger import AFAASLogger
LOG = AFAASLogger(name=__name__)

if TYPE_CHECKING:
    from AFAAS.interfaces.agent.abstract import AbstractAgent


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra= "allow", arbitrary_types_allowed=True)

    default : AbstractChatModelProvider 
    cheap : AbstractChatModelProvider
    long_context : AbstractChatModelProvider 
    code_expert : AbstractChatModelProvider 

    default_temperature : float = 0.7

    def __getattr__(self, name):
        try : 
            return getattr(self, name)
        except AttributeError:
            LOG.notice(f"Attribute {name} not found in Configuration, returning default value")
        return self.default

class AbstractPromptManager(ABC):
    """ Helper (Potential Mixin) that manage Prompt & LLM"""

    config : LLMConfig

    def __init__(
        self,
        config : LLMConfig,
    ) -> None:
        self.config = config
        super().__init__()

    @abstractmethod
    def add_strategies(self, strategies: list[AbstractPromptStrategy]) -> None:
        pass

    @abstractmethod
    def add_strategy(self, strategy: AbstractPromptStrategy) -> None:
        pass

    @abstractmethod
    def get_strategy(self, strategy_name: str) -> AbstractPromptStrategy:
        pass

    @abstractmethod
    def set_agent(self, agent: AbstractAgent) -> None:
        pass

    @abstractmethod
    def load_strategies(self) -> list[AbstractPromptStrategy]:
        pass

    @abstractmethod
    def load_strategy(self, strategy_module_attr: type) -> None:
        pass

    @abstractmethod
    async def execute_strategy(
        self, strategy_name: str, **kwargs
    ) -> AbstractChatModelResponse:
        pass

    @abstractmethod
    async def send(self, prompt_strategy: AbstractPromptStrategy, **kwargs) -> Any:
        pass

    @abstractmethod
    async def send_to_languagemodel(
        self, prompt_strategy: AbstractPromptStrategy, **kwargs
    ) -> Any:
        pass

    @abstractmethod
    async def send_to_chatmodel(
        self, prompt_strategy: AbstractPromptStrategy, **kwargs
    ) -> AbstractChatModelResponse:
        pass

    @abstractmethod
    def get_system_info(self, strategy: AbstractPromptStrategy) -> dict[str, Any]:
        pass

    @staticmethod
    @abstractmethod
    def get_os_info() -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str | tuple[Any, ...]:
        pass
