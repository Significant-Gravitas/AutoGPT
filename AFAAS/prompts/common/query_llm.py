from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.task.task import AbstractTask

from AFAAS.interfaces.adapters.chatmodel import AIMessage , HumanMessage, SystemMessage , ChatMessage

from AFAAS.interfaces.adapters import (
    AbstractLanguageModelProvider,
    AbstractPromptConfiguration,
    AssistantChatMessageDict,
    ChatPrompt,
    CompletionModelFunction,
)
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy,
    DefaultParsedResponse,
    PromptStrategiesConfiguration,
)
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.utils.json_schema import JSONSchema

LOG = AFAASLogger(name=__name__)


class QueryLLMStrategyFunctionNames(str, enum.Enum):
    pass


class QueryLLMStrategyConfiguration(PromptStrategiesConfiguration):
    default_tool_choice: str = "auto"
    task_context_length: int = 300
    temperature: float = 0.4


class QueryLLMStrategy(AbstractPromptStrategy):
    default_configuration = QueryLLMStrategyConfiguration()
    STRATEGY_NAME = "query_llm"

    def __init__(
        self,
        default_tool_choice: QueryLLMStrategyFunctionNames,
        temperature: float,
        count=0,
        exit_token: str = str(uuid.uuid4()),
        task_context_length: int = 300,
    ):
        self._count = count
        self._config = self.default_configuration
        self.default_tool_choice = default_tool_choice
        self.task_context_length = task_context_length

    def set_tools(
        self,
        task: AbstractTask,
        **kwargs,
    ):
        self._tools: list[CompletionModelFunction] = []

    async def build_message(
        self,
        task: AbstractTask,
        agent: BaseAgent,
        query: str,
        format: str,
        persona: str,
        example: str,
        **kwargs,
    ) -> ChatPrompt:
        LOG.debug("Building prompt for task : " + await task.debug_dump_str())
        self._task: AbstractTask = task
        query_llm_param = {
            "query": query,
            "format": format,
            "persona": persona,
            "example": example,
        }

        messages = []
        messages.append(
            SystemMessage(
                await self._build_jinja_message(
                    task=task,
                    template_name=f"{self.STRATEGY_NAME}.jinja",
                    template_params=query_llm_param,
                )
            )
        )

        return self.build_chat_prompt(messages=messages)

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> DefaultParsedResponse:
        return response_content["content"]

    def response_format_instruction(self) -> str:
        return super().response_format_instruction()

    def get_llm_provider(self) -> AbstractLanguageModelProvider:
        return super().get_llm_provider()

    def get_prompt_config(self) -> AbstractPromptConfiguration:
        return super().get_prompt_config()
