from __future__ import annotations

import enum
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.task.task import AbstractTask

from AFAAS.interfaces.adapters import (
    AbstractLanguageModelProvider,
    AbstractPromptConfiguration,
    AssistantChatMessageDict,
    ChatMessage,
    ChatPrompt,
    CompletionModelFunction,
)
from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy,
    DefaultParsedResponse,
    PromptStrategiesConfiguration,
)
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.utils.json_schema import JSONSchema
from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.core.tools.tools import Tool

LOG = AFAASLogger(name=__name__)


class SearchInfoStrategyFunctionNames(str, enum.Enum):
    QUERY_LANGUAGE_MODEL: str = "query_language_model"


class SearchInfoStrategyConfiguration(PromptStrategiesConfiguration):
    default_tool_choice: SearchInfoStrategyFunctionNames = (
        SearchInfoStrategyFunctionNames.QUERY_LANGUAGE_MODEL
    )
    task_context_length: int = 300
    temperature: float = 0.4


class SearchInfo_Strategy(AbstractPromptStrategy):
    default_configuration = SearchInfoStrategyConfiguration()
    STRATEGY_NAME = "search_info"

    def __init__(
        self,
        default_tool_choice: SearchInfoStrategyFunctionNames,
        temperature: float,  # if coding 0.05
        # top_p: Optional[float] ,
        # max_tokens : Optional[int] ,
        # frequency_penalty: Optional[float], # Avoid repeting oneselfif coding 0.3
        # presence_penalty : Optional[float], # Avoid certain subjects
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
        tools: list[Tool],
        **kwargs,
    ):
        self._tools : list[CompletionModelFunction] = []

        for tool in tools:
            self._tools.append(tool.dump())

    def build_message(  self, 
                        task: AbstractTask,
                        agent : BaseAgent, 
                        query : str, 
                        reasoning : str , 
                        tools : list[Tool],
                        **kwargs) -> ChatPrompt:
        LOG.debug("Building prompt for task : " + task.debug_dump_str())
        self._task: AbstractTask = task
        smart_rag_param = {
            "task_goal": task.task_goal,
            "additional_context_description": str(task.task_context),
            "query": query,
            "reasoning": reasoning,
            "tools": tools,
        }

        messages = []
        messages.append(
            ChatMessage.system(
                self._build_jinja_message(
                    task=task,
                    template_name=f"{self.STRATEGY_NAME}.jinja",
                    template_params=smart_rag_param,
                )
            )
        )
        messages.append(ChatMessage.system(self.response_format_instruction()))

        return self.build_chat_prompt(messages=messages)

    def parse_response_content(
        self,
        response_content: AssistantChatMessageDict,
    ) -> DefaultParsedResponse:
        return self.default_parse_response_content(response_content=response_content)

    def response_format_instruction(self) -> str:
        return super().response_format_instruction()

    def get_llm_provider(self) -> AbstractLanguageModelProvider:
        return super().get_llm_provider()

    def get_prompt_config(self) -> AbstractPromptConfiguration:
        return super().get_prompt_config()
