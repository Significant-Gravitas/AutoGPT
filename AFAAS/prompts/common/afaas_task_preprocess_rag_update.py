from __future__ import annotations

import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from AFAAS.interfaces.task.task import AbstractTask

from AFAAS.interfaces.adapters import (
    AbstractLanguageModelProvider,
    AssistantChatMessageDict,
    ChatPrompt,
    CompletionModelFunction,
)
from AFAAS.interfaces.adapters.language_model import AbstractPromptConfiguration
from AFAAS.interfaces.prompts.strategy import (
    AbstractPromptStrategy,
    DefaultParsedResponse,
    PromptStrategiesConfiguration,
)
from AFAAS.lib.sdk.logger import AFAASLogger
from AFAAS.lib.utils.json_schema import JSONSchema

from langchain_core.messages import ChatMessage, HumanMessage, SystemMessage, AIMessage
LOG = AFAASLogger(name=__name__)


class PostRagTaskUpdateStrategyFunctionNames(str, enum.Enum):
    POST_RAG_UPDATE: str = "afaas_task_post_rag_update"


class PostRagTaskUpdateStrategyConfiguration(PromptStrategiesConfiguration):
    default_tool_choice: PostRagTaskUpdateStrategyFunctionNames = (
        PostRagTaskUpdateStrategyFunctionNames.POST_RAG_UPDATE
    )
    task_context_length: int = 150
    temperature: float = 0.4


class AfaasPostRagTaskUpdateStrategy(AbstractPromptStrategy):
    STRATEGY_NAME = "afaas_task_post_rag_update"
    default_configuration = PostRagTaskUpdateStrategyConfiguration()

    def __init__(
        self,
        default_tool_choice: PostRagTaskUpdateStrategyFunctionNames,
        temperature: float,
        task_context_length: int,
        count=0,
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
        self.afaas_task_post_rag_update_function: CompletionModelFunction = (
            CompletionModelFunction(
                name=PostRagTaskUpdateStrategyFunctionNames.POST_RAG_UPDATE.value,
                description="Update a task before processing",
                parameters={
                    "long_description": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        description=f"New paragraph that should be {str(self.task_context_length * 0.8)} to {str(self.task_context_length *  1.25)} words long.",
                        required=True,
                    ),
                    "task_workflow": JSONSchema(
                        type=JSONSchema.Type.STRING,
                        description=f"The workflow to be used for the task",
                        required=True,
                        enum=[
                            workflow.name
                            for workflow in self._agent.workflow_registry.workflows.values()
                        ],
                    ),
                },
            )
        )

        self._tools = [
            self.afaas_task_post_rag_update_function,
        ]

    async def build_message(
        self,
        task: AbstractTask,
        task_path: list[AbstractTask] = None,
        task_history: list[AbstractTask] = None,
        task_followup: list[AbstractTask] = None,
        task_sibblings: list[AbstractTask] = None,
        related_tasks: list[AbstractTask] = None,
        **kwargs,
    ) -> ChatPrompt:
        LOG.debug("Building prompt for task : " + await task.debug_dump_str())
        self._task: AbstractTask = task
        smart_rag_param = {
            "task_history": task_history,
            "task_sibblings": task_sibblings,
            "task_path": task_path,
            "related_tasks": related_tasks,
            "workflows": self._agent.workflow_registry.workflows,
        }

        messages = []
        messages.append(
            SystemMessage(
                await self._build_jinja_message(
                    task=task,
                    template_name=f"{self.STRATEGY_NAME}.jinja",
                    template_params=smart_rag_param,
                )
            )
        )
        messages.append(SystemMessage(self.response_format_instruction()))

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
